"""Generate VLM pathology reports for BreakHis images.

Crawls the BreaKHis_v1 directory tree, runs a medical vision-language model
on each image, and writes one row per image to a CSV file.

The script is fully resumable: re-running it skips images whose
``image_path`` already appears in the output CSV.

Output CSV columns:
    image_path   – path relative to data_dir (matches BreakHisDataset)
    report       – VLM-generated pathological description
    subtype      – canonical subtype folder name (e.g. "ductal_carcinoma")
    magnification – e.g. "40X"
    patient_id   – slide identifier parsed from filename (e.g. "14-4659")

Usage — MedGemma (recommended, RTX 3080 10 GB VRAM):
    python scripts/generate_breakhis_reports.py \\
        --data_dir data/breakhis/BreaKHis_v1/BreaKHis_v1 \\
        --output_csv data/reports/breakhis_reports_medgemma.csv \\
        --model google/medgemma-4b-it \\
        --load_in_4bit

Usage — LLaVA-1.5 / LLaVA-Next:
    python scripts/generate_breakhis_reports.py \\
        --data_dir /path/to/BreaKHis_v1 \\
        --output_csv data/reports/llava_reports.csv \\
        --model llava-hf/llava-1.5-7b-hf \\
        --load_in_4bit

Supported model IDs:
    google/medgemma-4b-it                 (recommended — medical Gemma-3)
    llava-hf/llava-1.5-7b-hf             (general LLaVA baseline)
    llava-hf/llava-v1.6-mistral-7b-hf    (LLaVA-NeXT — use --llava_next flag)

Note: MedGemma is a gated model. Run `huggingface-cli login` and accept
terms at https://huggingface.co/google/medgemma-4b-it before use.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# Project root on sys.path so data.breakhis is importable when running
# from any working directory (mirrors pattern in preprocess_iu_xray.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

from data.breakhis import VALID_MAGNIFICATIONS, parse_breakhis_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["image_path", "report", "subtype", "magnification", "patient_id"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Single unified prompt template. Magnification is context, but the
# requested features are the SAME at every scale — this ensures reports
# for the same class cluster in BiomedCLIP's semantic space regardless of
# magnification.  Output-format constraints (no markdown, no intro, strict
# sentence count) prevent the preamble / repetition issues seen with the
# per-magnification prompts.

_PROMPT_TEMPLATE = (
    "You are a breast pathologist writing a concise microscopy report. "
    "This is an H&E stained breast biopsy image at {mag} magnification. "
    "Describe what you observe: tissue architecture, cellular morphology, "
    "nuclear features (size, shape, chromatin, nucleoli), stromal characteristics, "
    "and any signs of atypia or malignancy. "
    "Write exactly 3-4 sentences. "
    "Start directly with the pathological findings. "
    "Do not use markdown, bullet points, bold text, or section headers. "
    "Do not repeat yourself."
)


def get_prompt(magnification: str) -> str:
    return _PROMPT_TEMPLATE.format(mag=magnification)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _is_medgemma(model_id: str) -> bool:
    return "medgemma" in model_id.lower()


def load_model_and_processor(
    model_id: str,
    load_in_4bit: bool,
    llava_next: bool,
    processor_id: str | None = None,
) -> tuple:
    """Load VLM and processor from HuggingFace.

    Auto-detects MedGemma models (contain 'medgemma' in the ID) and uses
    AutoModelForImageTextToText + bfloat16 for them.  All other models use
    the LLaVA family path.

    Returns:
        ``(model, processor)`` tuple ready for inference.
    """
    if _is_medgemma(model_id):
        from transformers import AutoModelForImageTextToText
        logger.info(f"Loading MedGemma model: {model_id}")
        bnb_cfg = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config=bnb_cfg,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("MedGemma loaded.")
        return model, processor

    # ── LLaVA family ────────────────────────────────────────────────────────
    kwargs: dict = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if llava_next:
        from transformers import LlavaNextForConditionalGeneration
        model_cls = LlavaNextForConditionalGeneration
    else:
        model_cls = LlavaForConditionalGeneration

    logger.info(f"Loading model: {model_id}")
    model = model_cls.from_pretrained(model_id, _fast_init=False, trust_remote_code=True, **kwargs)
    model.eval()

    proc_id = processor_id or model_id
    try:
        processor = AutoProcessor.from_pretrained(proc_id, trust_remote_code=True)
    except (ValueError, OSError):
        processor = LlavaProcessor.from_pretrained(proc_id)
    logger.info("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_prompt(processor: AutoProcessor, magnification: str) -> str:
    """Return the formatted prompt string for the given magnification.

    Tries ``processor.apply_chat_template`` (available for llava-hf models
    on recent transformers).  Falls back to the LLaVA 1.5 manual format.
    """
    question = get_prompt(magnification)

    if hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        except Exception:
            pass  # fall through to manual format

    # LLaVA 1.5 fallback
    return f"USER: <image>\n{question}\nASSISTANT:"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_report(
    model,
    processor: AutoProcessor,
    image: Image.Image,
    magnification: str,
    max_new_tokens: int,
    is_medgemma: bool = False,
) -> str:
    """Run VLM inference on a single image and return the decoded report."""
    question = get_prompt(magnification)

    if is_medgemma:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    else:
        prompt = build_prompt(processor, magnification)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Dataset crawl
# ---------------------------------------------------------------------------

def crawl(data_dir: Path, magnifications: set[str]) -> list[dict]:
    """Collect all image records from the BreaKHis_v1 folder tree.

    Returns a list of dicts with keys:
        image_path, _abs_path, subtype, magnification, patient_id
    """
    slides_root = data_dir / "histology_slides" / "breast"
    samples: list[dict] = []

    for class_name in ("benign", "malignant"):
        sob_dir = slides_root / class_name / "SOB"
        if not sob_dir.is_dir():
            logger.warning(f"Directory not found, skipping: {sob_dir}")
            continue

        for subtype_dir in sorted(sob_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue

            for patient_dir in sorted(subtype_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                for mag_dir in sorted(patient_dir.iterdir()):
                    if not mag_dir.is_dir():
                        continue
                    if mag_dir.name not in magnifications:
                        continue

                    for img_file in sorted(mag_dir.glob("*.png")):
                        parsed = parse_breakhis_filename(img_file.name)
                        if parsed is None:
                            logger.debug(f"Skipping unrecognised filename: {img_file.name}")
                            continue

                        rel_path = str(img_file.relative_to(data_dir))
                        samples.append({
                            "image_path":   rel_path,
                            "_abs_path":    img_file,       # internal only, not written to CSV
                            "subtype":      subtype_dir.name,
                            "magnification": parsed["magnification"],
                            "patient_id":   parsed["patient_id"],
                        })

    return samples


def load_done(output_csv: Path) -> set[str]:
    """Return the set of image_path values already present in the CSV."""
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["image_path"] for row in reader if "image_path" in row}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate VLM pathology reports for BreakHis images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Root of BreaKHis_v1 download (contains histology_slides/).",
    )
    p.add_argument(
        "--output_csv", default="data/reports/llava_med_reports.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--model", default="microsoft/llava-med-v1.5-mistral-7b",
        help="HuggingFace model ID.",
    )
    p.add_argument(
        "--magnifications", nargs="+",
        choices=sorted(VALID_MAGNIFICATIONS), default=None,
        help="Magnification levels to process (default: all four).",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=150,
        help="Maximum tokens to generate per report.",
    )
    p.add_argument(
        "--load_in_4bit", action="store_true",
        help="4-bit BitsAndBytes quantization (required for <=16 GB VRAM).",
    )
    p.add_argument(
        "--llava_next", action="store_true",
        help="Use LlavaNextForConditionalGeneration (for LLaVA 1.6 / NeXT models).",
    )
    p.add_argument(
        "--processor_id", default=None,
        help=(
            "HuggingFace model ID to load the processor from. "
            "Use when the model repo lacks preprocessor_config.json "
            "(e.g. microsoft/llava-med-v1.5-mistral-7b). "
            "Example: llava-hf/llava-v1.6-mistral-7b-hf"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data_dir = Path(args.data_dir).resolve()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    magnifications = (
        set(args.magnifications) if args.magnifications else set(VALID_MAGNIFICATIONS)
    )

    # --- Collect all images, subtract already-done ones ---
    logger.info("Crawling dataset...")
    all_samples = crawl(data_dir, magnifications)
    done = load_done(output_csv)
    pending = [s for s in all_samples if s["image_path"] not in done]

    logger.info(f"Total images : {len(all_samples)}")
    logger.info(f"Already done : {len(done)}")
    logger.info(f"To process   : {len(pending)}")

    if not pending:
        logger.info("Nothing to do. Exiting.")
        return

    # --- Load VLM ---
    model, processor = load_model_and_processor(
        args.model, args.load_in_4bit, args.llava_next, args.processor_id
    )
    medgemma = _is_medgemma(args.model)

    # --- Open CSV in append mode; write header only for new files ---
    is_new_file = not output_csv.exists() or output_csv.stat().st_size == 0
    csv_file = output_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore"
    )
    if is_new_file:
        writer.writeheader()

    failed: list[str] = []

    try:
        for sample in tqdm(pending, desc="Generating reports", unit="img"):
            try:
                image = Image.open(sample["_abs_path"]).convert("RGB")
                report = generate_report(
                    model, processor, image,
                    sample["magnification"], args.max_new_tokens,
                    is_medgemma=medgemma,
                )
                writer.writerow({
                    "image_path":    sample["image_path"],
                    "report":        report,
                    "subtype":       sample["subtype"],
                    "magnification": sample["magnification"],
                    "patient_id":    sample["patient_id"],
                })
                csv_file.flush()  # persist every row — safe to interrupt
            except Exception as e:
                tqdm.write(f"FAILED {sample['image_path']}: {e}")
                failed.append(sample["image_path"])
    finally:
        csv_file.close()

    logger.info(f"Done. Output written to {output_csv}")

    if failed:
        logger.warning(f"{len(failed)} images failed:")
        for path in failed:
            logger.warning(f"  {path}")


if __name__ == "__main__":
    main()
