"""Report text cleaning and image transforms."""

import re

import torchvision.transforms as T


def preprocess_report(text: str) -> str:
    """Clean a radiology report by stripping boilerplate and normalizing.

    Args:
        text: Raw radiology report text (findings + impressions).

    Returns:
        Cleaned report string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove common boilerplate phrases
    boilerplate = [
        r"FINAL REPORT",
        r"EXAMINATION:.*?\n",
        r"INDICATION:.*?\n",
        r"TECHNIQUE:.*?\n",
        r"COMPARISON:.*?\n",
        r"CLINICAL HISTORY:.*?\n",
        r"REASON FOR EXAM:.*?\n",
        r"WET READ:.*?\n",
        r"\bXXXX\b",  # De-identification placeholder
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Extract findings and impressions sections if present
    sections = []
    for header in ["FINDINGS:", "IMPRESSION:"]:
        match = re.search(
            rf"{header}\s*(.*?)(?=\n[A-Z]+:|$)", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            sections.append(match.group(1).strip())

    if sections:
        text = " ".join(sections)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip(".-: ")

    return text


def get_image_transform(image_size: int = 224, is_train: bool = True) -> T.Compose:
    """Build image transform pipeline.

    Args:
        image_size: Target spatial resolution.
        is_train: Whether to include data augmentation.

    Returns:
        torchvision Compose transform.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
