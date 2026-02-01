"""Text encoder wrapper for radiology reports."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SemanticEncoder(nn.Module):
    """Semantic encoder wrapping a pretrained text model.

    Supports BiomedCLIP text tower, PubMedBERT, and BioBERT.

    Args:
        cfg: Model config with semantic_encoder settings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        enc_cfg = cfg.semantic_encoder

        self.max_length: int = enc_cfg.max_length

        if enc_cfg.type == "biomedclip":
            self._init_biomedclip(enc_cfg.name)
        else:
            self._init_hf_bert(enc_cfg.name)

        # Freeze/unfreeze
        if enc_cfg.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if enc_cfg.unfreeze_last_n_blocks > 0:
                self._unfreeze_last_n(enc_cfg.unfreeze_last_n_blocks)

    def _init_biomedclip(self, model_name: str) -> None:
        """Initialize from BiomedCLIP (open_clip) text tower."""
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:" + model_name
        )
        self.encoder = model.text
        self.tokenizer = open_clip.get_tokenizer("hf-hub:" + model_name)
        self._encoder_type = "open_clip"

    def _init_hf_bert(self, model_name: str) -> None:
        """Initialize from HuggingFace BERT-style model."""
        from transformers import AutoModel, AutoTokenizer

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._encoder_type = "hf"

    def _unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last N transformer layers."""
        if self._encoder_type == "open_clip":
            blocks = list(self.encoder.transformer.resblocks.children())
        else:
            blocks = list(self.encoder.encoder.layer)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def tokenize(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        """Tokenize a batch of text strings.

        Args:
            texts: List of report strings.
            device: Target device.

        Returns:
            Tokenized inputs dictionary.
        """
        if self._encoder_type == "open_clip":
            tokens = self.tokenizer(texts)
            if isinstance(tokens, torch.Tensor):
                return {"input_ids": tokens.to(device)}
            return {k: v.to(device) for k, v in tokens.items()}
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {k: v.to(device) for k, v in encoded.items()}

    def encode(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract text features.

        Args:
            texts: List of report strings.
            device: Target device for outputs.

        Returns:
            Tuple of:
                s_cls: [CLS] token features, shape (B, d_semantic).
                s_seq: Full token sequence, shape (B, seq_len, d_semantic).
        """
        tokens = self.tokenize(texts, device)

        if self._encoder_type == "open_clip":
            return self._encode_open_clip(tokens)
        else:
            return self._encode_hf(tokens)

    def _encode_open_clip(
        self, tokens: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode using open_clip text transformer."""
        # open_clip text encoder returns the final hidden states
        input_ids = tokens["input_ids"]
        x = self.encoder.token_embedding(input_ids)
        x = x + self.encoder.positional_embedding[: x.shape[1]]
        x = x.permute(1, 0, 2)  # (seq_len, B, d)
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # (B, seq_len, d)
        x = self.encoder.ln_final(x)

        # CLS = EoT token (argmax of input_ids is the EoT position)
        eot_indices = input_ids.argmax(dim=-1)
        s_cls = x[torch.arange(x.shape[0]), eot_indices]  # (B, d_semantic)
        s_seq = x  # (B, seq_len, d_semantic)
        return s_cls, s_seq

    def _encode_hf(
        self, tokens: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode using HuggingFace BERT."""
        outputs = self.encoder(**tokens)
        s_cls = outputs.last_hidden_state[:, 0]  # (B, d_semantic)
        s_seq = outputs.last_hidden_state  # (B, seq_len, d_semantic)
        return s_cls, s_seq

    def forward(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Alias for encode()."""
        return self.encode(texts, device)
