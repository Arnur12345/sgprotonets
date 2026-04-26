"""Check magnification-aware anchor separability."""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from omegaconf import OmegaConf
from models.semantic_encoder import SemanticEncoder
from data.class_descriptions_breakhis import MAGNIFICATION_DESCRIPTIONS
import torch.nn.functional as F

cfg = OmegaConf.create({
    'semantic_encoder': {
        'name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'type': 'biomedclip', 'freeze': True, 'unfreeze_last_n_blocks': 0, 'max_length': 256,
    }
})
enc = SemanticEncoder(cfg)
device = torch.device('cpu')

classes = list(MAGNIFICATION_DESCRIPTIONS.keys())

for tier in ("low", "high"):
    texts = [MAGNIFICATION_DESCRIPTIONS[c][tier] for c in classes]
    with torch.no_grad():
        s_cls, _ = enc(texts, device)
        s_norm = F.normalize(s_cls, dim=-1)
        sim = torch.mm(s_norm, s_norm.t())

    print(f"\n=== {tier.upper()} magnification (40X/100X)" if tier == "low" else f"\n=== {tier.upper()} magnification (200X/400X)")
    header = " " * 20
    for c in classes:
        header += f"{c[:8]:>10s}"
    print(header)
    for i, c in enumerate(classes):
        row = f"{c:>20s}"
        for j in range(len(classes)):
            row += f"{sim[i,j].item():10.3f}"
        print(row)

    val = ['phyllodes_tumor', 'tubular_adenoma', 'papillary_carcinoma']
    print(f"\n  Val pairwise ({tier}):")
    for i, a in enumerate(val):
        for j, b in enumerate(val):
            if j > i:
                idx_a, idx_b = classes.index(a), classes.index(b)
                print(f"    {a} <-> {b}: {sim[idx_a, idx_b].item():.3f}")
