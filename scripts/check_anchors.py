"""Check BiomedCLIP class anchor separability."""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from omegaconf import OmegaConf
from models.semantic_encoder import SemanticEncoder
from data.class_descriptions_breakhis import SUBTYPE_DESCRIPTIONS
import torch.nn.functional as F

cfg = OmegaConf.create({
    'semantic_encoder': {
        'name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'type': 'biomedclip', 'freeze': True, 'unfreeze_last_n_blocks': 0, 'max_length': 256,
    }
})
enc = SemanticEncoder(cfg)
device = torch.device('cpu')

classes = list(SUBTYPE_DESCRIPTIONS.keys())
texts = [SUBTYPE_DESCRIPTIONS[c] for c in classes]
with torch.no_grad():
    s_cls, _ = enc(texts, device)
    s_norm = F.normalize(s_cls, dim=-1)
    sim = torch.mm(s_norm, s_norm.t())

print("Class anchor cosine similarities:")
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
print("\nVal class pairwise similarities:")
for i, a in enumerate(val):
    for j, b in enumerate(val):
        if j > i:
            idx_a, idx_b = classes.index(a), classes.index(b)
            print(f"  {a} <-> {b}: {sim[idx_a, idx_b].item():.3f}")

print("\nVal vs Train similarities:")
train = ['adenosis', 'fibroadenoma', 'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma']
for v in val:
    sims = []
    for t in train:
        idx_v, idx_t = classes.index(v), classes.index(t)
        sims.append(f"{t[:8]}={sim[idx_v, idx_t].item():.3f}")
    print(f"  {v:>20s}: {', '.join(sims)}")
