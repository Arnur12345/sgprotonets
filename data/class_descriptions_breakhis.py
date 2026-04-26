"""Canonical pathology descriptions for BreakHis breast cancer classes.

Used as semantic anchors when per-image VLM reports are unavailable
(text_strategy = "class_anchors").  Three sets are provided:

* ``BINARY_DESCRIPTIONS`` — covers the two coarse classes used in binary
  (benign / malignant) episodic evaluation.
* ``SUBTYPE_DESCRIPTIONS`` — covers all eight fine-grained subtypes.
  Short, finding-focused descriptions matching the style of IU X-Ray
  anchors and BiomedCLIP's PubMed caption training data.
* ``MAGNIFICATION_DESCRIPTIONS`` — per-subtype per-magnification-tier
  descriptions.  Low-mag (40X/100X) emphasises tissue architecture;
  high-mag (200X/400X) emphasises cytology and nuclear detail.

Both dicts share the same key conventions as ``class_descriptions.py``
for IU X-Ray: keys are class name strings that match the labels produced
by ``BreakHisDataset``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Binary-level descriptions (2 classes)
# ---------------------------------------------------------------------------

BINARY_DESCRIPTIONS: dict[str, str] = {
    "benign": (
        "Regular glandular structures with preserved two-layer epithelium. "
        "No nuclear atypia, no mitotic figures, bland stroma."
    ),
    "malignant": (
        "Infiltrative growth with disrupted tissue architecture. "
        "Nuclear pleomorphism, prominent nucleoli, increased mitotic activity, desmoplastic stroma."
    ),
}

# ---------------------------------------------------------------------------
# Subtype-level descriptions (8 classes) — short, finding-focused
# ---------------------------------------------------------------------------

SUBTYPE_DESCRIPTIONS: dict[str, str] = {
    # ── Benign subtypes ─────────────────────────────────────────────────────
    "adenosis": (
        "Enlarged lobular acini with preserved myoepithelial layer. "
        "Bland epithelial cells, intact lobular architecture, no nuclear atypia."
    ),
    "fibroadenoma": (
        "Compressed slit-like ducts within hypocellular myxoid stroma. "
        "Biphasic pattern, uniform epithelium without atypia, intracanalicular growth."
    ),
    "phyllodes_tumor": (
        "Leaf-like stromal fronds projecting into cystic spaces. "
        "Cellular stroma with epithelial-lined clefts, expansile bulging architecture."
    ),
    "tubular_adenoma": (
        "Closely packed uniform tubular structures with minimal stroma. "
        "Double-layered tubules, no pleomorphism, well-circumscribed border."
    ),
    # ── Malignant subtypes ──────────────────────────────────────────────────
    "ductal_carcinoma": (
        "Irregular nests and cords of pleomorphic cells in desmoplastic stroma. "
        "Hyperchromatic nuclei, prominent nucleoli, brisk mitoses, reduced tubule formation."
    ),
    "lobular_carcinoma": (
        "Discohesive small cells in single-file linear cords and targetoid patterns. "
        "Round nuclei, inconspicuous nucleoli, no gland formation."
    ),
    "mucinous_carcinoma": (
        "Clusters of low-grade cells floating in abundant extracellular mucin pools. "
        "Monotonous round nuclei, fibrovascular septa, minimal atypia."
    ),
    "papillary_carcinoma": (
        "Papillary projections with fibrovascular cores within dilated ductal space. "
        "Loss of myoepithelial layer, mild nuclear atypia, arborizing architecture."
    ),
}

# ---------------------------------------------------------------------------
# Magnification-aware descriptions (8 classes × 2 tiers)
# Low-mag (40X, 100X): tissue architecture, spatial patterns
# High-mag (200X, 400X): cytology, nuclear detail, cell morphology
# ---------------------------------------------------------------------------

MAGNIFICATION_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # ── Benign subtypes ─────────────────────────────────────────────────────
    "adenosis": {
        "low": (
            "Enlarged lobular units with increased acinar density. "
            "Preserved lobular architecture, no stromal distortion, well-defined lobules."
        ),
        "high": (
            "Uniform round nuclei with fine chromatin lining acinar structures. "
            "Visible myoepithelial layer, no mitotic figures, bland cytoplasm."
        ),
    },
    "fibroadenoma": {
        "low": (
            "Well-circumscribed biphasic lesion with compressed ductal spaces. "
            "Abundant myxoid stroma, intracanalicular or pericanalicular growth pattern."
        ),
        "high": (
            "Uniform epithelial cells with small round nuclei and inconspicuous nucleoli. "
            "Bland stromal cells, no atypia, evenly spaced chromatin."
        ),
    },
    "phyllodes_tumor": {
        "low": (
            "Leaf-like stromal projections into epithelial-lined cystic spaces. "
            "Expansile bulging architecture, broad stromal fronds, exaggerated intracanalicular pattern."
        ),
        "high": (
            "Variably cellular stroma with spindle cells and mild nuclear pleomorphism. "
            "Stromal mitotic figures may be present, epithelium appears benign."
        ),
    },
    "tubular_adenoma": {
        "low": (
            "Well-circumscribed nodule of densely packed uniform tubular glands. "
            "Minimal intervening stroma, sharp border with surrounding tissue."
        ),
        "high": (
            "Double-layered tubules with inner epithelial and outer myoepithelial cells. "
            "Small round nuclei, regular chromatin, no mitotic activity."
        ),
    },
    # ── Malignant subtypes ──────────────────────────────────────────────────
    "ductal_carcinoma": {
        "low": (
            "Irregular infiltrative nests and cords disrupting normal tissue architecture. "
            "Dense desmoplastic stromal reaction, loss of lobular organization."
        ),
        "high": (
            "Pleomorphic hyperchromatic nuclei with prominent nucleoli and irregular chromatin. "
            "Increased nuclear-to-cytoplasmic ratio, atypical mitotic figures."
        ),
    },
    "lobular_carcinoma": {
        "low": (
            "Diffuse infiltration of small cells in single-file Indian-file pattern. "
            "Targetoid growth around residual ducts, minimal stromal reaction."
        ),
        "high": (
            "Small round uniform nuclei with smooth contours and inconspicuous nucleoli. "
            "Discohesive cells with intracytoplasmic lumina, low mitotic rate."
        ),
    },
    "mucinous_carcinoma": {
        "low": (
            "Large pools of pale extracellular mucin with floating cell clusters. "
            "Well-defined border, fibrovascular septa dividing mucin compartments."
        ),
        "high": (
            "Monotonous round nuclei with minimal atypia floating in mucin. "
            "Low nuclear-to-cytoplasmic ratio, rare mitoses, regular chromatin."
        ),
    },
    "papillary_carcinoma": {
        "low": (
            "Arborizing papillary fronds with fibrovascular cores filling dilated duct. "
            "Encapsulated or well-defined border, complex branching architecture."
        ),
        "high": (
            "Columnar to cuboidal cells with mild nuclear atypia lining fibrovascular stalks. "
            "Absent myoepithelial layer, nuclear stratification, occasional mitoses."
        ),
    },
}


def get_magnification_tier(magnification: str) -> str:
    """Map magnification string to 'low' or 'high' tier.

    Args:
        magnification: One of '40X', '100X', '200X', '400X'.

    Returns:
        'low' for 40X/100X, 'high' for 200X/400X.
    """
    return "low" if magnification in ("40X", "100X") else "high"


def get_mag_aware_description(class_name: str, magnification: str) -> str:
    """Get the magnification-aware description for a class.

    Falls back to the generic SUBTYPE_DESCRIPTIONS if the class or
    magnification tier is not found.

    Args:
        class_name: Subtype name (e.g. 'adenosis').
        magnification: One of '40X', '100X', '200X', '400X'.

    Returns:
        Description string.
    """
    tier = get_magnification_tier(magnification)
    mag_desc = MAGNIFICATION_DESCRIPTIONS.get(class_name, {})
    if tier in mag_desc:
        return mag_desc[tier]
    return SUBTYPE_DESCRIPTIONS.get(class_name, "")


# ---------------------------------------------------------------------------
# Unified lookup (binary + subtype) — generic (non-magnification-aware)
# ---------------------------------------------------------------------------

CLASS_DESCRIPTIONS_BREAKHIS: dict[str, str] = {
    **BINARY_DESCRIPTIONS,
    **SUBTYPE_DESCRIPTIONS,
}
