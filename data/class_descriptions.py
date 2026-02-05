"""Canonical text descriptions per pathology class for semantic anchors."""

from __future__ import annotations

# Each class has a canonical radiology-style description used as a semantic
# anchor when per-image text is unavailable at inference time.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "cardiomegaly": (
        "The cardiac silhouette is enlarged, suggesting cardiomegaly. "
        "The cardiothoracic ratio exceeds normal limits."
    ),
    "effusion": (
        "There is blunting of the costophrenic angle consistent with pleural effusion. "
        "Layering fluid is noted in the dependent portion of the hemithorax."
    ),
    "infiltrate": (
        "Patchy opacities are seen in the lung parenchyma, consistent with pulmonary infiltrates. "
        "These may represent infectious or inflammatory process."
    ),
    "mass": (
        "A focal opacity with well-defined or irregular margins is identified, "
        "suspicious for a pulmonary mass lesion."
    ),
    "nodule": (
        "A small rounded opacity is noted in the lung field, consistent with a pulmonary nodule. "
        "Follow-up imaging may be recommended."
    ),
    "pneumonia": (
        "There are areas of consolidation with air bronchograms in the lung parenchyma, "
        "consistent with pneumonia."
    ),
    "atelectasis": (
        "Linear or plate-like opacities with volume loss are present, "
        "consistent with atelectasis. Adjacent structures may show compensatory changes."
    ),
    "consolidation": (
        "Dense opacification of the lung parenchyma with air bronchograms is present, "
        "indicating consolidation."
    ),
    "pneumothorax": (
        "There is lucency in the pleural space without lung markings, "
        "consistent with pneumothorax."
    ),
    "edema": (
        "There is perihilar haziness, cephalization of vessels, and bilateral interstitial opacities "
        "consistent with pulmonary edema."
    ),
    "emphysema": (
        "The lungs are hyperinflated with flattening of the diaphragm "
        "and increased anteroposterior diameter, consistent with emphysema."
    ),
    "hernia": (
        "There is a retrocardiac opacity with air-fluid level, "
        "consistent with a hiatal hernia."
    ),
    "fibrosis": (
        "Reticular opacities with volume loss and architectural distortion are seen, "
        "suggestive of pulmonary fibrosis."
    ),
    "pleural_thickening": (
        "There is thickening of the pleural surfaces, which may be related to "
        "prior infection, inflammation, or asbestos exposure."
    ),
    "no_finding": (
        "The lungs are clear. The cardiac silhouette is within normal limits. "
        "No acute cardiopulmonary abnormality is identified."
    ),
}
