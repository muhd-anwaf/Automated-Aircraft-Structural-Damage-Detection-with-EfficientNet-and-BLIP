# Aircraft Damage Classification & Captioning

Automated detection and description of structural aircraft damage (cracks vs. dents) using transfer learning (EfficientNetB0) and vision-language captioning (BLIP).

---

## Problem Statement

Manual aircraft inspection is slow, expensive, and error-prone. This project builds a binary image classifier that distinguishes between **crack** and **dent** damage types from inspection photos, enabling faster triage during maintenance checks. A BLIP vision-language model then generates natural-language descriptions of each damage image.

---

## Project Structure

```
aircraft-damage-detection/
├── data/
│   └── aircraft_damage_dataset_v1/
│       ├── train/  (crack/, dent/)
│       ├── valid/  (crack/, dent/)
│       └── test/   (crack/, dent/)
├── outputs/
│   ├── figures/        ← training curves, confusion matrix, predictions
│   └── models/         ← saved best model weights
├── notebooks/
│   └── aircraft_damage_classification.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** [Roboflow Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk)
- **License:** CC BY 4.0
- ~300 training images, ~90 validation, ~50 test across 2 classes

---

## Approach

1. **EfficientNetB0** pretrained on ImageNet as the feature extractor
2. **Two-phase training:** frozen base → fine-tune top 20 layers
3. **Augmentation:** rotation, brightness, zoom, flips to combat small dataset size
4. **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
5. **BLIP** transformer for natural-language damage captioning

---

## Results

| Metric    | Crack | Dent |
|-----------|-------|------|
| Precision | ~0.92 | ~0.89 |
| Recall    | ~0.88 | ~0.93 |
| F1-Score  | ~0.90 | ~0.91 |

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/aircraft_damage_classification.ipynb
```

---

## Future Work

- Expand to multi-class (corrosion, missing fastener, paint damage)
- Grad-CAM for visual explainability
- REST API deployment (FastAPI + Docker)
- Semi-supervised learning on unlabeled inspection images

---

## Resume Bullets

- Built a two-phase transfer learning pipeline (EfficientNetB0 fine-tuning) for automated aircraft structural damage classification, achieving ~91% F1-score on a small imbalanced dataset through targeted augmentation and LR scheduling.
- Designed a modular ML evaluation framework with confusion matrix analysis, per-class precision/recall/F1, and high-confidence error visualization to surface model failure modes in a safety-critical aviation context.
- Integrated a BLIP vision-language model to generate natural-language damage descriptions from inspection images, combining classification and captioning into a unified pipeline for MRO workflows.
