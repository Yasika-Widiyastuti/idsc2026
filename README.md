# HYGD Glaucoma Detection — IDSC 2026

> **Mathematics for Hope in Healthcare**  
> International Data Science Challenge 2026 | UPM × UNAIR × UNMUL × UB

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![AUC](https://img.shields.io/badge/Test%20AUC-0.9974-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.15%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Overview

This repository contains our end-to-end pipeline for **glaucomatous optic neuropathy (GON) detection** using the [Hillel Yaffe Glaucoma Dataset (HYGD)](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/) from PhysioNet.

We fine-tuned an **EfficientNet-B3** model on retinal fundus images to classify GON+ (glaucoma) vs GON- (normal), achieving:

| Metric | Score |
|--------|-------|
| Test AUC-ROC | **0.9974** |
| Test Accuracy | **98.15%** |
| GON+ F1-score | **0.99** |
| GON- F1-score | **0.97** |

---

## Dataset

**Source:** [PhysioNet — HYGD v1.0.0](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)

**License:** PhysioNet Credentialed Health Data License

| Property | Value |
|----------|-------|
| Total images | 747 (JPG) |
| GON+ (glaucoma) | 548 (73.4%) |
| GON- (normal) | 199 (26.6%) |
| Labels file | `Labels.csv` |
| Label columns | Image Name, Patient, Label, Quality Score |

> **Note:** External labeled datasets are strictly prohibited per IDSC 2026 rules. Only the official HYGD dataset is used.

### Folder Structure Expected

```
hygd/
├── Images/
│   ├── 0_0.jpg
│   ├── 1_0.jpg
│   ├── 1_1.jpg
│   └── ...
└── Labels.csv
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Yasika-Widiyastuti/idsc2026.git
cd idsc2026
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download from PhysioNet (requires free account):
```
https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/
```
Place the downloaded folder as `./hygd/`

### 4. Run Pipeline

```bash
python hygd_pipeline.py
```

All outputs will be saved to `./outputs/`.

---

## ⚙️ Pipeline Architecture

```
Raw Data (747 images)
        │
        ▼
┌──────────────────┐
│  1. EDA          │  Class distribution, quality score analysis
│     + Filtering  │  Remove images with Quality Score < 3
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Patient-     │  Split by Patient ID (NOT by image)
│     Level Split  │  Train 70% / Val 15% / Test 15%
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. Augmentation │  RandomCrop, Flip, Rotation, ColorJitter
│     + Transforms │  ImageNet normalization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. EfficientNet │  Pretrained ImageNet → Fine-tuned
│     -B3          │  Custom classifier head (256-dim)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. Training     │  AdamW + Cosine LR Scheduler
│                  │  Weighted CrossEntropy (class imbalance)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. Evaluation   │  AUC-ROC, Accuracy, F1, Confusion Matrix
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  7. GradCAM      │  Heatmap visualization of model attention
│  Interpretability│  Overlaid on original fundus images
└──────────────────┘
```

---

## Key Design Decisions

### ① Patient-Level Split (Anti Data Leakage)
One patient can have multiple images (e.g., `1_0.jpg`, `1_1.jpg`). Splitting by image instead of patient would cause **data leakage** — the model would see the same patient in both train and test, inflating results artificially. We split strictly by `Patient` ID.

### ② Quality Filtering
Images with `Quality Score < 3` (6 images) are removed before training to reduce label noise from poor-quality fundus photos.

### ③ Class Imbalance Handling
With 73% GON+ vs 27% GON-, we apply **inverse-frequency class weights** to `CrossEntropyLoss`:
- `w(GON-) = 1.920`
- `w(GON+) = 0.676`

### ④ EfficientNet-B3 over ResNet
EfficientNet-B3 provides better accuracy-to-parameter ratio for medical imaging tasks, with compound scaling across depth, width, and resolution.

### ⑤ GradCAM for Interpretability
We use Gradient-weighted Class Activation Mapping (GradCAM) on the final convolutional layer to visualize which optic disc regions the model attends to — critical for clinical interpretability.

---

## Output Files

After running the pipeline, `./outputs/` will contain:

| File | Description |
|------|-------------|
| `best_model.pth` | Saved model weights (best val AUC) |
| `eda_plots.png` | Class distribution & quality score plots |
| `training_history.png` | Loss, accuracy, AUC curves per epoch |
| `confusion_matrix.png` | Test set confusion matrix |
| `roc_curve.png` | ROC curve with AUC score |
| `gradcam_results.png` | GradCAM heatmaps on test samples |

---

## Results

### Training Curve
Model converged well by epoch 11 (best val AUC: **0.9939**), with no significant overfitting.

### Test Set Performance

```
Classification Report:
              precision    recall  f1-score   support
        GON-       0.94      1.00      0.97        29
        GON+       1.00      0.97      0.99        79
    accuracy                           0.98       108
   macro avg       0.97      0.99      0.98       108
```

### GradCAM
Model correctly focuses on the **optic disc and cup region** — clinically aligned with how ophthalmologists diagnose glaucoma.

---

## Ethics & Limitations

### Ethical Considerations
- Dataset is de-identified and publicly available under PhysioNet license
- Model is intended as a **clinical decision support tool**, not a replacement for ophthalmologist judgment
- GradCAM heatmaps are provided to maintain transparency and clinician trust

### Limitations
- Small dataset (747 images) — results may not generalize to other populations
- Class imbalance (73:27) may bias model toward GON+ predictions
- Single institution dataset (Hillel Yaffe Medical Center, Israel) — external validation needed
- Model not validated on low-quality or non-standard fundus camera hardware

### Mathematics for Hope
Early glaucoma detection is critical — the disease is **asymptomatic until late stages**, and vision loss is irreversible. An accessible, accurate AI screening tool could extend specialist-level diagnosis to underserved regions with limited ophthalmology access, offering genuine hope through mathematics.

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

Install: `pip install -r requirements.txt`

---

## Team

| Name |
|------|
|Yasika Widiyastuti| 
|Fanti Amaliyah| 
|Dimas Ika Ningsih| 

*Affiliation: Universitas Airlangga*

---

## Citation

If you use this code or find it helpful:

```bibtex
@misc{idsc2026,
  title     = {Glaucoma Detection from Retinal Fundus Images — IDSC 2026},
  author    = {manut},
  year      = {2026},
  dataset   = {HYGD PhysioNet},
  note      = {International Data Science Challenge 2026}
}
```

**Dataset citation:**
> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*. 2000;101(23).

---

## 🔗 Links

- [HYGD Dataset on PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)
- [IDSC 2026 Official Website](https://idsc2026.github.io)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)
