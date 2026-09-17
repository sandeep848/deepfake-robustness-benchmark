# Deepfake Detection Under Real-World Degradation

[![CI](https://github.com/sandeep848/Deepfake-Detection-Using-EfficientnetB0/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeep848/Deepfake-Detection-Using-EfficientnetB0/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible research pipeline for measuring how compression, resizing, blur and combined social-media transformations affect lightweight deepfake detectors. The project compares clean and robustness-aware training with leakage-safe splits, calibration metrics and bootstrap confidence intervals.

## Research question

How much do realistic media transformations reduce the performance of a lightweight deepfake detector, and can degradation-aware training reduce that loss?

## Highlights

- EfficientNet-B0 baseline and spatial-frequency dual-branch model
- FaceForensics++ training and optional Celeb-DF cross-dataset evaluation
- JPEG, resize, blur, motion blur, noise and compound degradation tiers
- Video-grouped splits that prevent frames from the same source leaking across partitions
- ROC-AUC, ECE, bootstrap confidence intervals and paired comparisons
- Grad-CAM diagnostics with explicit interpretability limitations
- Streamlit inference interface, automated tests and GitHub Actions CI

## Architecture

~~~mermaid
flowchart TD
    A["Aligned face crop"] --> B["RGB EfficientNet-B0"]
    A --> C["Multi-scale SRM filters"]
    C --> D["Frequency encoder"]
    B --> E["Cross-attention fusion"]
    D --> E
    E --> F["Binary prediction"]
~~~

## Installation

~~~bash
git clone https://github.com/sandeep848/Deepfake-Detection-Using-EfficientnetB0.git
cd Deepfake-Detection-Using-EfficientnetB0
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,app,face_extraction]"
~~~

On Windows, activate the environment with `.venv\Scripts\activate`.

## Workflow

### 1. Prepare data

Download FaceForensics++ through the official access process, then create aligned face crops and grouped manifests:

~~~bash
deepfake-extract
~~~

Use the sanity option for a small pipeline check before full extraction.

### 2. Train

~~~bash
deepfake-train --model efficientnet_b0 --strategy clean --epochs 30 --batch_size 32
deepfake-train --model efficientnet_b0 --strategy degradation --epochs 30 --batch_size 32
~~~

### 3. Evaluate

~~~bash
deepfake-evaluate --mode comparative --bootstraps 1000
~~~

The comparative evaluation generates per-tier discrimination, calibration and uncertainty results. Report generated measurements as experimental results; do not replace missing runs with estimated values.

### 4. Interpret and inspect

~~~bash
deepfake-gradcam \
  --checkpoint outputs/efficientnet_b0_clean/best_model.pt \
  --image path/to/face.jpg \
  --output outputs/gradcam_example.png
~~~

~~~bash
deepfake-app
~~~

Grad-CAM is a qualitative diagnostic. It does not identify a ground-truth manipulation mask and must not be treated as causal evidence.

## Repository structure

~~~text
src/
├── configs/          # Training and degradation configuration
├── datasets/         # Download, extraction and leakage-safe manifests
├── degradations/     # Clean and robustness-aware transforms
├── evaluation/       # Metrics, comparative evaluation and Grad-CAM
├── models/           # EfficientNet and spatial-frequency models
└── training/         # Training engine
tests/                # Unit, leakage and smoke tests
.github/workflows/    # Continuous integration
~~~

## Reproducibility safeguards

- Source-video grouping is enforced across train, validation and test sets.
- Tiny datasets fail instead of silently creating overlapping splits.
- Validation data is used for threshold selection; test data remains isolated.
- Seeds, configurations and generated reports are stored with each experiment.
- Confidence intervals are computed with non-parametric bootstrapping.

## Tests

~~~bash
pytest
~~~

## Limitations

- Performance depends on face detection quality and dataset coverage.
- Results on FaceForensics++ do not guarantee performance on unseen manipulation methods.
- Platform transformations evolve and should be re-benchmarked periodically.
- This system is a research prototype and should not independently make moderation decisions.

## Documentation

- [Product requirements](PRD.md)
- [Implementation plan](PRD_v3_Implementation_Plan.md)
- [Final report](Phase_5_Final_Report.md)

## License

Distributed under the [MIT License](LICENSE).
