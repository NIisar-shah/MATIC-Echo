# MATIC-Echo

**Measurement-Aligned Temporal and Interval-Constrained Echo Segmentation**

MATIC-Echo is a research framework for left-ventricular segmentation in echocardiographic videos under sparse end-diastolic (ED) and end-systolic (ES) supervision. The method combines phase-complete dual-view temporal sampling with cardiac-trajectory constraints designed to improve spatial delineation, temporal consistency, and cross-dataset generalization.

## Repository status

> **Initial public scaffold.** The verified training, evaluation, and model source code is being consolidated from the experimental GPU server and will be released here after reproducibility checks. Local legacy code is intentionally excluded from this initial commit.

The current repository contains only:

- this project overview;
- a provisional Python dependency specification; and
- repository hygiene rules for excluding datasets, checkpoints, and generated outputs.

## Method overview

MATIC-Echo is built around four components:

1. **Dual-phase uniform sampling** constructs ED-anchored and ES-anchored clips so that both annotated cardiac phases contribute to training without sacrificing uniform temporal spacing.
2. **Shared EchoMamba segmentation** processes both views with the same parameters and produces a mask sequence for every input frame.
3. **Measurement-aligned cardiac consistency** converts soft masks into a differentiable cavity-area trajectory and applies interval-aware physiologic constraints.
4. **Sparse phase supervision** applies segmentation loss only at annotated phases present in each sampled clip, while temporal regularization uses the complete predicted trajectory during training.

The trajectory regularizer is used only during training and adds no inference-time parameters.

## Data

The experiments use publicly available echocardiography datasets, including:

- [EchoNet-Dynamic](https://echonet.github.io/dynamic/)
- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)

Datasets are **not redistributed** in this repository. Users must obtain them from their official providers and comply with the corresponding licenses and terms of use.

## Installation

The finalized environment file will accompany the verified code release. For the current scaffold:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA driver when GPU acceleration is required. The main experiments were developed with Python 3.11, PyTorch 2.4.1, and CUDA 12.1 on an NVIDIA Quadro GV100.

## Planned release

The verified release will add:

- model and loss implementations;
- data preparation and phase-aware sampling;
- reproducible configuration files;
- training and checkpoint-resume scripts;
- internal and external evaluation protocols;
- statistical analysis and publication-figure generation; and
- pretrained checkpoints where redistribution is permitted.

## Citation

The manuscript citation will be added after publication. Until then, please cite this repository by its GitHub URL and release version.

## Contact

For questions about the project, please use the GitHub issue tracker after the verified code release.

