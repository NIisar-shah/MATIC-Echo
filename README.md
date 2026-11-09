# MATIC-Echo

**Measurement-Aligned Temporal and Interval-Constrained Echo Segmentation**

MATIC-Echo is a research framework for left-ventricular segmentation in echocardiographic videos with sparse end-diastolic (ED) and end-systolic (ES) annotations. It combines phase-complete dual-view sampling, a shared EchoMamba segmentation pathway, and interval-masked cardiac-trajectory constraints.

## Public research snapshot

This repository contains a curated snapshot of the original implementation used to develop MATIC-Echo. The current public files document the central technical contributions while the manuscript is under peer review.

Included now:

- the EchoMamba model implementation;
- EchoNet-Dynamic data loading and dual-phase uniform sampling;
- interval-aware trajectory consistency and boundary-aware objectives;
- the paired statistical-analysis procedure used for fixed-checkpoint comparisons.

Not included in this review-stage snapshot:

- model checkpoints or learned weights;
- complete training entry points and resume logic;
- experiment-specific configuration files;
- raw or processed datasets; and
- generated outputs, logs, or unpublished result bundles.

The complete verified training and inference package, frozen configurations, and eligible pretrained checkpoints will be released upon acceptance of the associated manuscript. Confidential access can be provided to editors and peer reviewers when required. See [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md).

## Method overview

1. **Dual-phase uniform sampling** constructs an ED-anchored view and an ES-anchored view for every video. Each view preserves uniform temporal spacing while guaranteeing that its target annotated phase lies on the sampling grid.
2. **Shared EchoMamba segmentation** applies the same model parameters to both temporal views and produces a mask sequence for every input frame.
3. **Measurement-aligned cardiac consistency** converts the soft masks into a differentiable cavity-area trajectory and constrains only the ED--ES interval with area-bracket, direction, and curvature terms.
4. **Sparse phase supervision** applies segmentation loss to annotated phases present in each view. The trajectory regularizer is used during training only and adds no inference-time parameters.

## Repository structure

```text
MATIC-Echo/
├── models/
│   └── echomamba.py
├── datasets/
│   ├── echonet_dataset.py
│   └── dual_phase_uniform_echonet.py
├── engine/
│   ├── dual_phase_uniform_loss.py
│   └── boundary_aware_loss.py
├── evaluation/
│   └── paired_test_analysis.py
├── CODE_AVAILABILITY.md
├── requirements.txt
└── README.md
```

## Data

The experiments use publicly available echocardiography datasets:

- [EchoNet-Dynamic](https://echonet.github.io/dynamic/)
- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)

The datasets are not redistributed here. Users must obtain them from the official providers and comply with their licences and terms of use.

## Environment

The research environment used Python 3.11, PyTorch 2.4.1, and CUDA 12.1 on an NVIDIA Quadro GV100. Install a PyTorch build compatible with your local CUDA driver before installing the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current snapshot is provided for method inspection and is not yet the complete end-to-end reproduction package.

## Citation

The manuscript citation and archival release identifier will be added when available. Until then, please cite the repository URL and the exact Git commit used:

```text
MATIC-Echo: Measurement-Aligned Temporal and Interval-Constrained Echo Segmentation.
https://github.com/NIisar-shah/MATIC-Echo
```

## Rights and reuse

No open-source licence has yet been assigned to this review-stage snapshot. Copyright remains with the contributors, and no permission to redistribute or create derivative works is granted by default. A release licence will be added with the complete post-acceptance package.

## Contact

Please use the GitHub issue tracker for repository-related questions.
