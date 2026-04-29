# Learning to Smell With Less: Interpretable Spiking Meta-Learning for Olfactory Data

This repository contains the implementation of **Meta-SNN**, an explainable spiking meta-learning framework for multi-task olfactory data classification. The framework combines Spiking Neural Networks (SNNs) with the Reptile meta-learning algorithm to achieve data-efficient, resource-efficient learning across heterogeneous electronic nose (e-nose) datasets. This work has been published in the proceedings of the 16th International Workshop on Computer Science and Engineering, 2026 (WCSE'2026).

#### Paper Link:
<placeholder>

## Overview

Meta-SNN addresses the challenge of learning across heterogeneous e-nose datasets with varying sensor configurations, sampling rates, and class structures. Key features:

- **Spiking Neural Network Architecture**: Two-layer fully-connected LIF neurons with direct current injection for task-adaptive spike dynamics
- **Reptile Meta-Learning**: First-order meta-learning that learns transferable temporal representations
- **Task Interleaving**: Round-robin domain interleaving to prevent catastrophic forgetting
- **Explainability**: Temporal Spike Attribution (TSA) for identifying physically meaningful sensor-time patterns
- **Robustness**: Evaluation across 18 tasks spanning four real-world e-nose datasets

## Performance Highlights

- **Average Accuracy**: 88.59% across 18 heterogeneous tasks
- **Scalability**: Single trained model adapts to any task via 15 gradient steps without retraining
- **Explainability**: TSA identifies spurious correlations (e.g., environmental proxies) invisible to accuracy metrics alone
- **Resource Efficiency**: Spiking computation for potential neuromorphic hardware deployment

## Project Structure

```
root/
├── main.py                      # Main training and evaluation script
├── evaluate_meta_model.py       # Evaluation-only script
├── config.json                  # Hyperparameter configuration
├── config_heuristics.json       # Data preprocessing config
├── requirements.txt             # Dependencies
│
├── data/                        # Datasets (4 domains, 18 tasks)
│   ├── beef/                    # Beef freshness (10 sensors, 4 classes)
│   ├── seafood/                 # Seafood quality (4 sensors, 2 classes)
│   ├── chinese_vinegar/         # Vinegar brands (8 sensors, 6 classes)
│   └── wine/                    # Wine quality (6 sensors, 3 classes)
│
├── models/
│   ├── meta_snn.py              # Meta-SNN architecture (LIF neurons)
│   └── baselines/
│       └── baseline_snn.py      # Task-specific SNN baseline
│
├── training/
│   └── reptile.py               # Reptile meta-learning algorithm
│
├── explainability/
│   └── tsa.py                   # Temporal Spike Attribution
│
├── utils/
│   ├── data_loader.py           # Unified preprocessing & loading
│   ├── adaptive_window.py       # Window adaptation utilities
│   ├── metrics.py               # Evaluation metrics
│   └── logger.py                # Experiment logging
│
├── benchmarking/                # Baseline comparisons
│   ├── lstm_model.py            # LSTM baseline
│   ├── meta_lstm_train.py       # Meta-LSTM training
│   └── train.py                 # Standard LSTM training
│
└── results/                     # Experiment outputs (auto-generated)
    └── experiment_<TIMESTAMP>/
        ├── config.json
        ├── logs/
        ├── models/
        ├── metrics/
        └── explanations/
```

## Installation

1. Create and activate a Python 3.9+ virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# or
source .venv/bin/activate           # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Train Meta-SNN

```bash
python main.py
```

With custom hyperparameters:
```bash
python main.py --epochs 50 --meta_lr 0.1 --inner_lr 0.001 --inner_steps 8
```

### Evaluate on Specific Task

```bash
python evaluate_meta_model.py --model_path results/<experiment>/models/meta_model_final.pth --task beef_cut_1
```

### Configuration

Edit `config.json` to customize:
```json
{
  "data_root": "data",
  "meta_lr": 0.1,
  "inner_lr": 0.001,
  "inner_steps": 8,
  "epochs": 50,
  "batch_size": 32,
  "hidden_dims": [128, 64],
  "adaptation_steps": 15,
  "explain_samples": 5
}
```

## Datasets

The framework is evaluated on 18 tasks across 4 domains:

| Domain | Tasks | Sensors | Classes | Source |
|--------|-------|---------|---------|--------|
| **Beef Spoilage** | 12 | 10 | 4 | [Wijaya 2018](https://doi.org/10.7910/DVN/XNFVTS) |
| **Seafood Quality** | 4 | 4 | 2 | [Wijaya 2023](https://dx.doi.org/10.21227/mddf-qn59) |
| **Chinese Vinegar** | 1 | 8 | 6 | [Weng & Fu 2025](https://doi.org/10.17632/w5fcvyj398.2) |
| **Wine Quality** | 1 | 6 | 3 | [Rodriguez Gamboa et al. 2019](https://doi.org/10.17632/vpc887d53s.3) |

**Data Preprocessing**: All datasets are standardized to [samples × 500 × 10] through:
- Stabilization removal (first 100 timesteps)
- Temporal downsampling (stride of 2)
- Min-max normalization per sensor
- Sequence padding/truncation to 500 timesteps

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dims` | [128, 64] | LIF layer dimensions |
| `epochs` | 50 | Meta-training epochs |
| `meta_lr` | 0.1 | Reptile outer loop rate |
| `inner_lr` | 0.001 | Task adaptation rate |
| `inner_steps` | 8 | Inner loop gradient steps |
| `adaptation_steps` | 15 | Steps during evaluation |
| `tasks_per_epoch` | 8 | Stochastic task sampling |

## Architecture Details

### Spiking Neural Network

**Membrane potential dynamics** (LIF neuron):
$$U[t] = \beta U[t-1] + WX[t] + S[t-1]$$

**Spike generation**:
$$S[t] = \Theta(U[t] - 1.0)$$

**Direct Encoder**: Continuous sensor values fed directly to first LIF layer (no fixed thresholds), enabling task-adaptive spike dynamics. Achieves **88.6% accuracy vs 80.7% without** (+8% improvement).

**Classification**: Temporal integration of membrane potentials:
$$\text{logit}_c = \sum_{t=1}^{500} U_c[t]$$

### Meta-Learning (Reptile)

For each meta-epoch, sample task τ and update encoder parameters:
```
θ' ← θ
for i = 1 to k do
    θ' ← θ' - α∇L_τ(θ')      # Inner loop
θ ← θ + ε(θ' - θ)             # Meta-update
```

**Task Interleaving**: Round-robin sampling across domains prevents catastrophic forgetting and encourages domain-invariant features.

### Explainability (Temporal Spike Attribution)

For each neuron i at time t', compute N-value:
$$N_i[t'] = \sum_{t \in S_i} \beta^{t'-t}$$

Backward propagate through weights to compute per-feature, per-timestep attribution to each output class.

## Results

**Per-Task Classification Accuracy** (Mean ± Std, 5 runs):

| Domain | Meta-SNN | Meta-LSTM | LSTM | Baseline SNN |
|--------|----------|-----------|------|--------------|
| Beef | 0.90±0.07 | 0.85±0.01 | 0.89±0.04 | 0.96±0.01 |
| Wine | 0.84±0.05 | 0.55±0.02 | 0.75±0.01 | 0.91±0.01 |
| Seafood | 0.84±0.20 | 0.71±0.02 | 0.68±0.02 | 0.75±0.04 |
| Vinegar | 0.74±0.06 | 0.32±0.06 | 0.53±0.35 | 0.86±0.07 |

**Average Accuracy**: Meta-SNN **88.59%** vs Meta-LSTM 58.65% vs LSTM 73.22% vs Baseline SNN 86.97%

## Output Structure

Each training run produces timestamped results:

```
results/experiment_20260113_185539/
├── config.json                     # Full configuration
├── logs/
│   ├── training.log               # Text log
│   └── meta_training.csv          # Per-epoch metrics
├── models/
│   ├── meta_model_final.pth       # Final meta-model
│   └── task_<name>_adapted.pth    # Adapted models
├── metrics/
│   ├── summary.json               # Accuracy, precision, recall, F1
│   └── confusion_matrices/        # Per-task confusion matrices
└── explanations/
    └── task_<name>/
        ├── sample_0_heatmap.png   # TSA visualization
        ├── sample_0_attribution.json
        └── feature_importance.json
```

## Reproducibility

All experiments were conducted with:
- **5 independent runs** with different random seeds
- **80/20 stratified train-test split** per task
- **32 samples per task** per epoch during meta-training
- **Identical hardware and inference conditions** for fair comparison
- **Standard deviations** reported across runs

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kumar2026learning,
  title={Learning to Smell With Less: Interpretable Spiking Meta-Learning for Olfactory Data},
  author={Kumar, Advaith Sanil and Dcosta, Austin and Achyuthan, Namita and Das, Bhaskarjyoti},
  booktitle={<placeholder>},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue in the repository.