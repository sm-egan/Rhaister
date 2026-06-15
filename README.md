---
license: apache-2.0
tags:
- biology
- single-cell
- perturbation
- drug-response
- genomics
datasets:
- tahoebio/EmeraldBay
- tahoebio/replogle-nadig-de-rhaister
- tahoebio/tahoe-de-rhaister
- tahoebio/parse-de-rhaister
---

# Rhaister

**Back to basics: Observed statistics are sufficient to predict drug responses**

📄 [Read the paper](https://www.biorxiv.org/content/10.64898/2026.06.09.731197v1.full.pdf)

Rhaister predicts perturbation responses in unseen (cell, perturbation) combinations from aggregated summary statistics.
It operates directly on the output of standard single-cell analysis pipelines — log2 fold change, 
Mann-Whitney U-test p-values, and expression deltas — enabling training in seconds and prediction in milliseconds.

![img](plots/Rhaister_arch.png)

## Overview

Large-scale cellular perturbation screens measure how cells respond to drugs, cytokines, and genetic perturbations across many biological contexts. Rhaister learns from observed (context, perturbation) combinations to predict responses for unseen combinations, expressing each unseen perturbation as a linear combination of observed panel perturbations.

**Key properties:**
- Operates on summary statistics (not single cells), compatible with agentic reasoning workflows
- Trains in seconds, predicts in milliseconds
- Outperforms virtual cell models (STATE) that are orders of magnitude more expensive
- Performance approaches the half-sample reference ceiling (how well the data can predict itself)
- Supports both **fewshot** (unseen combinations) and **zeroshot** (unseen cell lines) prediction

## Method

### Fewshot model

For each unseen (context *c\**, perturbation *p\**) combination, Rhaister fits an additive baseline via alternating least squares and a drug-as-linear-combination ridge regression:

1. **Additive ALS**: *y = μ + α_c + β_p* — global mean + context effect + perturbation effect
2. **Drug ridge**: expresses each unseen perturbation as a weighted combination of panel perturbations, learned per held-out context via ridge regression
3. **Calibration network**: a small MLP that maps (|FC|, -log₁₀p, log reference expression, gene-level statistics) to calibrated p-values

Three views are predicted jointly: log2 fold change (Y), Mann-Whitney p-values (P), and expression deltas (D). Final p-values are an even blend of ridge and calibration network predictions.

### Zeroshot model (Rhaister-O)

For entirely unseen cell lines with no perturbation panel, Rhaister-O uses only baseline (DMSO) expression to predict responses:

*y(c\*, p, g) = β(p,g) + γ(p,g) · x(c\*,g)*

where *x* is the baseline pseudobulk expression. See [docs/zeroshot_architecture.md](docs/zeroshot_architecture.md) for the full model description.

## Datasets

![img](plots/Rhaister_split.png)

| Dataset                                                                               | Contexts | Perturbations | Modality |
|---------------------------------------------------------------------------------------|----------|---------------|----------|
| [Tahoe-100M](https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister)              | 50 cell lines | 384 drugs × 3 doses | Small molecule drug screen |
| [Parse PBMC](https://huggingface.co/datasets/tahoebio/parse-de-rhaister)              | 12 donors × 18 cell types | 90 cytokines | Cytokine screen |
| [Replogle-Nadig](https://huggingface.co/datasets/tahoebio/replogle-nadig-de-rhaister) | 4 cell lines | ~2023 gene knockdowns | CRISPR Perturb-seq |
| Emerald Bay                                                                           | 52 cell lines | 27 drugs × multiple doses | Growth rate (scalar) |
| PRISM                                                                                 | 480 cell lines | 1448 drugs × 8 doses | Growth rate (scalar) |

## Results

### Transcriptional prediction (fewshot)

| Dataset | Splits | Pearson Δ | PR-AUC | Spearman LFC | DE overlap |
|---------|--------|-----------|--------|--------------|------------|
| Tahoe-100M | 5 | 0.87 | 0.73 | 0.81 | 0.59 |
| Parse PBMC | 1 | 0.46 | 0.40 | 0.59 | 0.28 |
| Replogle-Nadig | 4 | 0.41 | 0.08 | 0.63 | 0.14 |

### Zeroshot prediction (Rhaister-O)

| Dataset | Splits | Pearson Δ | PR-AUC | Spearman LFC | DE overlap |
|---------|--------|-----------|--------|--------------|------------|
| Tahoe-100M | 5 | 0.63 | 0.53 | 0.49 | 0.34 |

### Sensitivity prediction (R²)

| Dataset     | Splits | Rhaister | Rhaister + features |
|-------------|--------|----------|---------------------|
| Emerald Bay | 5 | 0.26 | 0.31 |
| PRISM       | 5 | 0.87 | — |

All numbers are reproducible from the code in this repository. See [CLAUDE.md](CLAUDE.md) for full reproduction instructions.

## Installation

Download the repository from the HuggingFace Hub and install it:

```python
from huggingface_hub import snapshot_download

# Download the full repo into ./Rhaister
snapshot_download("tahoebio/Rhaister", local_dir="Rhaister")
```

```bash
cd Rhaister
uv pip install -e ".[dev]"
```

<details>
<summary>Alternative: install with <code>git clone</code></summary>

```bash
git clone https://huggingface.co/tahoebio/Rhaister
cd Rhaister
uv pip install -e ".[dev]"
```

</details>

Data is loaded automatically from [HuggingFace](https://huggingface.co/tahoebio) the first time a dataset is used, and cached locally thereafter. Set `RHAISTER_DATA_ROOT` to point to a local data directory to use existing data instead.

## Usage

```bash
# Fewshot model (tahoe_5_holdout by default)
python -m rhaister.train [experiment_name]

# Other datasets
python -m rhaister.train parse_test --split parse/split_0
python -m rhaister.train replogle_test --split replogle_nadig/split_0

# Zeroshot model
HP_ZEROSHOT=1 python -m rhaister.train zs_test

# Multi-split evaluation
python -m rhaister.eval_splits [experiment_name]

# Unit tests
python -m pytest tests/ -v
```

## Repository structure

```
rhaister/                  # Python package
├── train.py               # Model: ALS + drug ridge + calnet (fewshot & zeroshot)
├── prepare_combined.py    # Data loading, evaluation (6 State metrics), logging
├── state_metrics.py       # Six metric functions from the State paper
└── eval_splits.py         # Multi-split evaluation harness

splits/                    # Dataset configs (dataset.toml) and train/test split TOMLs
tests/                     # Unit tests (57 tests)
scripts/                   # Helper scripts (baselines, sweeps, data prep)
figures/                   # Paper figure generation
docs/                      # Architecture docs, zeroshot model description
```

## Citation

If you use Rhaister in your work, please cite:

```
@article{svensson2026back,
  title={Back to basics: Observed statistics are sufficient to predict drug responses},
  author={Svensson, Valentine and Khan, Umair and Heydari, Hamed and Ubas, Airol A and Thomas, Nicole and Merico, Daniele and Goodarzi, Hani and Yu, John and Alidoust, Nima and Gandhi, Shreshth},
  journal={bioRxiv},
  pages={2026--06},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
