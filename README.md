# Outlier-Robust Uncertainty Quantification: A MAD-Based TPL Heuristic

Minor project benchmarking **uncertainty-quantification (UQ)** methods for regression
under **outlier contamination**. It introduces a new **MAD-based Two-Piece-Laplace (TPL)
heuristic (`TPL-tauMAD`)** and compares it against established baselines on UCI regression
datasets and on controlled synthetic data, measuring predictive accuracy (**RMSE**) and
calibration (**ECE**) as data are corrupted with several outlier types and levels.

## Models compared

- **Pinball** — quantile regression with the pinball loss
- **TPL-3sig**, **TPL-tau**, **TPL-tauMAD** (new), **TPL-sweep** — Two-Piece-Laplace
  heuristic variants; the `τ` scale is estimated from MSE-MLP residuals, with `TPL-tauMAD`
  using a robust median-absolute-deviation (MAD) estimate
- **HPB** — heteroscedastic probabilistic baseline
- **MAQR** — moment/quantile regression baseline
- **QRF** — quantile regression forest (via scikit-learn `RandomForestRegressor`)

Each TPL `τ` is computed twice — a **Clean-τ** (from residuals on clean data) and a
**Contam-τ** (re-estimated per contaminated set) — to separate clean and contaminated
evaluation. Outlier types: Gaussian, Multiply, Skewed, Uniform; at 2%, 5%, and 10% levels.

## Repository layout

```
Benchmark/new_heuristic/      UCI benchmark (6 datasets, 8 models, multiple runs)
├── 03_Boston … 08_Yacht.ipynb    per-dataset experiment notebooks
├── generate_paper_plots.ipynb    builds the figures used in the writeup
├── *.txt, *.arff                 raw datasets
├── *_results.xlsx                aggregated metrics
├── paper_plots/                  curated figures
└── plots/<Dataset>/              full generated RMSE/ECE plots

Synthetic/New folder/         Synthetic experiments (Normal + Heteroscedastic)
├── 01_Synthetic_Normal.ipynb, 02_Synthetic_Hetero.ipynb
├── Synthetic_*_results.xlsx
└── plots/<Setting>/
```

## Getting started

```bash
pip install -r requirements.txt
```

Then open any notebook (e.g. `Benchmark/new_heuristic/07_Naval.ipynb`) in Jupyter and run
all cells. Each dataset notebook trains the models, injects outliers, and writes results to
the corresponding `*_results.xlsx` and `plots/` directory.

## Requirements

numpy, scipy, pandas, scikit-learn, torch, matplotlib, seaborn, openpyxl — see
[`requirements.txt`](requirements.txt).
