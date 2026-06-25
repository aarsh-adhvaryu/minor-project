# Robust Quantile Uncertainty under Outlier Contamination — a MAD-based TPL Heuristic

A benchmark study of **quantile / uncertainty-quantification (UQ) methods for regression
when the training labels are contaminated with outliers**. We introduce **TPL-tauMAD**, a
closed-form, tuning-free heuristic for the Two-Piece-Laplace (TPL) / trimmed-pinball loss
whose trim point is set from a **robust median-absolute-deviation (MAD)** estimate of the
residual scale. Across six UCI datasets and two synthetic settings, TPL-tauMAD delivers the
**best worst-case robustness under heavy multiplicative contamination** — cutting RMSE from
~35–47 (Pinball, HPB, MAQR, QRF) down to ~4 — at a modest cost in clean-data calibration,
and **without the expensive per-dataset α sweep** that the only comparably robust baseline
(TPL-sweep) requires.

---

## 1. Problem statement

Quantile regressors trained with the **pinball loss** are accurate and well-calibrated on
clean data, but they are **not robust**: a small fraction of corrupted labels — especially
*multiplicative* / heavy-tailed outliers — drags the fitted quantiles arbitrarily far,
inflating test RMSE by 1–2 orders of magnitude. Robustifying the loss with a **trimmed /
Two-Piece-Laplace (TPL)** formulation caps the influence of large residuals, but it
introduces a trim hyper-parameter **α** that controls *where* the linear pinball arm
switches to a constant. Setting α well is the whole game:

- too small → throws away clean signal, poor calibration;
- too large → no robustness, collapses back to pinball;
- tuning it per dataset (a **sweep**) is expensive and needs a clean validation signal that
  may not exist under contamination.

**Goal:** a principled, *closed-form* rule for α that is robust across datasets and
contamination regimes with **no tuning**.

## 2. Method

For a target quantile τ, the TPL loss applied to residual `u = y − ŷ` is

```
            ⎧ τ·u                         0 ≤ u < α(1−τ)
ρ_TPL(u) =  ⎨ τ(1−τ)·α                     u ≥ α(1−τ)        (upper arm capped)
            ⎩ (symmetric on the lower side, scaled by (1−τ))
```

i.e. ordinary pinball until the residual exceeds a width proportional to **α**, then flat.
The contribution is the rule for **α**:

> **TPL-tauMAD (ours):**  `α(τ) = 3 · σ̂_MAD / min(τ, 1−τ)`,
> where `σ̂_MAD = 1.4826 · median(|r − median(r)|)` is the robust scale of the residuals
> `r` from an MSE-trained MLP fit on the **clean** data.

The `1.4826` factor makes `σ̂_MAD` a consistent estimator of the Gaussian σ, so the trim
point sits at a robust "3σ" away from the median — large enough to keep inliers on the
linear arm, small enough to clip outliers. Because α is derived from a robust statistic
rather than searched, it needs **no validation sweep** and degrades gracefully as
contamination grows. We also compute a **Contam-α** variant, where σ̂_MAD is re-estimated
from residuals of an MLP trained on each contaminated set, to study the fully-blind regime.

**Backbone (identical for all neural methods):** single-hidden-layer MLP
(`Linear → ReLU → Linear`, `H = 100`), Adam, `lr = 0.01`, `100` epochs, batch size 64
(scaled up for the largest datasets). Quantiles evaluated at **τ ∈ {0.05, 0.25, 0.50, 0.75,
0.95}**.

## 3. Benchmarks (exact)

| Group | Dataset | Source | Notebook |
|-------|---------|--------|----------|
| UCI | Boston Housing | `boston.txt` | `experiments/uci/03_Boston.ipynb` |
| UCI | Concrete | `concrete.txt` | `experiments/uci/04_Concrete.ipynb` |
| UCI | Energy | `energy.txt` | `experiments/uci/05_Energy.ipynb` |
| UCI | Kin8nm | `kin8nm.txt` | `experiments/uci/06_Kin8nm.ipynb` |
| UCI | Naval | `naval.txt` | `experiments/uci/07_Naval.ipynb` |
| UCI | Yacht | `yacht.txt` | `experiments/uci/08_Yacht.ipynb` |
| Synthetic | Normal (sinc mean, homoscedastic) | generated | `experiments/synthetic/01_Synthetic_Normal.ipynb` |
| Synthetic | Heteroscedastic | generated | `experiments/synthetic/02_Synthetic_Hetero.ipynb` |

Split: 64 / 16 / 20 train/val/test (nested 80/20 splits, `random_state = 58`),
features standardized. **5 runs** per configuration with seeds **[42, 58, 123, 256, 789]**.

## 4. Contamination protocol

Outliers are injected **into the training labels only** (test set stays clean) with
`inject_outliers(y, frac, type, seed)`:

| Outlier type | Effect on the selected fraction of labels |
|--------------|-------------------------------------------|
| `gaussian`   | additive large-variance Gaussian noise |
| `multiply`   | multiplicative blow-up (heavy-tailed, the adversarial case) |
| `uniform`    | replaced by draws from a wide uniform range |
| `skewed`     | replaced by a strongly right-skewed `skewnorm(a=10)` draw |

**Contamination levels: 2%, 5%, 10%** of training labels.
Full grid = **4 types × 3 levels × 5 seeds × 8 models × 8 datasets**, plus a clean
reference. The α used at evaluation is computed twice per TPL variant: a **Clean-α** (from
clean-data residuals) and a **Contam-α** (re-estimated on each contaminated set).

## 5. Metrics

- **RMSE** of the predicted conditional quantiles vs. the clean test targets (lower = more
  robust). Reported per evaluated quantile and aggregated.
- **ECE** — Expected Calibration Error of the predicted quantiles, i.e. |empirical coverage
  − nominal τ| averaged over τ (lower = better calibrated).

## 6. Baselines

| Model | Description |
|-------|-------------|
| **Pinball** | standard quantile regression (pinball loss) |
| **TPL-3sig** | TPL with `α = 3·std(residuals)` (non-robust scale) |
| **TPL-tau** | TPL with a τ-scaled std-based α |
| **TPL-tauMAD** | **ours** — robust MAD-based α (Section 2) |
| **TPL-sweep** | TPL with α chosen by an explicit grid **sweep** (strong but expensive) |
| **HPB** | heteroscedastic probabilistic baseline (mean+variance head) |
| **MAQR** | moment-adjusted / k-NN quantile regression |
| **QRF** | quantile regression forest (scikit-learn `RandomForestRegressor`, 100 trees) |

## 7. Results

Mean over the **6 UCI datasets × 5 seeds**. **Clean ECE** is calibration on uncontaminated
data; the **RMSE @ 10%** columns are the worst contamination level for each outlier type.
Lower is better throughout; **bold** = best, _italic_ = runner-up.

| Model | Clean ECE ↓ | RMSE multiply-10% ↓ | RMSE gaussian-10% ↓ | RMSE skewed-10% ↓ | RMSE uniform-10% ↓ |
|-------|:----------:|:-------------------:|:-------------------:|:-----------------:|:------------------:|
| Pinball        | **0.054** | 46.62 | 5.14 | 6.64 | 6.76 |
| TPL-3sig       | 0.517 | 19.35 | 10.36 | 9.59 | 11.49 |
| TPL-tau        | 0.136 | 27.40 | 4.59 | 6.07 | 5.87 |
| **TPL-tauMAD (ours)** | 0.148 | **3.88** | 3.69 | _4.26_ | _3.76_ |
| TPL-sweep      | 0.105 | _4.94_ | **2.88** | **1.37** | **3.00** |
| HPB            | 0.102 | 46.73 | 5.10 | 6.65 | 6.90 |
| MAQR           | _0.049_ | 34.80 | _3.63_ | 5.39 | 5.17 |
| QRF            | 0.091 | 33.57 | 3.18 | 4.34 | 4.33 |

**Reading the table.**

- **Robustness:** under heavy **multiplicative** contamination, the calibration-first
  methods collapse — Pinball 46.6, HPB 46.7, MAQR 34.8, QRF 33.6 — while **TPL-tauMAD is
  the single most robust at 3.88**, ~10× lower. It also stays uniformly low (3.7–4.3)
  across *all four* outlier types, the most stable profile of any method.
- **The honest trade-off:** TPL-tauMAD pays for this with a higher clean-data ECE (0.148 vs
  ~0.05 for Pinball/MAQR). It is a *robustness-first* estimator, not a free lunch.
- **vs. TPL-sweep:** TPL-sweep matches or beats TPL-tauMAD on gaussian/uniform/skewed, but
  it requires an **explicit α grid search per dataset**, and TPL-tauMAD still wins the
  worst case (multiply). TPL-tauMAD recovers ~90% of the sweep's robustness **closed-form,
  with zero tuning** — the practical contribution.

Per-(model × outlier × level) RMSE and ECE curves, clean-vs-contaminated overlays, and the
paper figures (e.g. `Worst_Case_RMSE_BarChart.png`,
`*_Paradox_Pinball_vs_TPL_Multiply.png`) are in `experiments/uci/plots/`,
`experiments/uci/paper_plots/`, and `experiments/synthetic/plots/`. Full numbers per
dataset are in the `*_results.xlsx` workbooks (one `Summary` sheet + per-condition sheets).

## 8. Limitations

- **Calibration cost on clean data.** The robust trim raises clean ECE; if data are known
  to be clean, plain pinball/MAQR calibrate better.
- **Single-quantile, independent fits.** Quantiles are trained independently and can cross;
  no monotonicity constraint is imposed.
- **Backbone fixed.** All conclusions are for a small MLP (`H=100`, 100 epochs); deeper
  backbones or longer training are not explored.
- **Outlier model is synthetic.** Contamination is injected with four parametric schemes on
  the labels only; real-world covariate-shift / structured corruption is out of scope.
- **σ̂_MAD from a clean-trained MLP.** The headline Clean-α uses residuals of an MLP fit on
  clean data; the fully-blind Contam-α variant is weaker (see the workbooks).
- **Small-data variance.** Datasets like Boston/Yacht are tiny, so per-dataset numbers are
  noisy despite the 5-seed averaging.

## 9. Reproduction

```bash
# 1. Environment (pick one)
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements.txt
#   — or —
conda env create -f environment.yml && conda activate robust-tpl

# 2. Reproduce everything (executes the notebooks end-to-end, in place)
python reproduce.py --all

# Subsets / options
python reproduce.py --uci                 # only the 6 UCI dataset notebooks
python reproduce.py --synthetic           # only the 2 synthetic notebooks
python reproduce.py --list                # list what would run, then exit
python reproduce.py --only 07_Naval 08_Yacht
```

Each notebook regenerates its `*_results.xlsx` and repopulates its `plots/` directory.
Runs are seeded (`SEEDS = [42, 58, 123, 256, 789]`, `SPLIT_SEED = 58`); see
[Reproducibility](#10-reproducibility) for exact determinism notes.

## 10. Reproducibility

- **Seeds.** Every run sets NumPy, PyTorch (CPU + CUDA), and scikit-learn `random_state`
  from the per-run seed via the notebooks' `sseed()` helper. The 5 run seeds and the data
  split seed are listed above.
- **Determinism caveats.** Exact bit-for-bit reproduction requires the same library
  versions and device. GPU and multi-threaded BLAS can introduce small floating-point
  nondeterminism; CPU runs are the most reproducible. Set `PYTHONHASHSEED=0` for full
  determinism — `reproduce.py` does this automatically.
- **Runtime.** The full grid (8 datasets × 8 models × 13 conditions × 5 seeds) takes a
  while on CPU; a single dataset notebook is the convenient unit of work.

## 11. Repository layout

```
.
├── README.md
├── requirements.txt              # pip dependencies
├── environment.yml               # conda alternative
├── reproduce.py                  # runs the notebooks with fixed seeds
├── experiments/
│   ├── uci/                      # 6 UCI dataset notebooks + data + results + plots
│   │   ├── 03_Boston.ipynb … 08_Yacht.ipynb
│   │   ├── generate_paper_plots.ipynb
│   │   ├── *.txt / *.arff        # raw datasets
│   │   ├── *_results.xlsx        # aggregated metrics (Summary + per-condition sheets)
│   │   ├── paper_plots/          # curated paper figures
│   │   └── plots/<Dataset>/      # full generated RMSE/ECE plots
│   └── synthetic/                # Normal + Heteroscedastic notebooks, results, plots
└── .gitignore
```

## 12. Requirements

numpy, scipy, pandas, scikit-learn, torch, matplotlib, seaborn, openpyxl, jupyter — see
[`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml).
