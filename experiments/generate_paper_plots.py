"""Generate all paper figures from experiment result xlsx files.

Run from the repo root:  python scripts/generate_paper_plots.py
Outputs to paper_plots/ (safe to commit).

Reads ONLY the Summary sheets (mean over 5 seeds) of the synthetic
results, so figures are exactly reproducible from committed data.
"""
import os
import shutil
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────
QUANTILES = [0.01, 0.025, 0.03, 0.05, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.91, 0.93, 0.95, 0.97, 0.975, 0.99]
OUTLIER_TYPES = ["gaussian", "multiply", "uniform", "skewed"]
LEVELS = [2, 5, 10]
METHODS = [  # (dir, file tag, legend label, color, marker)
    ("pinball",    "Pinball",    "Pinball", "#1f2937", "o"),
    ("tpl_taumad", "TPL-tauMAD", "TPL",     "#DC2626", "s"),
    ("maqr",       "MAQR",       "MAQR",    "#2563EB", "^"),
    ("qrf",        "QRF",        "QRF",     "#059669", "D"),
]
DATASETS = {"normal": "Normal", "hetero": "Hetero"}
OUT = "experiments/paper_plots"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── data access ─────────────────────────────────────────────────────
def per_quantile_shift(ds_dir, ds_name, mdir, mtag, condition):
    """Seed-mean Shift-RMSE at each quantile level for one condition."""
    path = (f"experiments/synthetic/{ds_dir}/{mdir}/results/"
            f"Synthetic_{ds_name}_{mtag}_results.xlsx")
    wb = openpyxl.load_workbook(path, data_only=True)
    for row in wb["Summary"].values:
        if row[0] == condition:
            return [float(v) for v in row[1:1 + len(QUANTILES)]]
    raise KeyError(f"{condition} not found in {path}")

# ── single overlays: every type x level (24 files) ──────────────────
def single_overlay(ds_dir, ds_name, otype, level, ax=None, legend=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(QUANTILES))
    cond = f"{otype}_{level}pct"
    for mdir, mtag, label, color, marker in METHODS:
        vals = per_quantile_shift(ds_dir, ds_name, mdir, mtag, cond)
        ax.plot(x, vals, marker=marker, ms=4, lw=1.6, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in QUANTILES],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Quantile level $\tau$")
    ax.set_ylabel("Shift-RMSE")
    if legend:
        ax.legend(frameon=False, fontsize=9, ncol=2)
    if own_fig:
        fname = f"{OUT}/synth_{ds_dir}_shiftrmse_{otype}{level}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        return fname

for ds_dir, ds_name in DATASETS.items():
    for otype in OUTLIER_TYPES:
        for level in LEVELS:
            print("saved", single_overlay(ds_dir, ds_name, otype, level))

# ── 2x2 all-types panel per dataset at 10% (paper figures) ──────────
for ds_dir, ds_name in DATASETS.items():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    for ax, otype in zip(axes.ravel(), OUTLIER_TYPES):
        single_overlay(ds_dir, ds_name, otype, 10, ax=ax, legend=False)
        ax.set_title(otype.capitalize(), fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    fname = f"{OUT}/synth_{ds_dir}_shiftrmse_all10.png"
    plt.savefig(fname)
    plt.close()
    print("saved", fname)

# ── copy the six qcurve panels used in the paper ────────────────────
QCURVE = {
    "pn1": "normal/pinball",   "mn1": "normal/maqr",   "tn1": "normal/tpl_taumad",
    "ph1": "hetero/pinball",   "mh1": "hetero/maqr",   "th1": "hetero/tpl_taumad",
}
for dst, sub in QCURVE.items():
    src = f"experiments/synthetic/{sub}/plots/qcurve_gaussian_10pct_avg.png"
    shutil.copy(src, f"{OUT}/{dst}.png")
    print(f"copied {dst}.png <- {src}")

print("\nAll paper figures in", OUT)
