import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GOOD_DICE = 0.8


def pixel_panels(px, axes):
    conf = np.maximum(px.prob, 1 - px.prob).to_numpy()
    corr = px.correct.to_numpy()
    bins = np.linspace(0.5, 1, 51)

    ax = axes[0]
    ax.hist(conf[corr], bins, color="#4c72b0", alpha=0.75, label=f"correct (n={corr.sum():,})", log=True)
    ax.hist(conf[~corr], bins, color="#c44e52", alpha=0.75, label=f"incorrect (n={(~corr).sum():,})", log=True)
    ax.set_title("Per-pixel confidence, all pixels (log count)", fontsize=10)
    ax.set_xlabel("confidence max(p, 1-p)"); ax.legend(fontsize=8)

    ax = axes[1]
    m = (px.dist <= 5).to_numpy()
    ax.hist(conf[m & corr], bins, color="#4c72b0", alpha=0.75, density=True, label=f"correct (n={(m & corr).sum():,})")
    ax.hist(conf[m & ~corr], bins, color="#c44e52", alpha=0.75, density=True, label=f"incorrect (n={(m & ~corr).sum():,})")
    ax.set_title("Per-pixel confidence, boundary band ±5 px (density)", fontsize=10)
    ax.set_xlabel("confidence max(p, 1-p)"); ax.legend(fontsize=8)

    ax = axes[2]
    s = px.mc_std.to_numpy()
    sb = np.linspace(0, 0.3, 61)
    ax.hist(s[m & corr], sb, color="#4c72b0", alpha=0.75, density=True, label="correct")
    ax.hist(s[m & ~corr], sb, color="#c44e52", alpha=0.75, density=True, label="incorrect")
    ax.set_title("MC-dropout std, boundary band (density)", fontsize=10)
    ax.set_xlabel("std of 20 stochastic passes"); ax.legend(fontsize=8)

    frac_wrong_conf99 = (~corr & (conf > 0.99)).sum() / (~corr).sum()
    return {"pixels_incorrect": int((~corr).sum()),
            "frac_of_errors_with_conf>0.99": frac_wrong_conf99,
            "median_conf_correct": float(np.median(conf[corr])), "median_conf_incorrect": float(np.median(conf[~corr])),
            "band_median_mcstd_correct": float(np.median(s[m & corr])), "band_median_mcstd_incorrect": float(np.median(s[m & ~corr]))}


def slice_panels(sl, axes):
    good = sl.dice >= GOOD_DICE
    stats = {}
    for ax, col, label in zip(axes, ["mean_conf", "mean_conf_band", "mean_mc_std_fg"],
                              ["whole-slice mean confidence", "mean confidence within ±5 px of true edge", "mean MC-dropout std on true placenta"]):
        a, b = sl.loc[good, col].dropna(), sl.loc[~good, col].dropna()
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        bins = np.linspace(lo, hi, 31)
        ax.hist(a, bins, color="#4c72b0", alpha=0.75, label=f"good slices, Dice ≥ {GOOD_DICE} (n={len(a)})")
        ax.hist(b, bins, color="#c44e52", alpha=0.75, label=f"poor slices, Dice < {GOOD_DICE} (n={len(b)})")
        p = mannwhitneyu(a, b).pvalue
        ax.set_title(f"Per-slice: {label}\nmedian good {a.median():.4f} vs poor {b.median():.4f}   Mann-Whitney p={p:.1e}", fontsize=9)
        ax.set_xlabel(col); ax.legend(fontsize=8)
        stats[col] = {"median_good": a.median(), "median_poor": b.median(), "p": p}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/full_ssh_tse/pred_test_ssh_TSE")
    args = ap.parse_args()
    d = ROOT / args.pred
    px = pd.read_parquet(d / "pixels.parquet")
    sl = pd.read_csv(d / "slices.csv")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    pix = pixel_panels(px, axes[0])
    sls = slice_panels(sl, axes[1])
    plt.tight_layout()
    plt.savefig(d / "fig_histograms.png", dpi=110)

    print("PIXEL LEVEL")
    for k, v in pix.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v:,}")
    print(f"\nSLICE LEVEL  (good = Dice >= {GOOD_DICE}: {(sl.dice >= GOOD_DICE).sum()} slices, poor: {(sl.dice < GOOD_DICE).sum()})")
    print(pd.DataFrame(sls).T.round(4).to_string())
    print(f"\nsaved -> {(d / 'fig_histograms.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
