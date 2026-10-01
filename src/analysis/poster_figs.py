"""Re-render the four key figures at poster resolution (large fonts, high dpi).

Outputs go to <root>/poster/ by default. Nothing here changes the analysis;
it reuses the same parquet/csv/npz outputs as the regular scripts.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from dataset import load_frame
from calibration import reliability
from gallery import per_slice_error_stats, select, CONF_HI, OVERLAY

FS = 2.0  # font scale vs the screen figures


def fig_reliability_boundary(px, out, dpi):
    conf = np.maximum(px.prob, 1 - px.prob).to_numpy()
    correct = px.correct.to_numpy()
    m = (px.dist <= 5).to_numpy()
    t, ece, mce = reliability(conf[m], correct[m], 10)

    fig, ax = plt.subplots(figsize=(9, 8))
    w = t.hi - t.lo
    ax.bar(t.lo, t.acc, width=w, align="edge", color="#4c72b0",
           edgecolor="white", label="accuracy in bin")
    ax.bar(t.lo, t.conf - t.acc, bottom=t.acc, width=w, align="edge",
           color="#dd8452", alpha=0.55, edgecolor="white",
           label="gap (overconfidence)")
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=2, label="perfect calibration")
    ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1.001)
    ax.set_xlabel("confidence  max(p, 1-p)", fontsize=11 * FS)
    ax.set_ylabel("accuracy", fontsize=11 * FS)
    ax.tick_params(labelsize=9 * FS)
    ax.set_title(f"Reliability: pixels within 5 px of the true boundary\n"
                 f"ECE = {ece:.3f}   MCE = {mce:.3f}   n = {t.n.sum():,}",
                 fontsize=10 * FS)
    ax.legend(loc="upper left", fontsize=8 * FS)
    plt.tight_layout()
    plt.savefig(out / "poster_reliability_boundary.png", dpi=dpi)


def fig_histograms(px, out, dpi):
    conf = np.maximum(px.prob, 1 - px.prob).to_numpy()
    corr = px.correct.to_numpy()
    m = (px.dist <= 5).to_numpy()
    bins = np.linspace(0.5, 1, 51)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    ax.hist(conf[corr], bins, color="#4c72b0", alpha=0.75,
            label=f"correct (n={corr.sum():,})", log=True)
    ax.hist(conf[~corr], bins, color="#c44e52", alpha=0.75,
            label=f"incorrect (n={(~corr).sum():,})", log=True)
    ax.set_title("Per-pixel confidence, all pixels (log count)", fontsize=10 * FS)
    ax.set_xlabel("confidence max(p, 1-p)", fontsize=9 * FS)
    ax.tick_params(labelsize=8 * FS)
    ax.legend(fontsize=8 * FS)

    ax = axes[1]
    ax.hist(conf[m & corr], bins, color="#4c72b0", alpha=0.75, density=True,
            label=f"correct (n={(m & corr).sum():,})")
    ax.hist(conf[m & ~corr], bins, color="#c44e52", alpha=0.75, density=True,
            label=f"incorrect (n={(m & ~corr).sum():,})")
    ax.set_title("Per-pixel confidence, boundary band ±5 px (density)",
                 fontsize=10 * FS)
    ax.set_xlabel("confidence max(p, 1-p)", fontsize=9 * FS)
    ax.tick_params(labelsize=8 * FS)
    ax.legend(fontsize=8 * FS)
    plt.tight_layout()
    plt.savefig(out / "poster_histograms.png", dpi=dpi)


def draw_gallery(frame, maps, picks, mode, out, dpi):
    n = len(picks)
    fig, ax = plt.subplots(4, n, figsize=(3.4 * n, 13.2))
    for i, (_, r) in enumerate(picks.iterrows()):
        fr = frame[(frame.patient == r.patient) & (frame.slice == r.slice)].iloc[0]
        img = np.asarray(Image.open(ROOT / fr.image_path).convert("L").resize((512, 512)))
        p = maps["prob"][r.row] / 255.0
        sd = maps["std"][r.row] / 255.0 / 4
        gt = maps["gt"][r.row]
        pred = p > 0.5
        ov = np.zeros_like(gt, dtype=np.uint8)
        ov[gt & pred] = 1
        ov[~gt & pred] = 2
        ov[gt & ~pred] = 3
        n_sl = (frame.patient == r.patient).sum()
        pos = (frame[frame.patient == r.patient].slice < r.slice).sum() + 1

        ax[0, i].imshow(img, cmap="gray")
        ax[0, i].set_title(f"{r.patient}  slice {r.slice}  ({pos}/{n_sl} in stack)\n"
                           f"Dice {r.dice:.2f}   whole-slice conf {r.mean_conf:.3f}",
                           fontsize=9 * FS)
        ax[1, i].imshow(img, cmap="gray")
        ax[1, i].imshow(ov, cmap=OVERLAY, vmin=0, vmax=3, alpha=0.55,
                        interpolation="nearest")
        ax[1, i].set_title(f"TP {int((gt & pred).sum())}  FP {int(r.fp_px)}  "
                           f"FN {int(r.fn_px)} px", fontsize=9 * FS)
        ax[2, i].imshow(p, cmap="viridis", vmin=0, vmax=1)
        ax[2, i].set_title(f"predicted probability\nwrong px with conf>{CONF_HI}: "
                           f"{int(r.confident_wrong_px):,}", fontsize=9 * FS)
        ax[3, i].imshow(sd, cmap="magma", vmin=0, vmax=0.25)
        ax[3, i].set_title(f"MC-dropout std   (mean on placenta "
                           f"{r.mean_mc_std_fg:.3f})", fontsize=9 * FS)
        for k in range(4):
            ax[k, i].axis("off")

    ax[1, 0].legend(handles=[Patch(color="#2ca02c", label="TP (correct placenta)"),
                             Patch(color="#d62728", label="FP (predicted, not placenta)"),
                             Patch(color="#1f77b4", label="FN (missed placenta)")],
                    loc="lower left", fontsize=7 * FS, framealpha=0.8)
    title = ("High-confidence errors: Dice < 0.7 slices ranked by # wrong pixels "
             "with confidence > 0.95" if mode == "errors" else
             "Low-confidence correct: Dice ≥ 0.9 slices ranked by # correct pixels "
             "with confidence < 0.8")
    fig.suptitle(title, fontsize=12 * FS, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    name = ("poster_gallery_high_conf_errors.png" if mode == "errors"
            else "poster_gallery_low_conf_correct.png")
    plt.savefig(out / name, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/full_ssh_tse/pred_test_ssh_TSE")
    ap.add_argument("--outdir", default="poster")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--sequences", nargs="+", default=["ssh_TSE"])
    args = ap.parse_args()

    d = ROOT / args.pred
    out = ROOT / args.outdir
    out.mkdir(exist_ok=True)

    px = pd.read_parquet(d / "pixels.parquet")
    sl = pd.read_csv(d / "slices.csv")
    maps = np.load(d / "maps.npz")
    frame = load_frame(tuple(args.sequences), "test")

    stats_path = d / "slice_error_stats.csv"
    if not stats_path.exists():
        per_slice_error_stats(maps).to_csv(stats_path, index=False)
    sl = sl.merge(pd.read_csv(stats_path), on="row")

    fig_reliability_boundary(px, out, args.dpi)
    fig_histograms(px, out, args.dpi)
    draw_gallery(frame, maps, select(sl, "errors", args.n), "errors", out, args.dpi)
    draw_gallery(frame, maps, select(sl, "correct", args.n), "correct", out, args.dpi)
    print("saved to", out)


if __name__ == "__main__":
    main()
