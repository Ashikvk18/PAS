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
from dataset import load_frame

CONF_HI = 0.95
OVERLAY = ListedColormap(["none", "#2ca02c", "#d62728", "#1f77b4"])


def per_slice_error_stats(maps):
    p = maps["prob"].astype(np.float32) / 255
    gt = maps["gt"]
    pred = p > 0.5
    conf = np.maximum(p, 1 - p)
    wrong = pred != gt
    return pd.DataFrame({
        "row": np.arange(len(p)),
        "wrong_px": wrong.sum((1, 2)),
        "confident_wrong_px": (wrong & (conf > CONF_HI)).sum((1, 2)),
        "fn_px": (gt & ~pred).sum((1, 2)),
        "fp_px": (~gt & pred).sum((1, 2)),
        "uncertain_px": (conf < 0.8).sum((1, 2)),
        "uncertain_correct_px": (~wrong & (conf < 0.8)).sum((1, 2)),
    })


def select(sl, mode, n):
    if mode == "errors":
        cand = sl[sl.dice < 0.7]
        return cand.sort_values("confident_wrong_px", ascending=False).head(n)
    cand = sl[sl.dice >= 0.9]
    return cand.sort_values("uncertain_correct_px", ascending=False).head(n)


def draw(sl, maps, frame, picks, mode, out):
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
        ax[0, i].set_title(f"{r.patient}  slice {r.slice}  ({pos}/{n_sl} in stack)\nDice {r.dice:.2f}   whole-slice conf {r.mean_conf:.3f}", fontsize=9)
        ax[1, i].imshow(img, cmap="gray")
        ax[1, i].imshow(ov, cmap=OVERLAY, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        ax[1, i].set_title(f"TP {int((gt & pred).sum())}  FP {int(r.fp_px)}  FN {int(r.fn_px)} px", fontsize=9)
        ax[2, i].imshow(p, cmap="viridis", vmin=0, vmax=1)
        ax[2, i].set_title(f"predicted probability\nwrong px with conf>{CONF_HI}: {int(r.confident_wrong_px):,}", fontsize=9)
        ax[3, i].imshow(sd, cmap="magma", vmin=0, vmax=0.25)
        ax[3, i].set_title(f"MC-dropout std   (mean on placenta {r.mean_mc_std_fg:.3f})", fontsize=9)
        for k in range(4):
            ax[k, i].axis("off")

    ax[1, 0].legend(handles=[Patch(color="#2ca02c", label="TP (correct placenta)"), Patch(color="#d62728", label="FP (predicted, not placenta)"),
                             Patch(color="#1f77b4", label="FN (missed placenta)")], loc="lower left", fontsize=7, framealpha=0.8)
    title = ("High-confidence errors: Dice < 0.7 slices ranked by # wrong pixels with confidence > 0.95"
             if mode == "errors" else
             "Low-confidence correct: Dice ≥ 0.9 slices ranked by # correct pixels with confidence < 0.8")
    fig.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(out, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/full_ssh_tse/pred_test_ssh_TSE")
    ap.add_argument("--mode", choices=["errors", "correct"], default="errors")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--sequences", nargs="+", default=["ssh_TSE"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    d = ROOT / args.pred
    sl = pd.read_csv(d / "slices.csv")
    maps = np.load(d / "maps.npz")
    frame = load_frame(tuple(args.sequences), args.split)

    stats_path = d / "slice_error_stats.csv"
    if not stats_path.exists():
        per_slice_error_stats(maps).to_csv(stats_path, index=False)
    sl = sl.merge(pd.read_csv(stats_path), on="row")

    picks = select(sl, args.mode, args.n)
    out = d / f"fig_gallery_{'high_conf_errors' if args.mode == 'errors' else 'low_conf_correct'}.png"
    draw(sl, maps, frame, picks, args.mode, out)

    cols = ["patient", "cohort", "slice", "dice", "mean_conf", "mean_conf_band", "mean_mc_std_fg",
            "gt_area", "fn_px", "fp_px", "confident_wrong_px", "uncertain_correct_px"]
    print(picks[cols].round(3).to_string(index=False))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
