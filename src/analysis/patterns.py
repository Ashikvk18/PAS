import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from scipy.ndimage import label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_frame

POOR = 0.8


def add_slice_features(sl, frame, maps):
    order = frame.groupby("patient").slice.rank(method="first").astype(int)
    frame = frame.assign(pos=order.values, n_slices=frame.groupby("patient").slice.transform("count").values)
    frame["rel_pos"] = (frame.pos - 1) / (frame.n_slices - 1)
    frame["edge_dist"] = np.minimum(frame.pos - 1, frame.n_slices - frame.pos)
    sl = sl.merge(frame[["patient", "slice", "pos", "n_slices", "rel_pos", "edge_dist"]], on=["patient", "slice"])

    gt = maps["gt"]
    comps, elong = [], []
    for g in gt:
        _, n = label(g)
        comps.append(n)
        ys, xs = np.nonzero(g)
        cov = np.cov(np.vstack([ys, xs]))
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong.append(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    sl["gt_components"] = comps
    sl["gt_elongation"] = elong
    sl["gt_area_rel"] = sl.gt_area / sl.groupby("patient").gt_area.transform("max")
    sl["poor"] = sl.dice < POOR
    return sl


def pixel_by_distance(px):
    bins = [0, 1, 2, 3, 5, 8, 12, 20, 40, 1e9]
    labels = ["0-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20-40", ">40"]
    px = px.assign(dbin=pd.cut(px.dist, bins, labels=labels, include_lowest=True),
                   conf=np.maximum(px.prob, 1 - px.prob))
    g = px.groupby("dbin", observed=True).agg(n=("correct", "size"), acc=("correct", "mean"),
                                              conf=("conf", "mean"), mc_std=("mc_std", "mean"))
    g["gap"] = g.conf - g.acc
    g["share_of_errors"] = px[~px.correct].groupby("dbin", observed=True).size() / (~px.correct).sum()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/full_ssh_tse/pred_test_ssh_TSE")
    ap.add_argument("--sequences", nargs="+", default=["ssh_TSE"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    d = ROOT / args.pred

    sl = pd.read_csv(d / "slices.csv").merge(pd.read_csv(d / "slice_error_stats.csv"), on="row")
    px = pd.read_parquet(d / "pixels.parquet")
    maps = np.load(d / "maps.npz")
    frame = load_frame(tuple(args.sequences), args.split)
    sl = add_slice_features(sl, frame, maps)
    sl.to_csv(d / "slices_with_features.csv", index=False)

    print(f"=== A. What predicts a poor slice (Dice < {POOR})?  n={len(sl)}, poor={sl.poor.sum()} ===")
    feats = {"edge_dist": "distance from stack edge (slices)", "rel_pos": "relative position in stack (0=first,1=last)",
             "gt_area": "placenta area (px)", "gt_area_rel": "area relative to patient max",
             "gt_elongation": "elongation (major/minor axis)", "gt_components": "# separate GT components",
             "mean_mc_std_fg": "MC-dropout std on placenta", "mean_conf_band": "confidence at true boundary",
             "frac_uncertain": "fraction of pixels with conf<0.7"}
    rows = []
    for f, desc in feats.items():
        rho, p = spearmanr(sl[f], sl.dice, nan_policy="omit")
        a, b = sl.loc[~sl.poor, f].dropna(), sl.loc[sl.poor, f].dropna()
        rows.append({"feature": desc, "spearman_vs_dice": rho, "p": p, "median_good": a.median(), "median_poor": b.median(),
                     "mw_p": mannwhitneyu(a, b).pvalue})
    A = pd.DataFrame(rows).sort_values("spearman_vs_dice", key=abs, ascending=False)
    print(A.round(4).to_string(index=False))

    print("\n=== B. Edge-of-volume effect: Dice by distance from stack edge ===")
    sl["edge_bin"] = pd.cut(sl.edge_dist, [-1, 0, 1, 2, 4, 100], labels=["first/last", "2nd", "3rd", "4th-5th", "interior"])
    B = sl.groupby("edge_bin", observed=True).agg(n=("dice", "size"), dice_mean=("dice", "mean"), dice_median=("dice", "median"),
                                                  poor_rate=("poor", "mean"), gt_area=("gt_area", "median"))
    print(B.round(3).to_string())

    print("\n=== C. Cohort (scanner) effect ===")
    C = sl.groupby("cohort").agg(n=("dice", "size"), patients=("patient", "nunique"), dice=("dice", "mean"), poor_rate=("poor", "mean"),
                                 conf=("mean_conf", "mean"), mc_std_fg=("mean_mc_std_fg", "mean"), frac_uncertain=("frac_uncertain", "mean"))
    print(C.round(4).to_string())
    for col in ["dice", "mean_mc_std_fg", "frac_uncertain"]:
        p = mannwhitneyu(sl.loc[sl.cohort == "sub", col], sl.loc[sl.cohort == "v", col]).pvalue
        print(f"  sub vs v  {col:16s} Mann-Whitney p = {p:.3g}")

    print("\n=== D. Patient-level: is error concentrated in a few patients? ===")
    D = sl.groupby("patient").agg(n=("dice", "size"), dice=("dice", "mean"), poor=("poor", "sum"), conf_wrong=("confident_wrong_px", "sum")).sort_values("dice")
    D["share_conf_wrong"] = D.conf_wrong / D.conf_wrong.sum()
    print(D.round(3).to_string())
    top3 = D.share_conf_wrong.nlargest(3).sum()
    print(f"  -> worst 3 of 20 patients account for {top3:.0%} of all confidently-wrong pixels")

    print("\n=== E. Pixel level: accuracy, confidence and errors by distance from the true boundary ===")
    E = pixel_by_distance(px)
    print(E.round(4).to_string())
    print(f"  -> pixels within 5 px of the edge are {(px.dist <= 5).mean():.1%} of the image but hold {E.share_of_errors[:4].sum():.0%} of the errors")

    print("\n=== F. FN vs FP: which way does the model err? ===")
    print(f"  total FN px {sl.fn_px.sum():,}  |  total FP px {sl.fp_px.sum():,}  |  FN/FP ratio {sl.fn_px.sum()/sl.fp_px.sum():.2f}")
    print(f"  predicted/true area ratio: median {(sl.pred_area/sl.gt_area).median():.3f}  (1.0 = unbiased; <1 = under-segmenting)")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    ax[0, 0].scatter(sl.edge_dist + np.random.uniform(-.2, .2, len(sl)), sl.dice, s=10, alpha=.5, c=np.where(sl.poor, "#c44e52", "#4c72b0"))
    ax[0, 0].set_xlabel("distance from stack edge (slices)"); ax[0, 0].set_ylabel("Dice"); ax[0, 0].set_title("Edge-of-volume effect")
    ax[0, 1].scatter(sl.gt_area, sl.dice, s=10, alpha=.5, c=np.where(sl.poor, "#c44e52", "#4c72b0"))
    ax[0, 1].set_xlabel("true placenta area (px)"); ax[0, 1].set_ylabel("Dice"); ax[0, 1].set_title("Placenta size")
    ax[0, 2].scatter(sl.gt_elongation, sl.dice, s=10, alpha=.5, c=np.where(sl.poor, "#c44e52", "#4c72b0"))
    ax[0, 2].set_xlabel("elongation (major/minor axis)"); ax[0, 2].set_ylabel("Dice"); ax[0, 2].set_title("Shape")
    ax[1, 0].scatter(sl.mean_mc_std_fg, sl.dice, s=10, alpha=.5, c=np.where(sl.cohort == "v", "#dd8452", "#4c72b0"))
    ax[1, 0].set_xlabel("mean MC-dropout std on true placenta"); ax[1, 0].set_ylabel("Dice"); ax[1, 0].set_title("Does MC uncertainty predict Dice?  (orange = v cohort)")
    ax[1, 1].scatter(sl.mean_conf_band, sl.dice, s=10, alpha=.5, c=np.where(sl.poor, "#c44e52", "#4c72b0"))
    ax[1, 1].set_xlabel("mean confidence within 5 px of TRUE edge"); ax[1, 1].set_ylabel("Dice"); ax[1, 1].set_title("Confidence at the true boundary (inverted signal)")
    x = np.arange(len(E))
    ax[1, 2].bar(x - .2, E.acc, .4, label="accuracy", color="#4c72b0"); ax[1, 2].bar(x + .2, E.conf, .4, label="mean confidence", color="#dd8452")
    ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(E.index, rotation=45); ax[1, 2].set_ylim(0.5, 1.01); ax[1, 2].legend()
    ax[1, 2].set_xlabel("distance from true boundary (px)"); ax[1, 2].set_title("Accuracy vs confidence by distance from edge")
    plt.tight_layout(); plt.savefig(d / "fig_patterns.png", dpi=100)
    A.to_csv(d / "patterns_feature_table.csv", index=False); E.to_csv(d / "patterns_by_distance.csv")
    print(f"\nsaved -> {(d / 'fig_patterns.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
