import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def reliability(conf, correct, bins):
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"lo": edges[b], "hi": edges[b + 1], "conf": conf[m].mean(),
                     "acc": correct[m].mean(), "n": int(m.sum())})
    t = pd.DataFrame(rows)
    ece = (t.n / t.n.sum() * (t.acc - t.conf).abs()).sum()
    mce = (t.acc - t.conf).abs().max()
    return t, ece, mce


def plot_panel(ax, t, ece, mce, title):
    w = t.hi - t.lo
    ax.bar(t.lo, t.acc, width=w, align="edge", color="#4c72b0", edgecolor="white", label="accuracy in bin")
    ax.bar(t.lo, t.conf - t.acc, bottom=t.acc, width=w, align="edge", color="#dd8452", alpha=0.55,
           edgecolor="white", label="gap (overconfidence)")
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1.001)
    ax.set_xlabel("confidence  max(p, 1-p)"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}   MCE = {mce:.4f}   n = {t.n.sum():,}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    for _, r in t.iterrows():
        ax.text((r.lo + r.hi) / 2, 0.505, f"{r.n/1000:.0f}k", ha="center", fontsize=6, color="gray", rotation=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/full_ssh_tse/pred_test_ssh_TSE")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--band", type=float, default=5.0)
    args = ap.parse_args()

    d = ROOT / args.pred
    px = pd.read_parquet(d / "pixels.parquet")
    conf = np.maximum(px.prob, 1 - px.prob).to_numpy()
    correct = px.correct.to_numpy()

    subsets = {
        "All pixels": np.ones(len(px), bool),
        f"Boundary band (within {args.band:g} px of true edge)": (px.dist <= args.band).to_numpy(),
        "True placenta pixels only": (px["gt"] == 1).to_numpy(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    summary = []
    for ax, (name, m) in zip(axes, subsets.items()):
        t, ece, mce = reliability(conf[m], correct[m], args.bins)
        plot_panel(ax, t, ece, mce, name)
        summary.append({"subset": name, "n": int(m.sum()), "ECE": ece, "MCE": mce,
                        "acc": correct[m].mean(), "mean_conf": conf[m].mean(),
                        "frac_conf>0.99": (conf[m] > 0.99).mean(),
                        "acc_when_conf>0.99": correct[m & (conf > 0.99)].mean()})
        t.to_csv(d / f"reliability_{name.split(' ')[0].lower()}.csv", index=False)
    plt.tight_layout()
    plt.savefig(d / "fig_reliability.png", dpi=110)

    s = pd.DataFrame(summary)
    s.to_csv(d / "calibration_summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    print(f"\nsaved -> {(d / 'fig_reliability.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
