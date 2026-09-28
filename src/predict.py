import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from dataset import make_loader
from model import DenseUNet, enable_mc_dropout

ROOT = Path(__file__).resolve().parents[1]


def boundary_distance(mask):
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(1 - mask)
    return np.where(mask > 0, inside, outside)


@torch.no_grad()
def predict_batch(model, img, mc_samples):
    model.eval()
    p_det = torch.sigmoid(model(img))
    enable_mc_dropout(model)
    samples = torch.stack([torch.sigmoid(model(img)) for _ in range(mc_samples)])
    model.eval()
    return p_det, samples.mean(0), samples.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/full_ssh_tse_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sequences", nargs="+", default=["ssh_TSE"])
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--pixels-per-slice", type=int, default=4000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.checkpoint).stem.replace("_best", "")
    out = ROOT / "runs" / run / (args.out or f"pred_{args.split}_{'_'.join(args.sequences)}")
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ROOT / args.checkpoint, map_location=dev, weights_only=False)
    model = DenseUNet(pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.checkpoint} (epoch {ck['epoch']}, val dice {ck['val_dice']:.4f}) | split={args.split} seq={args.sequences}")

    loader = make_loader(args.split, tuple(args.sequences), args.batch, None, shuffle=False, num_workers=4)
    n = len(loader.dataset)

    slice_rows, pixel_chunks = [], []
    prob_maps = np.zeros((n, 512, 512), dtype=np.uint8)
    std_maps = np.zeros((n, 512, 512), dtype=np.uint8)
    gt_maps = np.zeros((n, 512, 512), dtype=bool)
    k = 0
    for img, msk, meta in tqdm(loader, desc="predict"):
        img = img.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            p_det, p_mc, p_std = predict_batch(model, img, args.mc_samples)
        p_det, p_mc, p_std, gt = (t.float().cpu().numpy()[:, 0] for t in (p_det, p_mc, p_std, msk))

        for b in range(img.size(0)):
            p, s, g = p_mc[b], p_std[b], gt[b].astype(np.uint8)
            pred = p > 0.5
            correct = pred == g.astype(bool)
            conf = np.maximum(p, 1 - p)
            inter = (pred & (g > 0)).sum()
            dice = 2 * inter / (pred.sum() + g.sum() + 1e-6)
            dist = boundary_distance(g)

            slice_rows.append({
                "patient": meta["patient"][b], "cohort": meta["cohort"][b], "sequence": meta["sequence"][b],
                "slice": int(meta["slice"][b]), "row": k,
                "dice": dice, "pixel_acc": correct.mean(),
                "gt_area": int(g.sum()), "pred_area": int(pred.sum()),
                "mean_conf": conf.mean(), "mean_conf_fg": conf[g > 0].mean() if g.sum() else np.nan,
                "mean_conf_band": conf[dist <= 5].mean(),
                "mean_mc_std": s.mean(), "mean_mc_std_fg": s[g > 0].mean() if g.sum() else np.nan,
                "frac_uncertain": (conf < 0.7).mean(),
                "det_mc_agreement": ((p_det[b] > 0.5) == pred).mean(),
            })

            idx = rng.choice(p.size, size=min(args.pixels_per_slice, p.size), replace=False)
            pixel_chunks.append(pd.DataFrame({
                "row": k, "prob": p.ravel()[idx].astype(np.float32),
                "gt": g.ravel()[idx].astype(np.uint8), "correct": correct.ravel()[idx],
                "mc_std": s.ravel()[idx].astype(np.float32),
                "dist": dist.ravel()[idx].astype(np.float32),
            }))

            prob_maps[k] = np.round(p * 255).astype(np.uint8)
            std_maps[k] = np.round(np.clip(s * 4, 0, 1) * 255).astype(np.uint8)
            gt_maps[k] = g > 0
            k += 1

    slices = pd.DataFrame(slice_rows)
    pixels = pd.concat(pixel_chunks, ignore_index=True)
    slices.to_csv(out / "slices.csv", index=False)
    pixels.to_parquet(out / "pixels.parquet", index=False)
    np.savez_compressed(out / "maps.npz", prob=prob_maps, std=std_maps, gt=gt_maps)

    print(f"\nsaved -> {out.relative_to(ROOT)}  ({len(slices)} slices, {len(pixels):,} sampled pixels)")
    print(f"slice Dice: mean {slices.dice.mean():.4f}  median {slices.dice.median():.4f}  min {slices.dice.min():.4f}")
    print(f"patient-level Dice (mean of per-patient means): {slices.groupby('patient').dice.mean().mean():.4f}")
    print(f"by cohort:\n{slices.groupby('cohort').dice.agg(['mean', 'count']).round(4).to_string()}")
    print(f"pixel accuracy {slices.pixel_acc.mean():.4f} | mean confidence {slices.mean_conf.mean():.4f} | "
          f"mean confidence within 5px of boundary {slices.mean_conf_band.mean():.4f}")


if __name__ == "__main__":
    main()
