import argparse
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import make_loader
from augment import TrainAugment
from model import DenseUNet

ROOT = Path(__file__).resolve().parents[1]


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def loss_fn(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def dice_per_slice(logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (denom + eps)


def run_epoch(model, loader, dev, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    tot_loss, dices, n = 0.0, [], 0
    for img, msk, _ in tqdm(loader, leave=False, desc="train" if training else "val  "):
        img, msk = img.to(dev, non_blocking=True), msk.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(img)
            loss = loss_fn(logits.float(), msk)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * img.size(0)
        dices.append(dice_per_slice(logits.float(), msk).cpu())
        n += img.size(0)
    return tot_loss / n, torch.cat(dices).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1)
    ap.add_argument("--sequences", nargs="+", default=["ssh_TSE"])
    ap.add_argument("--run", default="smoke")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)

    train_loader = make_loader("train", tuple(args.sequences), args.batch, TrainAugment(seed=args.seed), num_workers=args.workers)
    val_loader = make_loader("val", tuple(args.sequences), args.batch, None, num_workers=args.workers)
    print(f"device={dev} | train {len(train_loader.dataset)} slices | val {len(val_loader.dataset)} slices | batch {args.batch}")

    model = DenseUNet(pretrained=True).to(dev)
    enc_names = {"stem", "pool0", "block1", "trans1", "block2", "trans2", "block3", "trans3", "block4"}
    enc = [p for n, p in model.named_parameters() if n.split(".")[0] in enc_names]
    dec = [p for n, p in model.named_parameters() if n.split(".")[0] not in enc_names]
    optimizer = torch.optim.AdamW([
        {"params": enc, "lr": args.lr * args.encoder_lr_scale},
        {"params": dec, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = run_epoch(model, train_loader, dev, optimizer)
        va_loss, va_dice = run_epoch(model, val_loader, dev)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": tr_loss, "train_dice": tr_dice,
               "val_loss": va_loss, "val_dice": va_dice, "seconds": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        flag = ""
        if va_dice > best:
            best = va_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": va_dice, "args": vars(args)},
                       ROOT / "checkpoints" / f"{args.run}_best.pt")
            flag = "  <- saved"
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} dice {va_dice:.4f} | {row['seconds']:.0f}s{flag}")

    print(f"\nbest val dice {best:.4f} -> checkpoints/{args.run}_best.pt")


if __name__ == "__main__":
    main()
