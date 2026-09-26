from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 512


def load_frame(sequences=("ssh_TSE",), split=None) -> pd.DataFrame:
    index = pd.read_csv(ROOT / "data" / "index.csv")
    splits = pd.read_csv(ROOT / "data" / "split.csv")[["patient", "split"]]
    df = index.merge(splits, on="patient")
    df = df[df.sequence.isin(sequences)]
    if split is not None:
        df = df[df.split == split]
    return df.reset_index(drop=True)


class PlacentaDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None, channels=3):
        self.frame = frame
        self.transform = transform
        self.channels = channels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(ROOT / row.image_path).convert("L")
        msk = Image.open(ROOT / row.mask_path)
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        if msk.size != (IMG_SIZE, IMG_SIZE):
            msk = msk.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk) > 0).astype(np.float32)

        if self.transform is not None:
            img, msk = self.transform(img, msk)

        img = torch.from_numpy(np.ascontiguousarray(img))[None]
        msk = torch.from_numpy(np.ascontiguousarray(msk))[None]
        if self.channels == 3:
            img = img.repeat(3, 1, 1)

        meta = {"patient": row.patient, "slice": int(row.slice),
                "sequence": row.sequence, "cohort": row.cohort, "split": row.split}
        return img, msk, meta


def make_loader(split, sequences=("ssh_TSE",), batch_size=8, transform=None, shuffle=None, num_workers=4):
    frame = load_frame(sequences, split)
    ds = PlacentaDataset(frame, transform=transform)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        f = load_frame(("ssh_TSE",), split)
        print(f"{split:5s}: {len(f):4d} slices, {f.patient.nunique():3d} patients")

    loader = make_loader("train", batch_size=8, num_workers=0)
    img, msk, meta = next(iter(loader))
    print("\nimage batch:", tuple(img.shape), img.dtype, f"range [{img.min():.3f}, {img.max():.3f}]")
    print("mask  batch:", tuple(msk.shape), msk.dtype, f"unique {torch.unique(msk).tolist()}")
    print("mask coverage per slice (% of pixels):", [f"{v:.1f}" for v in (msk.mean(dim=(1, 2, 3)) * 100).tolist()])
    print("patients in batch:", meta["patient"])
