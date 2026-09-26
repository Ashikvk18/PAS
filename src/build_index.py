
from pathlib import Path
import re
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Placenta accreta spectrum disorders(PASDs)"
OUT = ROOT / "data" / "index.csv"

SEQUENCES = ["ssh_TSE", "BTFE"]
NAME_RE = re.compile(r"^(?P<cohort>sub|v)(?P<pid>\d+)_(?P<slice>\d+)\.jpg$")


def build_index() -> pd.DataFrame:
    rows = []
    for seq in SEQUENCES:
        img_dir = DATA / f"{seq}_image" / "JPEGImages"
        msk_dir = DATA / f"{seq}_mask" / "SegmentationClass"
        for img_path in sorted(img_dir.glob("*.jpg")):
            m = NAME_RE.match(img_path.name)
            if not m:
                raise ValueError(f"Unexpected filename: {img_path.name}")
            msk_path = msk_dir / (img_path.stem + ".png")
            if not msk_path.exists():
                raise FileNotFoundError(f"No mask for {img_path.name}")
            with Image.open(msk_path) as mk:
                mask_size = mk.size[0]
            rows.append({
                "sequence": seq,
                "cohort": m["cohort"],
                "patient": f"{m['cohort']}{int(m['pid']):03d}",
                "slice": int(m["slice"]),
                "image_path": str(img_path.relative_to(ROOT)),
                "mask_path": str(msk_path.relative_to(ROOT)),
                "mask_size": mask_size,
            })
    df = pd.DataFrame(rows).sort_values(["sequence", "patient", "slice"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"Total slices: {len(df)}")
    for seq, g in df.groupby("sequence"):
        pts = g.groupby("patient")["slice"].count()
        print(f"\n[{seq}] slices={len(g)} patients={g.patient.nunique()} "
              f"(sub={g[g.cohort=='sub'].patient.nunique()}, v={g[g.cohort=='v'].patient.nunique()})")
        print(f"  slices/patient: min={pts.min()} median={int(pts.median())} max={pts.max()}")
        print(f"  mask sizes: {sorted(g.mask_size.unique().tolist())}")
    both = set(df[df.sequence == 'ssh_TSE'].patient) & set(df[df.sequence == 'BTFE'].patient)
    print(f"\nPatients present in BOTH sequences: {len(both)}")


if __name__ == "__main__":
    df = build_index()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    summarize(df)
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")
    print("\nFirst rows:")
    print(df.head(5).to_string(index=False))
