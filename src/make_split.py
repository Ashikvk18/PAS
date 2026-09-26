from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index.csv"
OUT = ROOT / "data" / "split.csv"

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def make_split(index: pd.DataFrame) -> pd.DataFrame:
   
    patients = index[["patient", "cohort"]].drop_duplicates().reset_index(drop=True)

    train_val, test = train_test_split(
        patients, test_size=TEST_FRAC, stratify=patients.cohort, random_state=SEED)
    train, val = train_test_split(
        train_val, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=train_val.cohort, random_state=SEED)

    split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return split.sort_values("patient").reset_index(drop=True)


def summarize(split: pd.DataFrame, index: pd.DataFrame) -> None:
    merged = index.merge(split[["patient", "split"]], on="patient")
    print("Patients per split (by cohort):")
    print(split.pivot_table(index="split", columns="cohort", values="patient", aggfunc="count", fill_value=0, margins=True))
    print("\nSlices per split (by sequence):")
    print(merged.pivot_table(index="split", columns="sequence", values="slice", aggfunc="count", fill_value=0, margins=True))
   
    assert split.patient.is_unique, "A patient appears in more than one split!"
    overlap = merged.groupby("patient")["split"].nunique().max()
    print(f"\nLeakage check: max splits per patient = {overlap} (must be 1)  -> {'OK' if overlap == 1 else 'FAIL'}")


if __name__ == "__main__":
    index = pd.read_csv(INDEX)
    split = make_split(index)
    split.to_csv(OUT, index=False)
    summarize(split, index)
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")
