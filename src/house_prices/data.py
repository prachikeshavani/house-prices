"""Load the raw Kaggle CSVs.

Paths are relative to the project root, so tests and notebooks
agree on where the data lives.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"

# Reading the CSV is a decision, not a default:
# - keep_default_na=False stops pandas silently eating the string "None",
#   which in MasVnrType is a REAL category ("no masonry veneer") — 864 houses
# - "NA" as a literal means "this house has none of that feature" (the
#   garage/pool/alley families), so those become NaN on purpose and the
#   pipeline encodes absence as its own information
_READ = dict(keep_default_na=False, na_values=["", "NA"])


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv", **_READ)


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA / "test.csv", **_READ)
