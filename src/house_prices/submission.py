"""Generate the Kaggle submission file from the champion bundle.

The champion ships as {model, spec_name, columns}; this module rebuilds the
test features through the same spec function, aligns the column language,
predicts, and converts back to dollars with expm1. The file cannot be written
until validate() passes — a bad submission is worse than no submission.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from house_prices.data import load_test
from house_prices.features import SPEC_FNS, align_columns, inverse_target

ROOT = Path(__file__).resolve().parents[2]
CHAMPION_PATH = ROOT / "models" / "champion.joblib"
DEFAULT_OUT = ROOT / "data" / "processed" / "submission.csv"


def validate(sub: pd.DataFrame, expected_rows: int) -> None:
    """Every rule the Kaggle format demands, checked before anything is written."""
    if list(sub.columns) != ["Id", "SalePrice"]:
        raise ValueError(f"columns must be exactly [Id, SalePrice], got {list(sub.columns)}")
    if len(sub) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(sub)}")
    if not np.isfinite(sub["SalePrice"]).all():
        raise ValueError("SalePrice must be finite — no NaN or inf allowed")
    if (sub["SalePrice"] <= 0).all() or (sub["SalePrice"] <= 0).any():
        raise ValueError("SalePrice must be positive on the dollar scale")


def make_submission(champion_path=CHAMPION_PATH, out_path=DEFAULT_OUT) -> Path:
    bundle = joblib.load(champion_path)
    test = load_test()

    X_raw, _ = SPEC_FNS[bundle["spec_name"]](test)
    X = align_columns(X_raw, bundle["columns"])
    preds = inverse_target(bundle["model"].predict(X))

    sub = pd.DataFrame({"Id": test["Id"], "SalePrice": preds})
    validate(sub, expected_rows=len(test))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"submission written: {out_path} ({len(sub)} rows)")
    return out_path


if __name__ == "__main__":
    make_submission()
