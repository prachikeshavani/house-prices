"""Tests for the submission writer (Saturday).

The submission file is the only artifact Kaggle sees, so its format is
guarded hard: nothing gets written until every check passes.
"""
import numpy as np
import pandas as pd
import pytest

from house_prices.data import load_test
from house_prices.submission import CHAMPION_PATH, make_submission, validate


def test_validate_rejects_wrong_columns():
    bad = pd.DataFrame({"id": [1], "price": [100.0]})
    with pytest.raises(ValueError, match="columns"):
        validate(bad, expected_rows=1)


def test_validate_rejects_wrong_row_count():
    bad = pd.DataFrame({"Id": [1, 2], "SalePrice": [1e5, 2e5]})
    with pytest.raises(ValueError, match="rows"):
        validate(bad, expected_rows=1)


def test_validate_rejects_bad_prices():
    neg = pd.DataFrame({"Id": [1], "SalePrice": [-5.0]})
    with pytest.raises(ValueError, match="positive"):
        validate(neg, expected_rows=1)
    nan = pd.DataFrame({"Id": [1], "SalePrice": [np.nan]})
    with pytest.raises(ValueError, match="finite"):
        validate(nan, expected_rows=1)


def test_make_submission_end_to_end(tmp_path):
    out = tmp_path / "submission.csv"
    make_submission(champion_path=CHAMPION_PATH, out_path=out)

    sub = pd.read_csv(out)
    test = load_test()
    assert list(sub.columns) == ["Id", "SalePrice"]
    assert len(sub) == len(test) == 1459
    assert sub["Id"].tolist() == test["Id"].tolist()  # aligned, same order
    assert np.isfinite(sub["SalePrice"]).all()
    assert (sub["SalePrice"] > 0).all()
    # plausible dollar amounts for Ames, Iowa
    assert sub["SalePrice"].between(10_000, 1_000_000).all()
