"""Tests for training + the experiment log (Wednesday)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_prices.train import cross_val_metrics, fit_spec, log_run
from house_prices.features import align_columns


def test_log_run_creates_file_with_header_then_appends(tmp_path):
    p = tmp_path / "runs.csv"
    log_run(p, "smoke_a", rmse=31_000.0, rmsle=0.151, notes="first")
    log_run(p, "smoke_b", rmse=29_500.0, rmsle=0.148, notes="second try, fewer rooms")
    df = pd.read_csv(p)
    assert list(df.columns) == ["spec", "cv_rmse_usd", "cv_rmsle", "notes"]
    assert len(df) == 2
    assert df.loc[0, "spec"] == "smoke_a"
    assert df.loc[1, "notes"] == "second try, fewer rooms"


def test_log_run_rejects_empty_reason(tmp_path):
    # a run you can't explain is a run you didn't learn from
    with pytest.raises(ValueError, match="one-line reason"):
        log_run(tmp_path / "runs.csv", "mystery", rmse=1.0, rmsle=0.1, notes="  ")


def test_cross_val_metrics_returns_positive_errors(train):
    from house_prices.features import build_features
    X, y = build_features(train)
    m = cross_val_metrics(LinearRegression(), X, y)
    assert m["rmse"] > 0 and m["rmsle"] > 0
    # RMSLE lives on the log scale (~0.1), RMSE on the dollar scale (~tens of thousands)
    assert m["rmsle"] < 1.0
    assert m["rmse"] > 1_000


def test_fit_spec_returns_model_fitted_on_all_rows(train):
    model, metrics = fit_spec(train)
    assert hasattr(model, "coef_")  # fitted
    assert metrics["rmse"] > 0


def test_align_columns_reconciles_train_and_test():
    reference = ["a", "b"]
    other = pd.DataFrame({"b": [5], "c": [6]})
    aligned = align_columns(other, reference)
    assert list(aligned.columns) == ["a", "b"]
    assert aligned.loc[0, "a"] == 0  # column the test set never had -> 0
    assert aligned.loc[0, "b"] == 5  # shared column keeps its value
    assert "c" not in aligned.columns  # extra column dropped
