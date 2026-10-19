"""Fit OLS and log every run to a plain CSV — no experiment-tracking UI.

Two ideas do the work here:
- cross_val_metrics: 5-fold CV. We fit on 4/5 of the houses, predict the held-out
  1/5, and repeat until every house has a prediction it wasn't trained on
  ("out-of-fold"). Errors measured there are honest — the model has never seen
  those houses.
- Two error rulers: RMSE measured back in dollars (feels real), RMSLE measured
  in log space (this is the number Kaggle scores, and log errors don't let
  mansions dominate).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from house_prices.data import load_train
from house_prices.features import build_features, inverse_target

RESULTS = Path(__file__).resolve().parents[2] / "results" / "runs.csv"


def cross_val_metrics(model, X, y_log, n_splits=5, seed=42) -> dict:
    """Out-of-fold predictions over n_splits folds; returns RMSE (dollars) and RMSLE (log)."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_log))
    y_arr = np.asarray(y_log)
    for train_idx, test_idx in kfold.split(X):
        model.fit(X.iloc[train_idx], y_arr[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
    rmse = float(np.sqrt(np.mean((inverse_target(y_arr) - inverse_target(oof)) ** 2)))
    rmsle = float(np.sqrt(np.mean((y_arr - oof) ** 2)))
    return {"rmse": rmse, "rmsle": rmsle}


def fit_spec(df, feature_fn=build_features, spec_name="v1_full_pipeline"):
    """Build features, CV-score the spec, then fit on ALL rows for the returned model."""
    X, y = feature_fn(df)
    model = LinearRegression()
    metrics = cross_val_metrics(model, X, y)
    model.fit(X, np.asarray(y))
    return model, metrics


def log_run(path, spec, rmse, rmsle, notes=""):
    """Append one row to the experiment log, creating it with a header if absent.

    Every run must carry a one-line reason — 'we tried it because...'.
    A run you can't explain is a run you didn't learn from.
    """
    if not notes.strip():
        raise ValueError("log_run requires a one-line reason in notes=")
    row = pd.DataFrame([{
        "spec": spec,
        "cv_rmse_usd": round(rmse, 2),
        "cv_rmsle": round(rmsle, 5),
        "notes": notes,
    }])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    df = load_train()
    model, m = fit_spec(df)
    log_run(RESULTS, "v1_full_pipeline", m["rmse"], m["rmsle"],
            "first OLS fit: ordinal + impute + log + one-hot")
    print(f"v1_full_pipeline  CV RMSE ${m['rmse']:,.0f}  CV RMSLE {m['rmsle']:.5f}")
    print(f"logged -> {RESULTS}")


if __name__ == "__main__":
    main()
