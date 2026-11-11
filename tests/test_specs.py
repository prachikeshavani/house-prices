"""Tests for Friday's candidate specs (S1 rare-collapse, S3 twin-drops).

Each spec is a data decision, so each test pins the *data* outcome, not just
that code runs.
"""
import numpy as np
import pandas as pd

from house_prices.features import (
    TWIN_DROPS,
    build_features,
    build_features_collapse_rare,
    build_features_drop_twins,
    collapse_rare_categories,
)
from house_prices.train import fit_spec


# ---------- S1: rare-category collapse ----------

def test_rare_categories_become_other():
    df = pd.DataFrame({
        "roof": ["CompShg", "CompShg", "CompShg", "ClyTile", "Metal"],
        "SalePrice": [1, 2, 3, 4, 5],
    })
    out = collapse_rare_categories(df, min_count=2)
    assert list(out["roof"]) == ["CompShg", "CompShg", "CompShg", "Other", "Other"]


def test_common_categories_survive_collapse(train):
    out = collapse_rare_categories(train, min_count=10)
    n_brkface = (out["RoofMatl"] == "BrkFace").sum()
    assert n_brkface == (train["RoofMatl"] == "BrkFace").sum()
    # the one-house levers from Thursday are gone
    assert (out["RoofMatl"] == "ClyTile").sum() == 0
    assert (out["RoofMatl"] == "Metal").sum() == 0


def test_collapse_preserves_absence_blanks(train):
    # "house has none" blanks must NOT become "Other" — absence is its own fact
    out = collapse_rare_categories(train, min_count=10)
    assert out["PoolQC"].isna().sum() == train["PoolQC"].isna().sum()


def test_s1_produces_fewer_columns_than_v1(train):
    X_v1, _ = build_features(train)
    X_s1, _ = build_features_collapse_rare(train)
    assert X_s1.shape[1] < X_v1.shape[1]


# ---------- S3: collinear twin drops ----------

def test_twin_drops_remove_one_from_each_pair(train):
    df = train.drop(columns=["Id", "SalePrice"])
    remaining = [c for c in TWIN_DROPS if c in df.columns]
    assert set(remaining) <= set(TWIN_DROPS)
    X, _ = build_features_drop_twins(train)
    for dropped in TWIN_DROPS:
        assert dropped not in X.columns
        assert not any(c.startswith(dropped + "_") for c in X.columns)


# ---------- S2: outlier sensitivity (a data decision, not a feature one) ----------

def test_fit_spec_runs_on_outlier_filtered_data(train):
    kept = train[~train["Id"].isin([826, 524])]
    model, metrics = fit_spec(kept)
    assert metrics["rmsle"] > 0
    assert len(kept) == len(train) - 2


def test_spec_registry_covers_every_feature_spec():
    from house_prices.features import SPEC_FNS
    assert set(SPEC_FNS) >= {"v1_full_pipeline", "S1_collapse_rare",
                            "S3_drop_twins", "S4_collapse_and_drop_twins"}
    for fn in SPEC_FNS.values():
        assert callable(fn)
