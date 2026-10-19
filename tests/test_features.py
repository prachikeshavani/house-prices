"""Tests for the feature pipeline.

Each test pins down one decision from 01_eda.ipynb:
- quality words become numbers ("Ex" -> 5 ... "Po" -> 1, absent -> 0)
- true-unknown blanks are imputed (median/mode) so no NaNs survive
- skewed features and the target are log1p-transformed
- remaining text columns become one-hot 0/1 columns
- identifiers and the target never leak into X
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_prices.data import load_test
from house_prices.features import build_features, encode_ordinal, inverse_target

ROOT = Path(__file__).resolve().parents[1]


# ---------- data.py ----------

def test_load_train_shape(train):
    assert train.shape == (1460, 81)


def test_load_test_has_no_target():
    test = load_test()
    assert test.shape[0] == 1459
    assert "SalePrice" not in test.columns


def test_none_is_a_real_category_not_a_missing_value(train):
    # pandas' default NA list silently converts the string "None" to NaN;
    # in MasVnrType it means "no masonry veneer" — 864 houses (59%), not 8.
    assert (train["MasVnrType"] == "None").sum() == 864
    assert train["MasVnrType"].isna().sum() == 8


# ---------- ordinal encoding ----------

def test_quality_words_become_numbers(train):
    X, _ = build_features(train)
    # ExterQual has no blanks in train, so its encoded values must be 1..5
    assert set(X["ExterQual"].unique()) <= {1, 2, 3, 4, 5}
    assert X["ExterQual"].dtype.kind in "iu"


def test_known_quality_mapping(train):
    encoded = encode_ordinal(train[["ExterQual"]].copy())
    mask = train["ExterQual"] == "Ex"
    assert (encoded.loc[mask, "ExterQual"] == 5).all()


def test_absence_becomes_zero(train):
    # BsmtQual is blank exactly when the house has no basement (see EDA):
    # absence must map to 0, not to a median
    X, _ = build_features(train)
    no_basement = train["BsmtQual"].isna()
    assert (X.loc[no_basement, "BsmtQual"] == 0).all()
    assert X.loc[~no_basement, "BsmtQual"].min() >= 1


# ---------- imputation ----------

def test_no_nans_survive(built):
    X, _ = built
    assert X.isna().sum().sum() == 0


# ---------- log transforms ----------

def test_target_is_log_transformed(train, built):
    _, y = built
    assert np.allclose(y.to_numpy(), np.log1p(train["SalePrice"].to_numpy()))


def test_inverse_target_round_trips():
    prices = np.array([100_000.0, 250_000.0, 1_500_000.0])
    assert np.allclose(inverse_target(np.log1p(prices)), prices)


def test_skewed_features_are_logged(train, built):
    X, _ = built
    # LotArea skew was 12.2 raw; after log1p it must be far tamer
    assert abs(X["LotArea"].skew()) < abs(train["LotArea"].skew())


# ---------- one-hot & leakage ----------

def test_no_text_columns_remain(built):
    X, _ = built
    assert len(X.select_dtypes(include=["object"]).columns) == 0


def test_id_and_target_not_in_features(built):
    X, _ = built
    assert "Id" not in X.columns
    assert "SalePrice" not in X.columns


def test_row_count_preserved(train, built):
    X, y = built
    assert len(X) == len(train) == len(y)
