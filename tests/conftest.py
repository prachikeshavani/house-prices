"""Shared pytest fixtures.

conftest.py is pytest's convention: anything defined here is visible to
every test file, so all modules share the same loaded dataset instead of
re-reading the CSV per file.
"""
import pytest

from house_prices.data import load_train
from house_prices.features import build_features


@pytest.fixture(scope="module")
def train():
    return load_train()


@pytest.fixture(scope="module")
def built(train):
    X, y = build_features(train)
    return X, y
