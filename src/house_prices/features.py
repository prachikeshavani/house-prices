"""Feature pipeline — every choice here traces back to 01_eda.ipynb.

Stages, in order:
1. Target: log1p(SalePrice) — the skew fix from the EDA (raw skew 1.88 -> 0.12)
2. Ordinals: quality words -> numbers; blank means "house has none" -> 0
3. Imputation: only true unknowns remain blank by now; median/mode
4. Skewed numerics: log1p the heavy right tails (skew table in the EDA)
5. One-hot: whatever text is left becomes 0/1 columns

`build_features` returns (X, y) ready for scikit-learn; `inverse_target`
maps predictions back to dollars for submission day.
"""
import numpy as np
import pandas as pd

# "Excellent" -> 5 ... "Poor" -> 1; NaN (feature absent) handled separately -> 0
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

# The quality/condition family — per data_description.txt these share the
# same five-grade scale, and blanks mean "no such feature" (no basement,
# no garage, no fireplace, no pool), which is real information.
ORDINAL_COLUMNS = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]

# Top of the EDA skew table (skew > ~2 is a heavy right tail).
SKEWED_NUMERIC = [
    "MiscVal", "PoolArea", "LotArea", "3SsnPorch", "LowQualFinSF",
    "KitchenAbvGr", "BsmtFinSF2", "ScreenPorch", "BsmtHalfBath",
    "EnclosedPorch", "MasVnrArea", "OpenPorchSF",
]

ID_COLUMNS = ["Id"]
TARGET = "SalePrice"

# Thursday's evidence: one twin from each near-duplicate pair gets dropped
# (the other is kept). Monday's correlations decided which is the "keeper":
# GarageCars .640 > GarageArea .623 · TotalBsmtSF .614 > 1stFlrSF .606 ·
# GrLivArea .709 > TotRmsAbvGrd .534
TWIN_DROPS = ["GarageArea", "1stFlrSF", "TotRmsAbvGrd"]
RARE_MIN_COUNT = 10


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Map quality words to 1-5; blanks (absence) become 0."""
    df = df.copy()
    for col in ORDINAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining blanks: median for numbers, most common for text."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            fill = df[col].median() if df[col].dtype.kind in "if" else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
    return df


def log_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """log1p the heavy-tailed area/count features (log1p keeps zeros as 0)."""
    df = df.copy()
    for col in SKEWED_NUMERIC:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Turn remaining text columns into 0/1 indicator columns."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(text_cols) == 0:
        return df
    return pd.get_dummies(df, columns=text_cols, dtype=int)


def build_features(df: pd.DataFrame, fit: bool = True):
    """Run the full pipeline.

    Returns (X, y): y is the log1p target, or None when the input has no
    SalePrice column (the Kaggle test set).
    """
    df = df.copy()

    y = None
    if TARGET in df.columns:
        y = np.log1p(df[TARGET])
        df = df.drop(columns=[TARGET])
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    df = encode_ordinal(df)
    df = impute(df)
    df = log_skewed(df)
    X = one_hot(df)
    return X, y


def align_columns(X: pd.DataFrame, reference_columns) -> pd.DataFrame:
    """Reindex X to the reference (train's) column language.

    One-hot invents a column per category, and the test set doesn't have
    every category the train set has (and vice versa): missing columns
    become 0 ("this house is not in that category"), extra columns drop.
    """
    return X.reindex(columns=list(reference_columns), fill_value=0)


# ---------- Friday's candidate specs ----------

def collapse_rare_categories(df: pd.DataFrame, min_count: int = RARE_MIN_COUNT) -> pd.DataFrame:
    """S1: any text category with fewer than min_count houses becomes "Other".

    Thursday found categories that appear once (RoofMatl_ClyTile) act as
    private levers — leverage 1.0, absurd coefficients. Absence blanks
    (NaN) are untouched: "no pool" is a fact, not a rare category.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        rare = df[col].value_counts()
        rare = set(rare[rare < min_count].index)
        if rare:
            df[col] = df[col].replace(rare, "Other")
    return df


def build_features_collapse_rare(df: pd.DataFrame, min_count: int = RARE_MIN_COUNT):
    """Spec S1: collapse rare categories, then the standard pipeline."""
    return build_features(collapse_rare_categories(df, min_count))


def build_features_drop_twins(df: pd.DataFrame):
    """Spec S3: drop one collinear twin per pair (TWIN_DROPS), then pipeline."""
    df = df.drop(columns=[c for c in TWIN_DROPS if c in df.columns])
    return build_features(df)


def build_features_collapse_and_drop_twins(df: pd.DataFrame, min_count: int = RARE_MIN_COUNT):
    """Spec S4: S1 and S3 together — the kitchen-sink challenger."""
    df = collapse_rare_categories(df, min_count)
    df = df.drop(columns=[c for c in TWIN_DROPS if c in df.columns])
    return build_features(df)


# Names -> feature functions, for the spec arena and for submission day:
# the champion's name is serialized, and its function is looked up here.
SPEC_FNS = {
    "v1_full_pipeline": build_features,
    "S1_collapse_rare": build_features_collapse_rare,
    "S3_drop_twins": build_features_drop_twins,
    "S4_collapse_and_drop_twins": build_features_collapse_and_drop_twins,
}


def inverse_target(y_log: np.ndarray) -> np.ndarray:
    """Map log-space predictions back to dollars."""
    return np.expm1(y_log)
