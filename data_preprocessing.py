# data_preprocessing.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

RAW_FEATURE = "Level"
RAW_TARGET = "Salary"

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV; raise a clear error if file missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    return pd.read_csv(path)

def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce strings like '₹1,200' or '1,200' to numeric, else NaN."""
    s = series.astype(str).str.replace(r"[₹$,]", "", regex=True).str.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def clean_salary_data(
    df: pd.DataFrame,
    feature_col: str = RAW_FEATURE,
    target_col: str = RAW_TARGET,
    clip_outliers: bool = True,
) -> pd.DataFrame:
    """
    Practical cleaning:
      - ensure required columns
      - coerce to numeric
      - drop NA and non-positive values
      - drop duplicates
      - optional IQR clipping for stability
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = {feature_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[feature_col] = _to_numeric(df[feature_col])
    df[target_col] = _to_numeric(df[target_col])

    df = df.dropna(subset=[feature_col, target_col])
    df = df[(df[feature_col] > 0) & (df[target_col] > 0)]
    df = df.drop_duplicates(subset=[feature_col, target_col], keep="first")

    if clip_outliers:
        for col in [feature_col, target_col]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=low, upper=high)

    return df[[feature_col, target_col]].reset_index(drop=True)

def make_features(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (2D) and y (1D) ready for modeling."""
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Found NA values in features/target after cleaning.")
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    return X, y
