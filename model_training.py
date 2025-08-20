# model_training.py
from __future__ import annotations
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump, load

def train_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model: LinearRegression, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)

def evaluate(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }

def save_model(model: LinearRegression, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    return path

def load_model(path: str | Path) -> LinearRegression:
    return load(path)
