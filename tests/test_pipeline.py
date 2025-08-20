# tests/test_pipeline.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data_preprocessing import clean_salary_data, make_features
from model_training import train_model, predict, evaluate, save_model, load_model

def test_cleaning_and_features():
    # noisy input (strings, currency, commas, missing)
    df = pd.DataFrame({
        "Level": ["1", "2", "₹3", None, "bad", "5,000"],
        "Salary": ["1,000", "2,000", "3000", None, "oops", "6,000"]
    })

    df_clean = clean_salary_data(df)
    X, y = make_features(df_clean, ["Level"], "Salary")

    assert X.shape[0] == y.shape[0]
    assert X.dtypes["Level"] == "float64"
    assert y.dtype == "float64"
    assert not X.isna().any().any()
    assert not y.isna().any()

def test_train_predict_evaluate_and_persist(tmp_path):
    # perfect linear relation
    df = pd.DataFrame({
        "Level": [1, 2, 3, 4, 5],
        "Salary": [1000, 2000, 3000, 4000, 5000]
    })
    X, y = make_features(df, ["Level"], "Salary")

    model = train_model(X, y)
    assert isinstance(model, LinearRegression)

    y_pred = predict(model, X)
    assert len(y_pred) == len(y)

    metrics = evaluate(y, y_pred)
    assert 0.99 <= metrics["r2"] <= 1.0
    for v in metrics.values():
        assert isinstance(v, float)

    # save -> load -> predictions should match
    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)
    loaded = load_model(model_path)

    y_pred2 = predict(loaded, X)
    assert np.allclose(y_pred, y_pred2)