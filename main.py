# main.py
from sklearn.model_selection import train_test_split
from data_preprocessing import load_csv, clean_salary_data, make_features
from model_training import train_model, predict, evaluate, save_model

RAW_PATH = "data/raw/Position_Salaries.csv"
FEATURE_COL = "Level"
TARGET_COL = "Salary"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = "data/processed/model.pkl"

def run():
    # 1) load
    df_raw = load_csv(RAW_PATH)

    # 2) clean
    df_clean = clean_salary_data(
        df_raw, feature_col=FEATURE_COL, target_col=TARGET_COL, clip_outliers=True
        )

    # 3) features
    X, y = make_features(df_clean, [FEATURE_COL], TARGET_COL)

    # 4) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5) train
    model = train_model(X_train, y_train)

    # 6) save model (so tests / demo can load it)
    save_model(model, MODEL_PATH)

    # 7) predict
    y_pred = predict(model, X_test)

    # 8) evaluate
    metrics = evaluate(y_test, y_pred)

    print("=== Simple Linear Regression (from scratch) ===")
    print(f"Feature: {FEATURE_COL} | Target: {TARGET_COL}")
    for k, v in metrics.items():
        print(f"{k.upper():4}: {v:.4f}")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    run()
