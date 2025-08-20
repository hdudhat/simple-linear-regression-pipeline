# Simple Linear Regression Pipeline (Educational Project)

A small end-to-end ML pipeline project to demonstrate **data preprocessing, model training, evaluation, and testing** in Python.

## Project structure

├── data/
│ └── raw/ # input CSV (Position_Salaries.csv)
├── data_preprocessing.py # data loading + cleaning + features
├── model_training.py # model training + save/load + eval
├── main.py # orchestrates the pipeline
├── tests/ # pytest-based unit tests
└── requirements.txt

## How to run

```bash
# create environment
python -m venv .venv
.venv\Scripts\activate  # (Windows)

pip install -r requirements.txt

# run pipeline
python main.py

# run tests
pytest -q

=== Simple Linear Regression (from scratch) ===
Feature: Level | Target: Salary
R2   : 0.9213
MAE  : 1200.53
MSE  : 3500000.00
RMSE : 1870.83
Model saved at: data/processed/model.pkl
```
