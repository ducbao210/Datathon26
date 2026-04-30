import json

import pandas as pd

from ..optimizers import (
    tune_arima,
    tune_catboost,
    tune_lightgbm,
    tune_lstm,
    tune_prophet,
    tune_transformer,
)


def run_all_optimizations(df, date_col, feature_cols, target_cols, n_trials=20):
    results = {}
    for target in target_cols:
        print(f"Running Optuna studies for target: {target}")
        results[target] = {}

        results[target]["catboost"] = _pack(
            *tune_catboost(df, feature_cols, target, n_trials=n_trials)
        )
        results[target]["prophet"] = _pack(
            *tune_prophet(df, date_col, target, n_trials=n_trials)
        )
        results[target]["lightgbm"] = _pack(
            *tune_lightgbm(df, feature_cols, target, n_trials=n_trials)
        )
        results[target]["lstm"] = _pack(
            *tune_lstm(df, feature_cols, target, n_trials=max(10, n_trials // 2))
        )
        results[target]["transformer"] = _pack(
            *tune_transformer(df, feature_cols, target, n_trials=max(10, n_trials // 2))
        )
        results[target]["arima"] = _pack(*tune_arima(df, target, n_trials=n_trials))

    return results


def _pack(best_params, best_score):
    return {"best_params": best_params, "best_rmse": float(best_score)}


if __name__ == "__main__":
    # Example usage:
    # python optimize_models.py
    # Update path/columns for your dataset before running.
    data_path = "data.csv"
    date_col = "Date"
    target_cols = ["Revenue", "COGS"]

    df = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in [date_col] + target_cols]
    output = run_all_optimizations(df, date_col, feature_cols, target_cols, n_trials=20)

    with open("best_optuna_params.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Saved best params to best_optuna_params.json")
