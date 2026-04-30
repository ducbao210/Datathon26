import optuna
import gc
from catboost import CatBoostError, CatBoostRegressor
from .optuna_utils import rmse, train_val_split


def tune_catboost(df, feature_cols, target_col, n_trials=30, n_jobs=1, random_seed=42):
    train_df, val_df = train_val_split(df)

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_val = val_df[feature_cols]
    y_val = val_df[target_col]

    def objective(trial):
        # Giải phóng bộ nhớ trước mỗi trial
        gc.collect()

        params = {
            "iterations": trial.suggest_int(
                "iterations", 300, 2000
            ),  # Giảm bớt số lượng cây
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),  # GIẢM ĐỘ SÂU: Max 8 thay vì 10
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "random_seed": random_seed,
            "verbose": 0,
            "thread_count": 1,
            "allow_writing_files": False,
        }

        try:
            model = CatBoostRegressor(**params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
            preds = model.predict(x_val)
            return rmse(y_val, preds)
        except (CatBoostError, Exception) as e:  # Bắt mọi lỗi để tránh sập study
            print(f"[Warning] Trial failed due to error: {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    # Đảm bảo n_jobs=1 nếu RAM hạn chế
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value
