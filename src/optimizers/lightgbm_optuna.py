import lightgbm as lgb
import optuna

from .optuna_utils import rmse, train_val_split


def tune_lightgbm(df, feature_cols, target_col, n_trials=30, n_jobs=1, random_seed=42):
    train_df, val_df = train_val_split(df)
    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_val = val_df[feature_cols]
    y_val = val_df[target_col]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": random_seed,
            "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        return rmse(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value
