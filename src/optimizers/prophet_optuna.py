import optuna
from prophet import Prophet

from .optuna_utils import rmse, train_val_split


def tune_prophet(df, date_col, target_col, n_trials=30, n_jobs=1):
    train_df, val_df = train_val_split(df)
    prophet_train = train_df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
    prophet_val = val_df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})

    def objective(trial):
        params = {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 20.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        }
        model = Prophet(**params)
        model.fit(prophet_train)
        preds = model.predict(prophet_val[["ds"]])["yhat"].values
        return rmse(prophet_val["y"].values, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value
