import optuna
from statsmodels.tsa.arima.model import ARIMA

from .optuna_utils import rmse, train_val_split


def tune_arima(df, target_col, n_trials=30, n_jobs=1):
    train_df, val_df = train_val_split(df)
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values

    def objective(trial):
        p = trial.suggest_int("p", 0, 8)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 8)

        model = ARIMA(y_train, order=(p, d, q))
        fit_res = model.fit()
        preds = fit_res.forecast(steps=len(y_val))
        return rmse(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value
