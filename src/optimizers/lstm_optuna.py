import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .optuna_utils import rmse, train_val_split


class _LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(x, y, seq_len):
    x_seq, y_seq = [], []
    for i in range(len(x) - seq_len):
        x_seq.append(x[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(x_seq), np.array(y_seq)


def tune_lstm(df, feature_cols, target_col, seq_len=14, n_trials=20, n_jobs=1, device=None):
    train_df, val_df = train_val_split(df)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(train_df[feature_cols].fillna(0))
    y_train = y_scaler.fit_transform(train_df[[target_col]].fillna(0))
    x_val = x_scaler.transform(val_df[feature_cols].fillna(0))
    y_val = y_scaler.transform(val_df[[target_col]].fillna(0))

    x_train_seq, y_train_seq = _make_sequences(x_train, y_train, seq_len)
    x_val_seq, y_val_seq = _make_sequences(x_val, y_val, seq_len)

    train_ds = TensorDataset(
        torch.tensor(x_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 10, 50),
        }

        model = _LSTMRegressor(len(feature_cols), params["hidden_dim"], params["num_layers"]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

        model.train()
        for _ in range(params["epochs"]):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            x_val_tensor = torch.tensor(x_val_seq, dtype=torch.float32).to(device)
            preds_scaled = model(x_val_tensor).cpu().numpy()

        preds = y_scaler.inverse_transform(preds_scaled).flatten()
        y_true = y_scaler.inverse_transform(y_val_seq).flatten()
        return rmse(y_true, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value
