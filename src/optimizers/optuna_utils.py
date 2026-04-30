import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_val_split(df, val_ratio=0.2):
    split_idx = int(len(df) * (1 - val_ratio))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid split index. Check data size and val_ratio.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
