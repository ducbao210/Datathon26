import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_mae(y_true, y_pred):
    """Tính Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def get_rmse(y_true, y_pred):
    """Tính Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_mape(y_true, y_pred):
    """Tính Mean Absolute Percentage Error (có xử lý lỗi chia cho 0)"""
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100


def get_r2(y_true, y_pred):
    """Tính R-squared (Coefficient of Determination)"""
    return r2_score(y_true, y_pred)


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": get_mae(y_true, y_pred),
        "RMSE": get_rmse(y_true, y_pred),
        "MAPE": get_mape(y_true, y_pred),
        "R2": get_r2(y_true, y_pred),
    }
