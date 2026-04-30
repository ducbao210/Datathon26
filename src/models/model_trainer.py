import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Import các hàm đánh giá từ file metrics.py cùng thư mục
from ..metrics import get_mae, get_rmse, get_mape, evaluate_regression


class HybridResidualModel:

    def __init__(self, date_col="Date", target_cols=["Revenue", "COGS"]):
        self.date_col = date_col
        self.target_cols = target_cols

        self.prophet_models = {}
        self.lgb_models = {}

        self.feature_cols = None

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, train_df, feature_cols, prophet_params=None, lgb_params=None):
        self.feature_cols = feature_cols
        train_df = train_df.copy()
        prophet_params = prophet_params or {}
        lgb_params = lgb_params or {}

        for target in self.target_cols:
            print(f"\n========== TRAIN {target} ==========")

            # -------------------------------------------------
            # 1. PROPHET
            # -------------------------------------------------
            prophet_df = train_df[[self.date_col, target]].rename(
                columns={self.date_col: "ds", target: "y"}
            )

            default_prophet_params = {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05,
            }
            default_prophet_params.update(prophet_params)
            prophet_model = Prophet(**default_prophet_params)

            prophet_model.fit(prophet_df)
            self.prophet_models[target] = prophet_model

            # -------------------------------------------------
            # 2. TRAIN PREDICTION
            # -------------------------------------------------
            prophet_pred = prophet_model.predict(prophet_df[["ds"]])

            train_df[f"prophet_pred_{target}"] = prophet_pred["yhat"].values

            # -------------------------------------------------
            # 3. RESIDUAL
            # -------------------------------------------------
            train_df[f"residual_{target}"] = (
                train_df[target] - train_df[f"prophet_pred_{target}"]
            )

            # -------------------------------------------------
            # 4. LIGHTGBM
            # -------------------------------------------------
            cols_needed = self.feature_cols + [f"residual_{target}"]

            temp_df = train_df.dropna(subset=cols_needed)

            X_train = temp_df[self.feature_cols]

            y_train = temp_df[f"residual_{target}"]

            default_lgb_params = {
                "n_estimators": 1000,
                "learning_rate": 0.02,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 1.0,
                "importance_type": "gain",
                "random_state": 42,
                "verbosity": -1,
            }
            default_lgb_params.update(lgb_params)
            lgb_model = lgb.LGBMRegressor(**default_lgb_params)

            lgb_model.fit(X_train, y_train)

            self.lgb_models[target] = lgb_model

        print("\nHybrid training completed.\n")

    # =========================================================
    # VALIDATION
    # =========================================================
    def evaluate(self, val_df):
        val_df = val_df.copy()
        metrics = {}

        for target in self.target_cols:
            print(f"\n========== EVALUATE {target} ==========")

            # -------------------------------------------------
            # PROPHET PRED
            # -------------------------------------------------
            prophet_model = self.prophet_models[target]

            prophet_input = val_df[[self.date_col]].rename(
                columns={self.date_col: "ds"}
            )

            prophet_pred = prophet_model.predict(prophet_input)

            val_df[f"prophet_pred_{target}"] = prophet_pred["yhat"].values

            # -------------------------------------------------
            # LGB RESIDUAL PRED
            # -------------------------------------------------
            X_val = val_df[self.feature_cols]

            residual_pred = self.lgb_models[target].predict(X_val)

            # -------------------------------------------------
            # FINAL
            # -------------------------------------------------
            val_df[f"final_pred_{target}"] = (
                val_df[f"prophet_pred_{target}"] + residual_pred
            )

            val_df[f"final_pred_{target}"] = np.clip(
                val_df[f"final_pred_{target}"], 0, None
            )

            # -------------------------------------------------
            # METRICS (Sử dụng hàm từ metrics.py)
            # -------------------------------------------------
            y_true = val_df[target]
            y_pred = val_df[f"final_pred_{target}"]

            metrics[target] = evaluate_regression(y_true, y_pred)

            print(f"MAE  : {metrics[target]['MAE']:.2f}")
            print(f"RMSE : {metrics[target]['RMSE']:.2f}")
            print(f"MAPE : {metrics[target]['MAPE']:.2f}%")
            print(f"R2   : {metrics[target]['R2']:.4f}")

        return metrics

    # =========================================================
    # RECURSIVE FORECAST
    # =========================================================
    def recursive_forecast(
        self, history_df, future_dates, fe_pipeline, buffer_days=120
    ):
        history_df = history_df.copy()
        predictions = []
        prophet_future_df = pd.DataFrame({"ds": future_dates})
        prophet_forecasts = {}

        for target in self.target_cols:
            preds = self.prophet_models[target].predict(prophet_future_df)
            # Lưu lại dạng Series với index là ngày để tra cứu cho nhanh
            prophet_forecasts[target] = preds.set_index("ds")["yhat"]
        for future_date in future_dates:

            # -------------------------------------------------
            # BUFFER
            # -------------------------------------------------
            buffer_df = history_df.tail(buffer_days).copy()

            # -------------------------------------------------
            # CREATE NEXT ROW
            # -------------------------------------------------
            next_row = {self.date_col: future_date}

            for target in self.target_cols:
                next_row[target] = np.nan
            for col in self.feature_cols:
                if col not in next_row:
                    next_row[col] = np.nan
            next_row = pd.DataFrame([next_row])

            temp_df = pd.concat([buffer_df, next_row], ignore_index=True)

            # -------------------------------------------------
            # FEATURE ENGINEERING
            # -------------------------------------------------
            temp_feat = fe_pipeline.transform(temp_df)

            X_next = temp_feat.iloc[[-1]][self.feature_cols]

            pred_dict = {self.date_col: future_date}

            # -------------------------------------------------
            # PREDICT EACH TARGET
            # -------------------------------------------------
            for target in self.target_cols:

                # Prophet forecast
                prophet_model = self.prophet_models[target]

                prophet_future = pd.DataFrame({"ds": [future_date]})

                prophet_base = prophet_forecasts[target].loc[future_date]

                # Residual prediction
                residual_pred = self.lgb_models[target].predict(X_next)[0]

                # Final prediction
                final_pred = prophet_base + residual_pred

                final_pred = max(final_pred, 0)

                pred_dict[target] = final_pred
                next_row[target] = final_pred

            # -------------------------------------------------
            # APPEND HISTORY
            # -------------------------------------------------
            history_df = pd.concat([history_df, next_row], ignore_index=True)

            predictions.append(pred_dict)

        return pd.DataFrame(predictions)


from catboost import CatBoostRegressor


class CatBoostTrainer:
    def __init__(self, target_cols=["Revenue", "COGS"]):
        self.target_cols = target_cols
        self.models = {}

    def fit(self, train_df, val_df, feature_cols, model_params=None):
        model_params = model_params or {}
        for target in self.target_cols:
            print(f"Training CatBoost for {target}...")
            default_params = {
                "iterations": 2000,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 3,
                "loss_function": "RMSE",
                "eval_metric": "MAE",
                "random_seed": 42,
                "verbose": 200,
                "early_stopping_rounds": 100,
            }
            default_params.update(model_params)
            model = CatBoostRegressor(**default_params)
            model.fit(
                train_df[feature_cols],
                train_df[target],
                eval_set=(val_df[feature_cols], val_df[target]),
                use_best_model=True,
            )
            self.models[target] = model

    def recursive_forecast(
        self,
        history_df,
        future_dates,
        fe_pipeline,
        feature_cols,
        target_cols=["Revenue", "COGS"],
    ):
        current_df = history_df.copy()
        predictions = []

        for date in future_dates:
            new_row = pd.DataFrame({"Date": [date]})
            for col in target_cols:
                new_row[col] = np.nan

            combined_df = pd.concat([current_df, new_row], ignore_index=True)
            processed_df = fe_pipeline.transform(combined_df)
            target_row = processed_df[processed_df["Date"] == date]

            preds_dict = {}
            for target in target_cols:
                # SỬA LỖI: Sử dụng self.models thay vì model.models
                preds_dict[target] = self.models[target].predict(
                    target_row[feature_cols]
                )[0]

            preds_dict["Date"] = date
            predictions.append(preds_dict)

            for target in target_cols:
                new_row[target] = preds_dict[target]
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            current_df = current_df.tail(150)

        return pd.DataFrame(predictions)

    def predict(self, df, feature_cols):
        res = df[["Date"]].copy()
        for target in self.target_cols:
            res[target] = self.models[target].predict(df[feature_cols])
        return res


# =========================================================
# ARIMA TRAINER (Sử dụng statsmodels)
# =========================================================
class ARIMATrainer:
    def __init__(self, target_cols=["Revenue", "COGS"], order=(5, 1, 0)):
        """
        order: (p, d, q) - tham số của mô hình ARIMA.
        Mặc định (5, 1, 0) thường ổn cho dữ liệu có trend.
        """
        self.target_cols = target_cols
        self.order = order
        self.models_fit = {}

    def fit(self, train_df):
        """
        Fit mô hình cho từng cột mục tiêu.
        """
        for target in self.target_cols:
            print(f"Đang huấn luyện ARIMA{self.order} cho {target}...")
            # endog là chuỗi thời gian mục tiêu
            model = ARIMA(train_df[target].values, order=self.order)
            self.models_fit[target] = model.fit()
            print(f"-> Hoàn tất fit ARIMA cho {target}")

    def predict(self, future_dates):
        """
        future_dates: list các ngày cần dự báo
        """
        horizon = len(future_dates)
        predictions = {"Date": future_dates}

        for target, model_res in self.models_fit.items():
            forecast_values = model_res.forecast(steps=horizon)
            predictions[target] = np.clip(forecast_values, 0, None)

        return pd.DataFrame(predictions)


# =========================================================
# 1. KIẾN TRÚC MÔ HÌNH (LSTM & TRANSFORMER)
# =========================================================


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Lấy output của time step cuối cùng
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.embedding = nn.Linear(input_dim, d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out


# =========================================================
# 2. TRAINER CHO DEEP LEARNING
# =========================================================


class DeepLearningTrainer:
    def __init__(self, model_type="lstm", target_cols=["Revenue", "COGS"], seq_len=14):
        self.model_type = model_type
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.models = {}
        self.scalers = {col: MinMaxScaler() for col in target_cols}
        self.feature_scaler = MinMaxScaler()

        # Kiểm tra thiết bị
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Khởi tạo {model_type.upper()} trên thiết bị: {self.device}")

    def fit(self, train_df, feature_cols, epochs=50, batch_size=32, model_params=None):
        model_params = model_params or {}
        hidden_dim = int(model_params.get("hidden_dim", 64))
        num_layers = int(model_params.get("num_layers", 2))
        d_model = int(model_params.get("d_model", 64))
        nhead = int(model_params.get("nhead", 4))
        lr = float(model_params.get("lr", 0.001))
        epochs = int(model_params.get("epochs", epochs))
        batch_size = int(model_params.get("batch_size", batch_size))
        self.feature_scaler.fit(train_df[feature_cols].fillna(0))
        for target in self.target_cols:
            print(f"Đang huấn luyện {self.model_type.upper()} cho {target}...")
            X_train, y_train = self._prepare_data(train_df, feature_cols, target)
            dataset = TensorDataset(X_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            if self.model_type == "lstm":
                model = LSTMModel(
                    input_dim=len(feature_cols),
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                )
            else:
                model = TransformerModel(
                    input_dim=len(feature_cols),
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    output_dim=1,
                )

            # CHUYỂN MODEL SANG GPU/CPU
            model.to(self.device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in loader:
                    # CHUYỂN DỮ LIỆU BATCH SANG GPU/CPU
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

            self.models[target] = model

    def predict(self, history_df, future_dates, feature_cols, fe_pipeline):
        """
        history_df: Dữ liệu quá khứ để lấy seq_len cuối cùng
        """
        df_current = history_df.copy()
        predictions = []

        for future_date in future_dates:
            next_row_preds = {"Date": future_date}

            # 1. Tính toán lại features cho đoạn cuối của tập current
            # (Chỉ cần lấy đủ lượng data > seq_len để tạo feature)
            buffer_df = df_current.tail(self.seq_len + 60)
            temp_feat = fe_pipeline.transform(buffer_df)

            # Lấy seq_len dòng cuối cùng sau khi transform
            X_feat = self.feature_scaler.transform(temp_feat[feature_cols].fillna(0))
            X_input = (
                torch.tensor(X_feat[-self.seq_len :], dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
            )

            next_row_df = {"Date": future_date}

            # 2. Dự báo từng target
            for target in self.target_cols:
                model = self.models[target]
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(X_input).cpu().numpy()

                pred_value = (
                    self.scalers[target].inverse_transform(pred_scaled).flatten()[0]
                )
                pred_value = max(pred_value, 0)

                next_row_preds[target] = pred_value
                next_row_df[target] = pred_value  # Lưu lại để append vào lịch sử

            # 3. Cập nhật lịch sử để dự báo step tiếp theo
            df_current = pd.concat(
                [df_current, pd.DataFrame([next_row_df])], ignore_index=True
            )
            predictions.append(next_row_preds)

        return pd.DataFrame(predictions)

    def _prepare_data(self, df, feature_cols, target_col):
        """
        Chuyển dữ liệu từ DataFrame sang Tensor dạng Sequence (Samples, Seq_Len, Features)
        """
        # Chuẩn hóa Features
        X_data = self.feature_scaler.fit_transform(df[feature_cols].fillna(0))
        # Chuẩn hóa Target
        y_data = self.scalers[target_col].fit_transform(df[[target_col]].fillna(0))

        X_seq, y_seq = [], []
        for i in range(len(df) - self.seq_len):
            X_seq.append(X_data[i : i + self.seq_len])
            y_seq.append(y_data[i + self.seq_len])

        return (
            torch.tensor(np.array(X_seq), dtype=torch.float32),
            torch.tensor(np.array(y_seq), dtype=torch.float32),
        )
