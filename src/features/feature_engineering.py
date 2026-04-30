import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(
        self,
        lag_periods=[1, 2, 7, 14, 30, 60],
        ema_windows=[7, 14, 30],
        fourier_orders=3,
    ):
        """
        Parameters
        ----------
        lag_periods : list
            Các lag dùng để tạo historical features
        ema_windows : list
            Các span dùng cho EMA và Rolling
        fourier_orders : int
            Số bậc Fourier terms để bắt mùa vụ
        """
        self.lag_periods = lag_periods
        self.ema_windows = ema_windows
        self.fourier_orders = fourier_orders

        # Biến lưu trữ thông tin học được từ tập Train
        self.historical_mappings = {}
        self.revenue_baseline_100 = None

    # =========================================================
    # BƯỚC 1: FIT (HỌC TỪ TẬP TRAIN)
    # =========================================================
    def fit(self, df_train, target_cols=["Revenue", "COGS"]):
        """
        Hàm này CHỈ ĐƯỢC CHẠY TRÊN TẬP TRAIN để tránh Data Leakage.
        Dùng để tính toán Target Encoding (Historical Mapping) và Baseline.
        """
        print("Đang fit dữ liệu lịch sử từ tập Train...")
        df = self.create_time_features(df_train.copy())
        for target in target_cols:
            if target in df.columns:
                # 1. Historical Mapping: Trung bình doanh thu theo Tháng & Thứ
                mapping = (
                    df.groupby(["month", "day_of_week"])[target].mean().reset_index()
                )
                mapping.rename(
                    columns={target: f"hist_avg_{target}_m_dow"}, inplace=True
                )
                self.historical_mappings[target] = mapping

                # 2. Baseline vĩ mô: Giá trị MA 100 ở ngày cuối cùng của tập Train
                # Dùng làm mỏ neo nếu cần dự báo tỷ lệ cho 18 tháng sau
                df_sorted = df.sort_values("Date")
                self.revenue_baseline_100 = (
                    df_sorted["Revenue"].rolling(100).mean().iloc[-1]
                )
                print(
                    f"-> Đã ghi nhận Baseline MA_100: {self.revenue_baseline_100:,.2f}"
                )

        return self

    # =========================================================
    # TIME FEATURES (Không sợ Leakage, áp dụng cả Train/Test)
    # =========================================================
    def create_time_features(self, df):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        # Calendar features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day_of_month"] = df["Date"].dt.day
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["year"] = df["Date"].dt.year
        df["day_of_year"] = df["Date"].dt.dayofyear

        # Weekend & Month flags
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

        # Trend & Regime shift
        start_date = df["Date"].min()
        df["days_since_start"] = (df["Date"] - start_date).dt.days
        df["is_post_2019"] = (df["Date"] >= "2019-01-01").astype(int)

        # Cyclical Encoding (Sin/Cos)
        days_in_year = 365.25
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / days_in_year)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / days_in_year)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Fourier Terms
        for order in range(1, self.fourier_orders + 1):
            df[f"fourier_sin_year_order_{order}"] = np.sin(
                2 * np.pi * order * df["day_of_year"] / days_in_year
            )
            df[f"fourier_cos_year_order_{order}"] = np.cos(
                2 * np.pi * order * df["day_of_year"] / days_in_year
            )

        return df

    # =========================================================
    # HISTORY FEATURES (ĐÃ SỬA LỖI LEAKAGE VÀ LỖI MẤT LAG_1)
    # =========================================================
    def create_historical_features(self, df, target_cols=["Revenue", "COGS"]):
        df = df.sort_values("Date").copy()

        # Tự động lấy lag nhỏ nhất làm "mốc" (Ví dụ: [7, 14, 30] thì mốc là 7)
        if not self.lag_periods:
            self.lag_periods = [7]  # Đề phòng list rỗng
        base_lag = min(self.lag_periods)

        for col in target_cols:
            if col not in df.columns:
                continue

            # 1. LAG FEATURES
            for lag in self.lag_periods:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            # --- SỬA LỖI LEAKAGE: Dùng base_lag thay vì shift(1) ---
            base_shifted = df[col].shift(base_lag)
            shifted_7 = df[col].shift(base_lag + 7)  # Đẩy lùi thêm 7 ngày từ mốc
            shifted_30 = df[col].shift(base_lag + 30)

            # 2. DIFFERENCE FEATURES (Tính trên base_shifted)
            df[f"{col}_diff_1"] = base_shifted.diff(1)
            df[f"{col}_diff_7"] = base_shifted.diff(7)
            df[f"{col}_pct_change_1"] = base_shifted.pct_change(1)
            df[f"{col}_pct_change_7"] = base_shifted.pct_change(7)

            # 3. EMA & ROLLING FEATURES (Tính trên base_shifted)
            for span in self.ema_windows:
                df[f"{col}_ema_{span}"] = base_shifted.ewm(
                    span=span, adjust=False
                ).mean()
                rolling_obj = base_shifted.rolling(window=span)
                df[f"{col}_rolling_std_{span}"] = rolling_obj.std()
                df[f"{col}_rolling_max_{span}"] = rolling_obj.max()
                df[f"{col}_rolling_min_{span}"] = rolling_obj.min()

            # EMA RATIO FEATURE
            if f"{col}_ema_7" in df.columns and f"{col}_ema_30" in df.columns:
                df[f"{col}_ema_ratio_7_30"] = df[f"{col}_ema_7"] / (
                    df[f"{col}_ema_30"] + 1e-5
                )

            # 4. MOMENTUM FEATURES
            df[f"{col}_momentum_7"] = base_shifted - shifted_7
            df[f"{col}_momentum_30"] = base_shifted - shifted_30

        # 5. INTERACTION FEATURES (Sử dụng động theo base_lag)
        rev_base_col = f"Revenue_lag_{base_lag}"
        cogs_base_col = f"COGS_lag_{base_lag}"

        if rev_base_col in df.columns and cogs_base_col in df.columns:
            df[f"gross_margin_lag_{base_lag}"] = df[rev_base_col] - df[cogs_base_col]

            df[f"gross_margin_ratio_lag_{base_lag}"] = df[
                f"gross_margin_lag_{base_lag}"
            ] / (df[rev_base_col] + 1e-5)

            df[f"cogs_revenue_ratio_lag_{base_lag}"] = df[cogs_base_col] / (
                df[rev_base_col] + 1e-5
            )

            df[f"is_high_cogs_ratio_lag_{base_lag}"] = (
                df[f"cogs_revenue_ratio_lag_{base_lag}"] > 1.2
            ).astype(int)

        # 6. PAYDAY PERIOD
        df["is_payday_period"] = (
            df["day_of_month"].isin([25, 26, 27, 28, 29, 30, 1, 2, 3])
        ).astype(int)

        return df

    # =========================================================
    # MAIN PIPELINE (TRANSFORM)
    # =========================================================
    # =========================================================
    # MAIN PIPELINE (TRANSFORM)
    # =========================================================
    def transform(self, df_input):
        """
        Chạy cho cả tập Train và Test.
        """
        print("Đang tạo Time features...")
        df = self.create_time_features(df_input)

        print("Đang tạo Historical features (Lag, Diff, EMA)...")
        df = self.create_historical_features(df)

        # Áp dụng Historical Mapping đã học được từ tập Fit
        # Kiểm tra xem dictionary historical_mappings có dữ liệu hay không
        if hasattr(self, "historical_mappings") and self.historical_mappings:
            print("Đang Map Target Encoding (Historical Mapping)...")

            # Lặp qua từng target (Revenue, COGS,...) đã được lưu trong quá trình fit
            for target, mapping_df in self.historical_mappings.items():
                col_name = f"hist_avg_{target}_m_dow"

                # =====================================================
                # FIX DUPLICATE COLUMN
                # =====================================================
                if col_name in df.columns:
                    df = df.drop(columns=[col_name])

                # =====================================================
                # MERGE HISTORICAL MAPPING
                # =====================================================
                df = df.merge(mapping_df, on=["month", "day_of_week"], how="left")

                # =====================================================
                # FILL MISSING
                # =====================================================
                # Nếu có ngày nào ở tập Test chưa từng xuất hiện ở Train, điền bằng mean
                mean_val = mapping_df[col_name].mean()
                df[col_name] = df[col_name].fillna(mean_val)

        print("Feature engineering hoàn tất!\n")
        return df
