![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Final-orange)

# Datathon 2026 - Dự báo Doanh thu & Giá vốn hàng bán (COGS)
Dự báo các chỉ số tài chính (Revenue & COGS) cho ngành bán lẻ/thương mại điện tử sử dụng 6 mô hình Học máy và Học sâu:
- Mô hình Thống kê & Chuỗi thời gian:
    - ARIMA
    - Prophet
- Mô hình Dựa trên cây (Tree-based):
    - LightGBM
    - CatBoost
- Mô hình Học sâu (Deep Learning):
    - LSTM
    - Transformer
- Mô hình Lai (Đề xuất chính):
    - Prophet + LightGBM Residuals

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Metrics](#metrics) 


## Tổng quan
- Chủ đề: Bán lẻ & Thương mại điện tử
- Mục đích: Dự án Datathon này nhằm dự đoán Doanh thu và COGS trong tương lai sử dụng Học máy, dựa trên các thông tin lịch sử hiện có như:
    - Đặc trưng trễ (Lag) và Cửa sổ trượt (Rolling windows).
    - Các sự kiện khuyến mãi, cờ cộng dồn và số chiến dịch đang chạy.
    - Lưu lượng truy cập Web (số phiên, số lượt xem trang, tỷ lệ thoát).
    - Yếu tố thời gian có tính chu kỳ (Chuỗi Fourier, kỳ nghỉ Tết Nguyên Đán).
- Mục tiêu: Dự án đánh giá và so sánh hiệu suất của nhiều mô hình, bao gồm **Prophet**, **CatBoost**, **LightGBM**, **LSTM**, và **HybridResidualModel**.
Dự án cũng cung cấp một luồng xử lý (pipeline) hoàn chỉnh từ tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá, tinh chỉnh siêu tham số, cho đến recursive forecasting.

## Tính năng
- Làm sạch và tiền xử lý dữ liệu chuỗi thời gian thô nhằm ngăn chặn rò rỉ dữ liệu (data leakage).
- Kỹ thuật trích xuất đặc trưng nâng cao (Lags, EMA, Mã hóa chu kỳ) để cải thiện hiệu suất mô hình.
- Huấn luyện đồng thời nhiều mô hình Học máy và Học sâu.
- Tự động hóa quá trình tối ưu siêu tham số bằng Optuna. 
- Đánh giá mô hình thông qua các chỉ số chuẩn xác: RMSE, MAE, và MAPE.
- Triển khai cơ chế recursive forecasting cho các dự báo dài hạn. 
- Pipeline dễ dàng mở rộng để cắm thêm các thuật toán hoặc tập dữ liệu mới.

## Công nghệ sử dụng

- Python: 3.9+

### 1. Xử lý & Thao tác dữ liệu

- NumPy – tính toán mảng số học

- Pandas – thao tác dữ liệu và phân tích chuỗi thời gian

- Pathlib, os, datetime – quản lý đường dẫn file và tiện ích thời gian

### 2. Tiền xử lý Dữ liệu

- Scikit-learn – công cụ tiền xử lý (MinMaxScaler) và pipeline

### 3. Học máy & Mô hình hóa

- Prophet – mô hình dự báo xu hướng và tính mùa vụ

- Statsmodels – xây dựng mô hình thống kê ARIMA

- LightGBM, CatBoost – các mô hình Gradient Boosting hiệu suất cao

- PyTorch – framework Học sâu để xây dựng LSTM và Transformer

### 4. Tối ưu hóa Siêu tham số

- Optuna – tự động hóa quá trình dò tìm và tối ưu siêu tham số

### 5. Trực quan hóa & Giải thích Mô hình

- Matplotlib – vẽ biểu đồ và đồ thị thống kê

- SHAP, Feature Importance, PDP – giải thích mô hình và phân tích mức độ quan trọng của đặc trưng


## Tập dữ liệu

- **Tập dữ liệu được cung cấp bởi Ban tổ chức Datathon**:  
Bao gồm dữ liệu bán hàng lịch sử (`sales.csv`) và tệp mẫu dự đoán (`sample_submission.csv`). 

- **Lưu ý**: Đảm bảo rằng tập dữ liệu được đặt chính xác vào thư mục `dataset/` trước khi chạy pipeline.

## Cấu trúc dự án
```bash
    Datathon26/
    ├── dataset/                        # Chứa dữ liệu thô và đã làm sạch (sales.csv)
    ├── output/                         # Chứa các file dự báo cuối cùng (submission.csv)
    ├── notebooks/                      
    │   ├── main_pipeline.ipynb         # Chạy pipeline end-to-end
    │   └── summary.ipynb               # Trực quan hóa dữ liệu huấn luyện
    ├── src/
    │   ├── explainability/             # Package hỗ trợ giải thích mô hình (SHAP/FI/ PDP)
    │   │   ├── __init__.py
    │   │   └── explainability.py
    │   ├── features/                   # Package tiền xử lý và trích xuất đặc trưng
    │   │   ├── __init__.py
    │   │   └── feature_engineering.py
    │   ├── metrics/                    # Package lưu trữ các hàm đánh giá
    │   │   ├── __init__.py
    │   │   └── metrics.py
    │   ├── models/                     # Package huấn luyện và định nghĩa kiến trúc mô hình
    │   │   ├── __init__.py
    │   │   └── model_trainer.py
    │   ├── optimization/               # Package fine-tune các sigle models
    │   │   ├── __init__.py
    │   │   └── optimization.py
    │   └── optimizers/                 # Các kịch bản Optuna riêng biệt cho từng mô hình
    │       ├── __init__.py
    │       ├── arima_optuna.py
    │       ├── catboost_optuna.py
    │       ├── lightgbm_optuna.py
    │       ├── lstm_optuna.py
    │       ├── optuna_utils.py
    │       ├── prophet_optuna.py
    │       └── transformer_optuna.py
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt
```
## Cài đặt

1. **Clone repository về máy:**
    - Clone repo
    ```bash
    git clone https://github.com/ducbao210/Datathon26.git
    ```
    - Change direction
    ```bash
    cd ../Datathon26
    ```
    ```bash
    cd /d ../Datathon26
    ```
2. Tạo môi trường ảo (Khuyên dùng).
    - Sử dụng cmd-Windows / macOS terminal
    ```bash
    python -m venv venv 
    ```
    - Sử dụng Windows powershell
    ```bash
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    python -m venv venv
    ```
    - Kích hoạt môi trường ảo
        - Windows - sử dụng cmd
        ```bash
        venv\Scripts\activate 
        ```
        - Linux/Mac
        ```bash
        source venv/bin/activate
        ```

3. Tải và cài đặt các thư viện yêu cầu.
    ```bash
    pip install -r requirements.txt
    ```
4. Trực quan hoá dữ liệu.
    - Run các cells trong file notebook `summary.ipynb` để xem trend, phân phối dữ liệu... của tập dữ liệu `sales.csv`.
5. Pipe line chính - Timeseries Forcasting.
    - Run các cells trong file note book `main_pipeline.ipynb`.
    - Pipeline bao gồm:
        + Đọc dữ liệu từ file config.ini
        + Chia tập train/valid theo `split_date`
        + Hiệu chỉnh siêu tham số của các models.
        + Chọn model dựa trên metric `RMSE` cho từng biến target.
        + Sử dụng các best models trên từng biến target để dự đoán.
        + Vẽ các biểu đồ SHAP, Feature Importance, PDP để giải thích mô hình

## Configuration file:
    - Lưu các directory của `dataset`, `output`...
    - Lưu mốc thời gian để chia tập dữ liệu train/val.
    - Lưu các settings của optuna, và target columns.
```bash
[PATHS]
DATA_DIR = dataset
OUTPUT_DIR = output
TRAIN_FILE = dataset/sales.csv
TEST_FILE = dataset/sample_submission.csv
OUT_FILE = output/submission.csv

[VALID]
split_date = 2022-01-01

[SETTINGS]
N_TRIALS = 30
# Full cores
N_JOBS = -1 
TARGET_COLS = Revenue, COGS
```

## Metrics

Sau khi chạy luồng ML pipeline hoàn chỉnh cùng tính năng tự động tối ưu hóa siêu tham số của Optuna (30 trials cho mỗi mô hình), chúng tôi thu được các chỉ số hiệu suất sau trên tập Validation.

**Lưu ý**: Các chỉ số có thể thay đổi nhẹ sau mỗi lần huấn luyện; kết quả dưới đây đại diện cho một lần chạy tiêu biểu.
### Revenue
| Rank | Model            | MAE          | RMSE         | MAPE       | R2        |
|------|------------------|--------------|--------------|------------|-----------|
| 1    | CatBoost_Default | 6.025845e+05 | 7.746742e+05 | 25.184866  | 0.785797  |
| 2    | CatBoost         | 6.001841e+05 | 7.767493e+05 | 24.565779  | 0.784648  |
| 3    | LightGBM         | 6.368707e+05 | 8.228433e+05 | 27.430566  | 0.758330  |
| 4    | Hybrid           | 6.850462e+05 | 8.922849e+05 | 23.616578  | 0.715819  |
| 5    | LSTM             | 8.673502e+05 | 1.185817e+06 | 25.128402  | 0.498093  |
| 6    | Transformer      | 1.053783e+06 | 1.390080e+06 | 39.062051  | 0.310288  |
| 7    | Prophet          | 1.290420e+06 | 1.796211e+06 | 35.594855  | -0.151602 |
| 8    | ARIMA            | 1.519176e+06 | 2.123129e+06 | 41.385803  | -0.608943 |

**🏆 Best model for Revenue:**  
**CatBoost_Default** — RMSE: **774,674.22**
### COGS

| Rank | Model            | MAE          | RMSE         | MAPE       | R2        |
|------|------------------|--------------|--------------|------------|-----------|
| 1    | CatBoost         | 4.900783e+05 | 6.475990e+05 | 22.314793  | 0.802867  |
| 2    | CatBoost_Default | 5.206648e+05 | 6.744026e+05 | 24.715843  | 0.786211  |
| 3    | LightGBM         | 5.408947e+05 | 7.094137e+05 | 24.996923  | 0.763437  |
| 4    | Hybrid           | 5.679532e+05 | 7.360745e+05 | 22.410983  | 0.745323  |
| 5    | LSTM             | 6.209393e+05 | 8.654275e+05 | 22.439999  | 0.647947  |
| 6    | Transformer      | 8.032693e+05 | 1.139743e+06 | 31.379857  | 0.389395  |
| 7    | ARIMA            | 1.161755e+06 | 1.603263e+06 | 44.400277  | -0.208249 |
| 8    | Prophet          | 1.213113e+06 | 1.540600e+06 | 48.086824  | -0.115647 |

**🏆 Best model for COGS:**  
**CatBoost** — RMSE: **647,599.00**

---

Sử dụng `RMSE` làm tiêu chí đánh giá chính, kết quả cho thấy mô hình **CatBoost-based** vượt trội hơn so với các mô hình còn lại ở cả hai target.

- Với **Revenue**, mô hình tốt nhất là `CatBoost_Default`, đạt RMSE thấp nhất khoảng **774,674**, cho thấy khả năng dự báo ổn định hơn so với LightGBM, Hybrid và các mô hình deep learning như LSTM/Transformer.
  
- Với **COGS**, mô hình tốt nhất là `CatBoost`, đạt RMSE khoảng **647,599**, tiếp tục khẳng định hiệu quả của boosting tree trong bài toán này.

---

## LICENSE
[MIT License](LICENSE)