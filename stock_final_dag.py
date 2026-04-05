"""
DAG TaskFlow API v4 - Hệ thống dự đoán chứng khoán tự động
Phiên bản đã fix lỗi {{ ds }} và tối ưu cho Airflow 2.8.1
"""

from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import datetime, timedelta
from typing import cast

import os

# ====================== CẤU HÌNH ======================
DATA_DIR = Variable.get(
    "stock_data_dir",
    default_var="/home/duckanh/airflow/data/stock"
)

TICKER = Variable.get("stock_ticker", default_var="DIG.VN")
START_DATE_STR = "2023-01-01"

# Tạo thư mục dữ liệu
os.makedirs(DATA_DIR, exist_ok=True)

# Đường dẫn các file
RAW_FILE    = os.path.join(DATA_DIR, "raw.csv")
PROC_FILE   = os.path.join(DATA_DIR, "processed.csv")
MODEL_FILE  = os.path.join(DATA_DIR, "model.h5")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.gz")


@dag(
    dag_id="stock_automl_full_taskflow_v4",
    description="Hệ thống học máy tự động dự đoán chứng khoán sử dụng LSTM + Inference",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "duckanh",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=60),
    },
    tags=["stock", "lstm", "mlops", "inference", "tttn"],
)
def stock_prediction_pipeline():

    # ==================== TASK 2.1: Thu thập dữ liệu ====================
    @task(task_id="ingestion_module", retries=5)
    def collect_data_task(**context):           # <-- Thêm **context
        from modules.ingestion import collect_data
        
        # Lấy ngày thực tế từ Airflow Context
        execution_date = context['ds']          # Ví dụ: '2026-04-05'
        
        print(f"--- Task: Thu thập dữ liệu {TICKER} đến ngày {execution_date} ---")
        
        success = collect_data(
            ticker=TICKER,
            start_date=START_DATE_STR,
            end_date=execution_date,            # Truyền ngày thực tế
            save_path=RAW_FILE
        )
        if not success:
            raise ValueError(f"Thu thập dữ liệu cho {TICKER} thất bại!")
        return RAW_FILE

    # ==================== TASK 2.2: Xử lý dữ liệu ====================
    @task(task_id="processing_module")
    def preprocess_task(raw_file):
        from modules.processing import preprocess_and_sync
        print("--- Task: Xử lý và đồng bộ dữ liệu ---")
        success = preprocess_and_sync(
            input_path=raw_file,
            output_path=PROC_FILE,
            scaler_path=SCALER_FILE
        )
        if not success:
            raise ValueError("Xử lý dữ liệu thất bại!")
        return PROC_FILE

    # ==================== TASK 2.3: Huấn luyện mô hình ====================
    @task(task_id="modeling_module")
    def modeling_task(processed_file):
        from modules.modeling import build_and_train
        print("--- Task: Huấn luyện mô hình LSTM ---")
        success = build_and_train(
            data_path=processed_file,
            model_path=MODEL_FILE
        )
        if not success:
            raise ValueError("Huấn luyện mô hình thất bại!")
        return MODEL_FILE

    # ==================== TASK 2.4: Đánh giá mô hình ====================
    @task(task_id="evaluation_module")
    def evaluation_task(processed_file, model_file):
        from modules.evaluation import evaluate_model
        print("--- Task: Đánh giá mô hình ---")
        success = evaluate_model(
            data_path=processed_file,
            model_path=model_file
        )
        if not success:
            raise ValueError("Đánh giá mô hình thất bại!")
        return "Evaluation completed"

    # ==================== TASK 2.5: Dự báo tương lai ====================
    @task(task_id="inference_module")
    def inference_task(processed_file, model_file):
        from modules.inference import predict_future
        print("--- Task: Dự báo giá 5 ngày tới ---")
        results = predict_future(
            model_path=model_file,
            scaler_path=SCALER_FILE,
            processed_data_path=processed_file,
            days_to_predict=5,
            seq_length=60
        )
        return results

    # ==================== XÂY DỰNG PIPELINE ====================
    raw = collect_data_task()
    
    processed = preprocess_task(cast(str, raw))
    model = modeling_task(cast(str, processed))
    
    evaluation_task(cast(str, processed), cast(str, model))
    inference_task(cast(str, processed), cast(str, model))


# Khởi tạo DAG
stock_prediction_pipeline()
