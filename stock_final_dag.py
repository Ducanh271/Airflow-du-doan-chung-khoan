from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import sys
import logging

# Cấu hình Root Logger cho Airflow bắt được
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(__file__))

DATA_DIR = Variable.get("stock_data_dir", default_var="/home/duckanh/airflow/data/stock")
TICKER = Variable.get("stock_ticker", default_var="DIG.VN")
START_DATE_STR = Variable.get("stock_start_date", default_var="2023-01-01")
SEQ_LENGTH = int(Variable.get("stock_seq_length", default_var=60))
DAYS_PREDICT = int(Variable.get("stock_days_predict", default_var=5))

# Lấy danh sách ngày nghỉ lễ từ Variable (dạng chuỗi cách nhau dấu phẩy)
HOLIDAYS_STR = Variable.get("stock_holidays", default_var="2026-04-30,2026-05-01,2026-09-02")
HOLIDAYS_LIST = [h.strip() for h in HOLIDAYS_STR.split(',')]

os.makedirs(DATA_DIR, exist_ok=True)

@dag(
    dag_id="stock_automl_full_taskflow_v6",
    description="Pipeline MLOps hoàn chỉnh: No Leakage, Logging Chuẩn, Config Variables",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "duckanh",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=60), # Tăng lên 60p cho an toàn
    },
    tags=["stock", "mlops", "tttn"],
)
def stock_prediction_pipeline():

    @task(task_id="ingestion_module", retries=5)
    def collect_data_task(**context):
        from modules.ingestion import collect_data
        ds = context['ds']
        raw_file = os.path.join(DATA_DIR, f"raw_{ds}.csv")
        if not collect_data(TICKER, START_DATE_STR, ds, raw_file): raise ValueError("Lỗi Ingestion")
        return raw_file

    @task(task_id="processing_module")
    def preprocess_task(raw_file, **context):
        from modules.processing import preprocess_data
        ds = context['ds']
        proc_file = os.path.join(DATA_DIR, f"processed_{ds}.csv")
        if not preprocess_data(raw_file, proc_file): raise ValueError("Lỗi Processing")
        return proc_file

    @task(task_id="modeling_module")
    def modeling_task(processed_file, **context):
        from modules.modeling import build_and_train
        ds = context['ds']
        model_file = os.path.join(DATA_DIR, f"model_{ds}.h5")
        scaler_file = os.path.join(DATA_DIR, f"scaler_{ds}.gz")
        if not build_and_train(processed_file, model_file, scaler_file, seq_length=SEQ_LENGTH): raise ValueError("Lỗi Modeling")
        return {"model_path": model_file, "scaler_path": scaler_file}

    @task(task_id="evaluation_module")
    def evaluation_task(processed_file, model_artifacts):
        from modules.evaluation import evaluate_model
        if not evaluate_model(processed_file, model_artifacts['model_path'], model_artifacts['scaler_path'], seq_length=SEQ_LENGTH): raise ValueError("Lỗi Evaluation")
        return True

    @task(task_id="inference_module")
    def inference_task(processed_file, model_artifacts):
        from modules.inference import predict_future
        results = predict_future(TICKER, model_artifacts['model_path'], model_artifacts['scaler_path'], processed_file, holidays=HOLIDAYS_LIST, days_to_predict=DAYS_PREDICT, seq_length=SEQ_LENGTH)
        if not results: raise ValueError("Lỗi Inference")
        return results

    # Pipeline
    raw_path = collect_data_task()
    proc_path = preprocess_task(raw_path)
    artifacts = modeling_task(proc_path)
    
    evaluation_task(proc_path, artifacts)
    inference_task(proc_path, artifacts)

stock_prediction_pipeline()
