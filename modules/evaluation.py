import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import logging

logger = logging.getLogger(__name__)

def evaluate_model(data_path: str, model_path: str, scaler_path: str, seq_length: int = 60) -> bool:
    logger.info("--- Module 2.4: Đánh giá mô hình (Tập Test Khách Quan) ---")
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        df = pd.read_csv(data_path)
        dataset = df['Close'].values.reshape(-1, 1)
        
        train_size = int(len(dataset) * 0.8)
        
        # Tách biệt hoàn toàn Train và Test
        train_raw = dataset[:train_size]
        test_raw = dataset[train_size:]
        
        # Transform qua Scaler của Train
        train_scaled = scaler.transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        
        # Dùng seq_length ngày cuối của Train làm "Warm-up" để dự đoán ngày đầu của Test
        warmup_sequence = train_scaled[-seq_length:]
        combined_scaled = np.vstack((warmup_sequence, test_scaled))
        
        X_test, y_test_scaled = [], []
        for i in range(seq_length, len(combined_scaled)):
            X_test.append(combined_scaled[i-seq_length:i, 0])
            y_test_scaled.append(combined_scaled[i, 0])
            
        X_test = np.array(X_test).reshape(-1, seq_length, 1)
        predictions_scaled = model.predict(X_test, verbose=0)
        
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test = scaler.inverse_transform(np.array(y_test_scaled).reshape(-1, 1))
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100 # Đổi ra phần trăm
        
        logger.info(f"📊 MSE: {mse:,.2f} | MAE: {mae:,.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")
        
        log_path = model_path.replace('.h5', '_eval.txt')
        with open(log_path, 'w') as f:
            f.write(f"Date: {pd.Timestamp.now()}\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape:.2f}%\nR2: {r2}\n")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi đánh giá: {e}")
        return False
