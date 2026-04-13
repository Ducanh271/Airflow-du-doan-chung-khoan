import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def predict_future(ticker: str, model_path: str, scaler_path: str, processed_data_path: str, holidays: list, days_to_predict: int = 5, seq_length: int = 60) -> dict:
    logger.info(f"--- Module Inference: Dự báo {days_to_predict} ngày tới cho {ticker} ---")
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        df = pd.read_csv(processed_data_path)
        raw_values = df['Close'].values.reshape(-1, 1)
        
        # Parse ngày an toàn bằng Pandas
        last_actual_date = pd.to_datetime(df['Date'].iloc[-1]).strftime('%Y-%m-%d')
        last_actual_close = float(raw_values[-1][0])
        
        last_60_scaled = scaler.transform(raw_values[-seq_length:])
        current_sequence = last_60_scaled.reshape(1, seq_length, 1)
        
        predictions_scaled = []
        for _ in range(days_to_predict):
            pred_scaled = model.predict(current_sequence, verbose=0)
            predictions_scaled.append(pred_scaled[0, 0])
            current_sequence = np.append(current_sequence[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
        
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions_scaled).flatten()
        
        future_dates = []
        current_date = datetime.strptime(last_actual_date, "%Y-%m-%d").date() + timedelta(days=1)
        
        i = 0
        while len(future_dates) < days_to_predict:
            date_str = current_date.strftime('%Y-%m-%d')
            # Bỏ qua cuối tuần và ngày lễ (được cấu hình từ Airflow Variable)
            if current_date.weekday() < 5 and date_str not in holidays:
                future_dates.append(date_str)
            current_date += timedelta(days=1)
            i += 1
            if i > 30: break # Vòng lặp an toàn
            
        results = {
            "ticker": ticker,
            "last_actual_date": last_actual_date,
            "last_actual_close": round(last_actual_close, 2),
            "predictions": dict(zip(future_dates, [round(float(p), 2) for p in predictions])),
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        pred_file = model_path.replace('.h5', '_predictions.json')
        import json
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info("✅ Dự báo hoàn tất!")
        return results
    except Exception as e:
        logger.error(f"❌ Lỗi dự báo: {e}")
        return {}
