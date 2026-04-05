import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta

def predict_future(model_path: str, scaler_path: str, processed_data_path: str, 
                  days_to_predict: int = 5, seq_length: int = 60) -> dict:
    """
    Dự báo giá đóng cửa cho N ngày tới dựa trên mô hình LSTM đã train.
    Trả về dictionary chứa kết quả để dễ lưu log và hiển thị.
    """
    print(f"--- Module Inference: Đang dự báo {days_to_predict} ngày tới cho {TICKER if 'TICKER' in globals() else 'stock'} ---")
    
    # Load model và scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Đọc dữ liệu đã scale (lấy 60 ngày gần nhất làm input)
    df_scaled = pd.read_csv(processed_data_path).values
    last_sequence = df_scaled[-seq_length:].reshape(1, seq_length, 1)
    
    predictions_scaled = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        pred_scaled = model.predict(current_sequence, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])
        
        # Cập nhật sequence cho bước dự báo tiếp theo (rolling forecast)
        current_sequence = np.append(current_sequence[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
    
    # Inverse transform về giá thực tế
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    
    # Tạo ngày dự báo (bỏ qua cuối tuần đơn giản)
    today = datetime.now().date()
    future_dates = []
    current_date = today + timedelta(days=1)
    i = 0
    while len(future_dates) < days_to_predict:
        if current_date.weekday() < 5:  # 0-4: Thứ 2 đến Thứ 6
            future_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
        i += 1
        if i > 30: break  # an toàn
    
    # Kết quả
    results = {
        "ticker": TICKER if 'TICKER' in globals() else "DIG.VN",
        "last_actual_date": pd.read_csv(processed_data_path.replace('processed', 'raw')).iloc[-1]['Date'] if os.path.exists(processed_data_path.replace('processed', 'raw')) else "N/A",
        "last_actual_close": scaler.inverse_transform(df_scaled[-1:])[0][0],
        "predictions": dict(zip(future_dates, [round(float(p), 2) for p in predictions])),
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Lưu kết quả ra file JSON để dễ theo dõi
    output_dir = os.path.dirname(processed_data_path)
    pred_file = os.path.join(output_dir, "future_predictions.json")
    import json
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ Dự báo hoàn tất!")
    print("Dự báo giá đóng cửa tương lai:")
    for date, price in results["predictions"].items():
        print(f"  {date}: {price:,.2f} VND")
    
    return results
