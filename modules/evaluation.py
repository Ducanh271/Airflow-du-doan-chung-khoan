import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def evaluate_model(data_path: str, model_path: str) -> bool:
    print("--- Module 2.4: Đánh giá mô hình ---")
    model = load_model(model_path)
    dataset = pd.read_csv(data_path).values
    
    # Test trên 20% cuối (không dùng để train)
    test_data = dataset[int(len(dataset)*0.8):]
    X_test, y_test = [], []
    seq_length = 60
    for i in range(seq_length, len(test_data)):
        X_test.append(test_data[i-seq_length:i, 0])
        y_test.append(test_data[i, 0])
    
    X_test = np.array(X_test).reshape(-1, seq_length, 1)
    predictions = model.predict(X_test, verbose=0)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"📊 Kết quả: MSE={mse:.6f} | MAE={mae:.6f} | R²={r2:.4f}")
    
    # Ghi log chi tiết
    log_path = data_path.replace('.csv', '_eval.txt')
    with open(log_path, 'w') as f:
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\n")
    
    return True
