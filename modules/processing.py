import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_and_sync(input_path, output_path, scaler_path):
    print("--- Module 2.2: Đang xử lý và đồng bộ dữ liệu ---")
    df = pd.read_csv(input_path)
    # Lấy giá đóng cửa (Close)
    data = df[['Close']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Lưu scaler để dùng cho bước dự báo sau này (đồng bộ hóa)
    joblib.dump(scaler, scaler_path)
    
    pd.DataFrame(scaled_data).to_csv(output_path, index=False)
    return True
