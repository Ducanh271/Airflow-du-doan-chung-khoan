import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

def build_and_train(data_path: str, model_path: str, scaler_path: str, seq_length: int = 60, epochs: int = 50) -> bool:
    logger.info("--- Module 2.3: Huấn luyện LSTM (Đã xử lý Data Leakage) ---")
    try:
        df = pd.read_csv(data_path)
        # Lấy an toàn cột Close và reshape chuẩn
        dataset = df['Close'].values.reshape(-1, 1)
        
        # Validate số lượng dòng
        if len(dataset) < seq_length * 3:
            raise ValueError(f"Dữ liệu quá ngắn ({len(dataset)} dòng). Không đủ để train và test với seq_length={seq_length}.")
        
        # Split Train/Test TRƯỚC (80/20)
        train_size = int(len(dataset) * 0.8)
        train_data = dataset[:train_size]
        
        # FIT SCALER CHỈ TRÊN TẬP TRAIN
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        
        # LƯU SCALER ĐỂ ĐỒNG BỘ
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        
        X_train, y_train = [], []
        for i in range(seq_length, len(scaled_train)):
            X_train.append(scaled_train[i-seq_length:i, 0])
            y_train.append(scaled_train[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"✅ Huấn luyện xong. Epochs: {len(history.history['loss'])}")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi huấn luyện: {e}")
        return False
