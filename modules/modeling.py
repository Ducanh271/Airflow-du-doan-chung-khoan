import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_and_train(data_path: str, model_path: str, seq_length: int = 60, epochs: int = 50) -> bool:
    print("--- Module 2.3: Xây dựng & huấn luyện mô hình LSTM ---")
    dataset = pd.read_csv(data_path).values
    
    # Train/test split (80/20 theo thời gian)
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    
    X_train, y_train = [], []
    for i in range(seq_length, len(train_data)):
        X_train.append(train_data[i-seq_length:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.array(y_train)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                       callbacks=[early_stop], verbose=0)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ Huấn luyện xong. Epochs thực tế: {len(history.history['loss'])}")
    return True
