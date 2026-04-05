import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def collect_data(ticker: str, start_date: str, end_date: str, save_path: str) -> bool:
    print(f"--- Module 2.1: Đang thu thập dữ liệu {ticker} từ {start_date} đến {end_date} ---")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError("Không có dữ liệu trả về từ yfinance")
        
        # === FIX LỖI TIÊU ĐỀ MULTI-INDEX CỦA YFINANCE ===
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # ================================================
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"✅ Thu thập thành công: {len(data)} rows → {save_path}")
        return True
    except Exception as e:
        print(f"❌ Lỗi thu thập dữ liệu: {e}")
        return False
