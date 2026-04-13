import yfinance as yf
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def collect_data(ticker: str, start_date: str, end_date: str, save_path: str) -> bool:
    logger.info(f"--- Module 2.1: Thu thập {ticker} từ {start_date} đến {end_date} ---")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"Không có dữ liệu trả về cho {ticker}")
        
        # Fix lỗi tiêu đề MultiIndex của yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Giữ lại cột Date để dùng cho dự báo sau này
        data = data.reset_index()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        logger.info(f"✅ Thu thập thành công: {len(data)} rows -> {save_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi thu thập dữ liệu: {e}")
        return False
