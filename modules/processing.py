import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_data(input_path: str, output_path: str) -> bool:
    logger.info("--- Module 2.2: Xử lý làm sạch dữ liệu (Chưa Scale) ---")
    try:
        df = pd.read_csv(input_path)
        
        # Chỉ giữ lại cột Ngày và Giá đóng cửa
        # Tuyệt đối không Scale ở đây để chống Data Leakage
        data = df[['Date', 'Close']]
        
        data.to_csv(output_path, index=False)
        logger.info(f"✅ Xử lý hoàn tất -> {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý dữ liệu: {e}")
        return False
