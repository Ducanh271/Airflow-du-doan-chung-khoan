import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Giám Sát Dự Báo Cổ Phiếu", layout="wide")
st.title("📊 Bảng điều khiển Giám sát Dự báo Chứng khoán (AutoML)")
st.markdown("Hệ thống tự động cập nhật và dự báo giá cổ phiếu hàng ngày thông qua Apache Airflow.")

# 2. ĐỌC DỮ LIỆU TỪ FILE JSON CỦA AIRFLOW
# Đảm bảo đường dẫn này trỏ đúng đến file JSON mà hệ thống của cậu sinh ra hôm nay
json_path = "/home/duckanh/airflow/data/stock/model_2026-04-08_predictions.json"

try:
    with open(json_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error(f"Không tìm thấy file: {json_path}")
    st.stop()

# 3. HIỂN THỊ KPI METRICS (Chỉ số đánh giá từ eval.txt)
st.header(f"Mã Chứng Khoán: {data['ticker']}")
col1, col2, col3 = st.columns(3)
col1.metric(label="MAPE (Sai số %)", value="3.96%", delta="- Tốt", delta_color="inverse")
col2.metric(label="MAE (Sai số tuyệt đối)", value="746.43 VND")
col3.metric(label="R2 Score", value="0.9325")

# 4. CHUẨN BỊ DỮ LIỆU BIỂU ĐỒ
last_date = data["last_actual_date"]
last_close = data["last_actual_close"]
predictions = data["predictions"]
pred_dates = list(predictions.keys())
pred_values = list(predictions.values())

# Nối điểm thực tế với điểm dự báo để nét vẽ không bị đứt
x_pred = [last_date] + pred_dates
y_pred = [last_close] + pred_values

# 5. VẼ BIỂU ĐỒ TRỰC QUAN BẰNG PLOTLY
fig = go.Figure()

# Điểm thực tế cuối cùng
fig.add_trace(go.Scatter(
    x=[last_date], y=[last_close], 
    mode='markers+lines', name='Giá Thực Tế', marker=dict(size=12, color='blue')
))

# Đường dự báo tương lai
fig.add_trace(go.Scatter(
    x=x_pred, y=y_pred, 
    mode='lines+markers', name='Giá Dự Báo Tương Lai', line=dict(dash='dash', color='orange')
))

fig.update_layout(xaxis_title="Thời gian", yaxis_title="Giá đóng cửa (VND)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# 6. HIỂN THỊ BẢNG SỐ LIỆU CHI TIẾT BÊN DƯỚI
st.subheader("Chi tiết giá dự báo 5 ngày tới")
df_pred = pd.DataFrame({"Ngày": pred_dates, "Giá Dự Báo (VND)": pred_values})
st.table(df_pred.set_index("Ngày"))
