import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os

# Load the LightGBM model from file
loaded_model_lgb = joblib.load('lightgbm_model.pkl')

# Dữ liệu trước đó
previous_data = None

# Các đặc trưng được chọn cho LightGBM
selected_features_lgb = [
    'CLRSKY_SFC_SW_DWN', 'TOA_SW_DWN', 'TS', 'T2M_RANGE', 'T2M_MAX',
    'ALLSKY_SFC_LW_DWN', 'RH2M', 'PRECTOTCORR', 'PS', 'WD2M', 'WD10M', 'GWETTOP'
]

def load_realtime_data():
    try:
        df_realtime = pd.read_csv('data_2024_final.csv')
        return df_realtime
    except FileNotFoundError:
        result_label.config(text="Không tìm thấy tệp CSV. Vui lòng kiểm tra lại đường dẫn và tên tệp.")
        return None

def predict_realtime():
    global previous_data

    # Đọc dữ liệu từ file CSV real-time
    df_realtime = load_realtime_data()

    if df_realtime is None or len(df_realtime) == 0:
        result_label.config(text="Không có dữ liệu để dự đoán.")
        return

    # Lấy dữ liệu cuối cùng
    latest_data = df_realtime.iloc[-1]

    # Kiểm tra sự thay đổi trong dữ liệu
    if previous_data is None or not previous_data.equals(latest_data):
        # Chuẩn bị dữ liệu mới cần dự đoán
        new_data_lgb = np.array([latest_data[selected_features_lgb].values], dtype=float)

        # Dự đoán
        try:
            new_data_predictions_lgb = loaded_model_lgb.predict(new_data_lgb)
            predicted_temperature_ts = new_data_predictions_lgb[0]  # Nhiệt độ TS dự đoán

            # Cộng thêm 1 ngày vào ngày dự đoán
            prediction_date = datetime(int(latest_data['YEAR']), int(latest_data['MONTH']), int(latest_data['DAY'])) + timedelta(days=1)

            # Dự đoán lượng mưa (giả định)
            predicted_precipitation = np.random.uniform(0, 20)  # Giá trị dự đoán lượng mưa giả định

            # Hiển thị thông tin dự đoán
            prediction_info = f"Dự báo nhiệt độ và lượng mưa tại Đại học Bách khoa Hà Nội ngày: {prediction_date.strftime('%Y-%m-%d')} \nNhiệt độ TS: {predicted_temperature_ts:.2f} °C \nLượng mưa: {predicted_precipitation:.2f} mm"
            temperature_label.config(text=f"{predicted_temperature_ts:.2f} °C")
            precipitation_label.config(text=f"{predicted_precipitation:.2f} mm")
            date_label.config(text=f"Dự báo nhiệt độ và lượng mưa\n{prediction_date.strftime('%d-%m-%Y')} tại Đại học Bách khoa Hà Nội")

        except Exception as e:
            result_label.config(text=f"Lỗi khi dự đoán: {str(e)}")

        finally:
            # Cập nhật dữ liệu trước đó
            previous_data = latest_data

    # Lặp lại hàm này sau một khoảng thời gian (ví dụ: 5000 ms)
    root.after(5000, predict_realtime)

def refresh_data():
    global previous_data

    # Gọi lại hàm để cập nhật dữ liệu mới từ file CSV
    predict_realtime()

    # Tạo Label thông báo cập nhật thành công màu xanh lá
    success_label = ttk.Label(main_frame, text="Dữ liệu đã cập nhật thành công", foreground="green", background="white")
    success_label.grid(row=4, column=0, columnspan=2, pady=10)

    # Tự động đóng Label sau 3 giây
    root.after(3000, lambda: success_label.destroy())

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Ứng dụng Dự Báo Thời tiết Nhóm 2")

# Đặt icon cho cửa sổ
icon_path = 'fami.png'
if os.path.exists(icon_path):
    try:
        root.iconphoto(False, PhotoImage(file=icon_path))
    except tk.TclError as e:
        print(f"Không thể đặt biểu tượng: {e}")
else:
    print(f"Icon file '{icon_path}' not found.")

# Sử dụng chủ đề "clam" của ttk để có giao diện hiện đại hơn
style = ttk.Style(root)
style.theme_use("clam")

# Đặt nền trắng cho toàn bộ cửa sổ
root.configure(background="white")

# Frame chứa các phần tử
main_frame = tk.Frame(root, background="white", padx=10, pady=10)
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Header với màu trắng và chữ màu đen
header_frame = tk.Frame(main_frame, background="white")
header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

header_label = tk.Label(header_frame, text="Dự Báo Nhiệt Độ và Lượng Mưa", font=("Arial", 16, "bold"), foreground="black", background="white")
header_label.grid(row=0, column=0, pady=10, padx=10)

# Thêm hình ảnh mặt trời với kích thước nhỏ hơn
sun_image_path = 'sun.png'
if os.path.exists(sun_image_path):
    try:
        sun_image = PhotoImage(file=sun_image_path).subsample(3, 3)  # Giảm kích thước ảnh
        sun_label = tk.Label(main_frame, image=sun_image, background="white")
        sun_label.grid(row=1, column=0, pady=10, padx=10, sticky=tk.W)
    except tk.TclError as e:
        print(f"Không thể nạp hình ảnh mặt trời: {e}")
else:
    print(f"Sun image file '{sun_image_path}' not found.")

# Thêm hình ảnh mưa với kích thước nhỏ hơn
rain_image_path = 'rain.png'
if os.path.exists(rain_image_path):
    try:
        rain_image = PhotoImage(file=rain_image_path).subsample(26, 26)  # Giảm kích thước ảnh
        rain_label = tk.Label(main_frame, image=rain_image, background="white")
        rain_label.grid(row=2, column=0, pady=10, padx=10, sticky=tk.W)
    except tk.TclError as e:
        print(f"Không thể nạp hình ảnh mưa: {e}")
else:
    print(f"Rain image file '{rain_image_path}' not found.")

# Label hiển thị nhiệt độ
temperature_label = tk.Label(main_frame, text="-- °C", font=("Arial", 24), background="white")
temperature_label.grid(row=1, column=1, pady=10, padx=10, sticky=tk.W)

# Label hiển thị lượng mưa
precipitation_label = tk.Label(main_frame, text="-- mm", font=("Arial", 24), background="white")
precipitation_label.grid(row=2, column=1, pady=10, padx=10, sticky=tk.W)

# Label hiển thị thông tin dự đoán
date_label = tk.Label(main_frame, text="", font=("Arial", 14), background="white", justify=tk.LEFT)
date_label.grid(row=3, column=0, columnspan=2, pady=10)

# Label để hiển thị thông báo lỗi
result_label = tk.Label(main_frame, text="", background="white")
result_label.grid(row=4, column=0, columnspan=2, pady=10)

# Nút "Refresh" để cập nhật dữ liệu mới
refresh_button = ttk.Button(main_frame, text="Refresh", command=refresh_data)
refresh_button.grid(row=5, column=0, columnspan=2, pady=10)

# Gọi hàm predict_realtime() lần đầu và sau đó sẽ tự động lặp lại sau mỗi khoảng thời gian
predict_realtime()

# Chạy vòng lặp sự kiện Tkinter
root.mainloop()
