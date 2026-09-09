# 🚀 Tổng Hợp Các Thay Đổi (Cập Nhật Mới Nhất)

Dưới đây là danh sách toàn bộ các tinh chỉnh và tính năng mới đã được thực hiện để nâng cấp Pipeline, hoàn thiện cấu trúc chia dữ liệu và cải thiện giao diện Streamlit từ đầu phiên làm việc đến nay:

## 1. Cải Tiến Core Pipeline (`src/pipeline/__init__.py`)
- **Động hóa thời gian Locked Test (Dynamic Temporal Split):** Gỡ bỏ việc gán cố định (hardcode) tháng Locked Test là tháng `3/2026`. Thuật toán nay tự động quét toàn bộ dữ liệu, lấy tháng và năm mới nhất (Max Date) làm tập `LOCKED_TEST`, và giữ toàn bộ dữ liệu lịch sử trước đó làm tập `TRAIN_DEV` để chạy Expanding-Window CV.
- **Tái cấu trúc Ablation Study (Model A, B, C, D):** Viết lại kịch bản thí nghiệm Bóc tách Đặc trưng (Feature Ablation) để kiểm định xem cấu trúc dữ liệu có bị phụ thuộc hoàn toàn vào một vài biến cơ bản hay không. Bốn mô hình bao gồm:
  - **Model A (Full Features):** Tất cả các đặc trưng.
  - **Model B (Categories + Experience):** Chỉ dùng nhóm Ngành nghề/Vai trò/Địa điểm và Số năm Kinh nghiệm.
  - **Model C (Bỏ Categories):** Bỏ nhóm Categorical, giữ lại Experience và các biến phụ.
  - **Model D (Bỏ Experience):** Bỏ Số năm kinh nghiệm, giữ lại Categories và các biến phụ.
- **Bảo toàn tính độc lập của Study Cases:** Hệ thống chủ động trích xuất ngẫu nhiên 10 bản ghi làm **Study Cases** (mẫu ngoại lệ) từ tập `LOCKED_TEST` trước khi tiến hành dự đoán, đảm bảo chống Data Leakage tuyệt đối khi trình diễn thực tế.

## 2. Nâng Cấp Giao Diện: Trang Model Comparison (`src/pages/_model_comparison.py`)
- **Bảng cấu trúc Fold động:** Thay thế bảng mô phỏng Fold (với thời gian chết) bằng cơ chế tự động đọc và trích xuất chu kỳ thời gian từ dữ liệu Pipeline. Bảng giờ đây khớp chính xác với số tháng có trong dữ liệu (ví dụ: hiển thị linh hoạt 2026-05, 2026-06,...).
- **Điều chỉnh thứ tự Tabs:** Đẩy Tab **"Fold Stability & CV"** lên vị trí đầu tiên (thay thế cho Key Metrics) để chú trọng vào độ ổn định của mô hình.
- **Bổ sung số liệu trực tiếp vào đồ thị MAE Fold Stability:** Cập nhật Plotly Line Chart để hiển thị rõ các con số sai số MAE trực tiếp trên các điểm của đường đồ thị, giúp người dùng dễ dàng theo dõi mà không cần thao tác rê chuột.
- **Sửa lỗi hiển thị bị che khuất:** Điều chỉnh vị trí chữ (textposition="auto") cho các biểu đồ Bar Chart so sánh (MAE, RMSE, R²) để tránh việc thông số bị cắt xén khi thanh bar quá ngắn.
- **Biểu đồ thời gian huấn luyện (Fit Time Chart):** Bổ sung thêm một biểu đồ đường (Line Chart) mới ngay dưới bảng thông số từng Fold, trực quan hóa thời gian chạy (bằng giây) của mỗi mô hình trên từng Fold tương ứng.

## 3. Nâng Cấp Giao Diện: Trang Model Evaluation & Predictions (`src/pages/_prediction.py`)
- **Cập nhật diễn giải Ablation Study:** Viết lại toàn bộ phần giải thích kết quả cho Tab Ablation Study theo chuẩn Model A, B, C, D mới. Hệ thống sẽ tự động bắt các giá trị $R^2$ tương ứng và đưa ra nhận định logic (Nếu Model A ≈ B và C, D giảm mạnh thì chứng tỏ bộ dữ liệu có cấu trúc đơn giản, phụ thuộc hoàn toàn vào Categories và Experience).

## 4. Nâng Cấp Giao Diện: Trang Best Model (`src/pages/_best_model.py`)
- **Điều chỉnh thứ tự Tabs:** Ưu tiên đẩy tab **"Chi Tiết Mẫu Locked Test"** lên vị trí số 1 thay vì nằm ở góc cuối.
- **Trực quan hóa Study Cases:** Bổ sung giao diện thông báo làm nổi bật lượng dữ liệu **Study Cases** đã được chừa ra. Giao diện tự động tính toán phần trăm dữ liệu được hold out so với tháng Locked Test, và đính kèm một Expander cho phép người dùng click để xem trực tiếp danh sách các bản ghi này. 
- Nhấn mạnh thông điệp các bản ghi Study Cases hoàn toàn không tham gia vào Metrics hay Ablation để phản ánh 100% tính thực tế.
