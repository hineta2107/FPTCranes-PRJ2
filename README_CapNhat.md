# BÁO CÁO NHỮNG THAY ĐỔI ĐÃ THỰC HIỆN TRÊN DỰ ÁN (Cập nhật liên tục)

Tài liệu này ghi chú lại tất cả các thay đổi về mã nguồn và giao diện so với dự án ban đầu nhằm đáp ứng yêu cầu của file báo cáo DOCX (`Document_QD Project KHDL&AI_duy.docx`).

## 🌟 TÓM TẮT CÁC THAY ĐỔI LỚN
- **Lược bỏ mô hình SVR**: Xóa hoàn toàn thuật toán Support Vector Regressor ra khỏi pipeline huấn luyện.
- **Mở rộng Tuning Random Forest**: Chạy đánh giá chi tiết 4 cấu hình siêu tham số, đạt kết quả MAE chốt là $12,540.
- **Tích hợp MLOps thu nhỏ (Dataset Upload & Retrain)**: Bổ sung tính năng cho phép Admin tải lên dataset CSV mới từ thanh công cụ (sidebar), kiểm tra schema (25 cột) và kích hoạt chạy lại toàn bộ pipeline 12 bước trực tiếp từ giao diện web, tự động reload khi hoàn thành.
- **Tái cấu trúc Giao diện Streamlit**: Tích hợp phân quyền (Admin/User), gỡ bỏ phần hiển thị của Stage 11 (Deployment Gate) trong mục Best Model để giao diện tập trung hơn.
- **Cập nhật Báo cáo & Tài liệu (Docs)**: Tạo tự động các file nội dung chuẩn hóa (cả Tiếng Anh và Tiếng Việt), bao gồm cả script Python để chèn nội dung an toàn vào file Word báo cáo cuối kỳ.

---


## 1. Tinh chỉnh Hyperparameters của mô hình Random Forest (Stage 10)
- **Tình trạng ban đầu:** Mã nguồn chỉ định nghĩa cấu hình chạy thử với 2 bộ tham số (2 candidates).
- **Sự thay đổi:** Cập nhật thành 4 cấu hình chạy thử chính xác theo yêu cầu DOCX (`n_estimators`, `min_samples_leaf`, `max_features`, `max_depth`).
- **File đã chỉnh sửa:** 
  - `src/training/tuning.py`
  - `migrate_projects/FPTCranes-PRJ2_Clean/src/training/tuning.py`
- **Kết quả:** Đã chạy lại toàn bộ luồng huấn luyện (`python pipeline.py`), các file CSV/PNG trong `outputs/` đều đã được đồng bộ với 4 cấu hình và cho ra CV MAE tốt nhất là **$12,540**.

## 2. Tạo bản nháp nội dung Báo cáo (Docs)
- **Tình trạng ban đầu:** Chưa có cấu trúc sẵn sàng cho tài liệu Word.
- **Sự thay đổi:** Trích xuất kết quả mới nhất từ hệ thống để tạo nội dung copy-paste trực tiếp.
- **File mới tạo:**
  - `docs/stage_09_12_report_draft.md`: Dàn ý tổng hợp toàn bộ từ Stage 9 đến Stage 12, bao gồm số liệu mới nhất.
  - `docs/random_forest_tuning_draft.md`: Bảng kết quả tinh chỉnh siêu tham số và phân tích chi tiết của 4 mô hình, có thể dán thẳng vào bản báo cáo.

## 3. Điều chỉnh giao diện Streamlit (Stage 12)
- **Tình trạng ban đầu:** Bảng thông tin "Final locked-test serving summary" nằm ở phần **5. Salary Prediction**. Tuy nhiên bảng này mang tính chất đánh giá mô hình nhiều hơn là dự đoán mức lương.
- **Sự thay đổi:** 
  - Đã loại bỏ bảng summary khỏi tab Salary Prediction.
  - Chuyển bảng hiển thị summary này lên giao diện của trang **4. Best model** (tab: *10 · Best Model & Importance*) để phù hợp với logic trình bày cấu hình chốt.
  - Xóa bỏ 2 hình ảnh biểu đồ thừa (Actual vs Predicted & Residuals) ở trang "5. Salary prediction".
  - Di chuyển bảng "Locked-test prediction examples" từ trang số 5 sang phần Best model để gom nhóm toàn bộ dữ liệu đánh giá mô hình.
  - Xóa bỏ hoàn toàn tab "Stage 11" khỏi trang Best Model, biến phần đánh giá của "Stage 10" thành nội dung chính yếu và duy nhất trên trang này.
  - Thay đổi luồng nghiệp vụ trên trang "5. Salary prediction": Áp dụng cơ chế **Hàng đợi (Queue)** kết hợp **Deployment Gate**. Cụ thể: Nút "Predict" cũ được đổi thành "Add to Validation Queue", dữ liệu nhập vào sẽ trải qua việc kiểm duyệt logic (ví dụ: Thành phố và Quốc gia), nếu hợp lệ sẽ được đưa vào bảng hàng đợi (Queue Table). Sau đó người dùng bấm "Predict Annual Salaries" để tiến hành dự đoán hàng loạt cho tất cả các bản ghi có trong Queue.
  - Bổ sung nút **"💾 Download CSV"** ở phần hàng đợi: Người dùng có thể chiết xuất (export) toàn bộ các kịch bản đầu vào đã nhập trong bảng thành file `.csv` lưu về máy tính. Đặc biệt, dữ liệu khi xuất ra đã được **định dạng tự động khớp hoàn toàn với 25 cột của bộ dữ liệu gốc**. Các trường chưa có trong form sẽ được xử lý rất thông minh:
    - `job_id`: Hệ thống ngầm đọc file data gốc, tìm ID lớn nhất và tự động tăng dần (Auto-increment) nối tiếp cho các bản ghi mới.
    - `posting_year`, `posting_month`: Tự động điền theo thời gian thực (hiện tại) của hệ thống.
    - Các trường còn lại (như `salary_min_usd`, `is_senior`...): Tự động để trống.
  - Tính năng này hỗ trợ đắc lực cho việc thu thập dữ liệu nhãn (data collection) phục vụ quá trình Retrain sau này.
  - Tích hợp **Tính năng phân quyền (Role-based UI):** 
    - Tài khoản `admin` (mật khẩu: `AIJob2026!`) sẽ nhìn thấy đầy đủ toàn bộ giao diện như cũ, bao gồm cả quyền giả lập các chỉ số thị trường (Demand score, Benefits score).
    - Tài khoản `user` (mật khẩu: `user123`) dành riêng cho ứng viên phổ thông. Các trường đánh giá thị trường vô lý (với ứng viên) bị ẩn đi hoàn toàn, hệ thống tự động gán giá trị mặc định ngầm (median) để đảm bảo mô hình vẫn chạy mượt mà.
- **File đã chỉnh sửa:**
  - `src/components/_login.py` (Cấu trúc lại luồng đăng nhập, tách bạch tài khoản admin và user, lưu biến Role vào Session)
  - `src/pages/_prediction.py` (Cấu trúc lại toàn bộ luồng dự đoán bằng `st.session_state` để tạo tính năng Input Validation & Deployment Gate Queue, bổ sung logic ẩn/hiện trường nhập liệu theo Role)
  - `src/pages/_best_model.py` (Gỡ bỏ Component `st.tabs`, loại bỏ hoàn toàn mã nguồn render của Stage 11, đưa nội dung của Stage 10 ra cấp ngoài cùng)
  - `README.md` (Bổ sung thông tin về 2 loại tài khoản đăng nhập)

## 4. Xóa bỏ mô hình SVR khỏi Hệ thống
- **Tình trạng ban đầu:** Mô hình Support Vector Regressor (SVR) nằm trong danh sách các thuật toán chạy đánh giá.
- **Sự thay đổi:** 
  - Gỡ bỏ SVR ra khỏi danh mục mô hình (`src/training/model_catalog.py`).
  - Gỡ bỏ Unit test yêu cầu sự tồn tại của SVR (`tests/test_stage8_10_requirements.py`).
  - Cập nhật lại script giải thích tính huống mô hình (`src/pipeline/best_model_selection_feature_importance_review.py`).
  - Đã chạy lại toàn bộ Pipeline (`python pipeline.py`) để đồng bộ biểu đồ và kết quả.

## 5. Cập nhật mới nhất: Đơn giản hóa giao diện Streamlit và Bổ sung Tài liệu tiếng Anh
- **Tình trạng:** 
  - Yêu cầu bổ sung tính năng MLOps thu nhỏ: Cho phép người dùng upload bộ dữ liệu mới (định dạng CSV) và huấn luyện lại pipeline (Retrain) ngay trên thanh công cụ của Streamlit.
  - Yêu cầu gỡ bỏ phần hiển thị của Cổng kiểm duyệt (Deployment Gate - Stage 11) trong tab Best Model để báo cáo gọn gàng hơn, tuy nhiên vẫn giữ nguyên cơ chế Hàng đợi (Queue) ở phần dự đoán.
- **Sự thay đổi về cấu trúc code:**
  - `src/components/_sidebar_upload.py`: File mới chịu trách nhiệm vẽ giao diện Upload Dataset trên thanh Sidebar (chỉ hiện cho role `admin`). Nó đọc DataFrame, đếm số dòng/cột, đánh giá Schema PASS/FAIL, cho xem trước dữ liệu, và có nút để lưu đè file gốc rồi chạy ngầm tiến trình huấn luyện (`python pipeline.py`) bằng thư viện `subprocess`. Đồng thời có chức năng khôi phục (restore) lại bộ dữ liệu chuẩn ban đầu.
  - `streamlit.py`: Import và render component `_sidebar_upload.py` vào thanh công cụ.
  - `src/pages/_best_model.py`: Xóa bỏ hoàn toàn phần giao diện `🚀 Deployment Gate (Stage 11)` và các cảnh báo (evidence) liên quan đến việc kiểm duyệt artifact khỏi trang Best Model.
- **Sự thay đổi về tài liệu (Docs):**
  - Giữ nguyên tài liệu `docs/stage_09_12_and_section_E.md` phản ánh đúng tính năng Queue và CSV download ở Stage 12.
  - Tạo mới file `docs/section_B_system_architecture.md`: Khởi tạo chi tiết phần cấu trúc hệ thống (B.1 Hardware, B.2 Software, B.3 Tech Stack).

## Tổng kết hướng tiếp theo
- Dữ liệu chạy hiện đã 100% khớp với biểu đồ và bảng mẫu của DOCX.
- Kiến trúc Code cũng như giao diện Streamlit đã linh hoạt và bám sát thực tế nhất.
- Pipeline hoàn toàn có thể tái lập (`reproducible`) chỉ với 1 câu lệnh `python pipeline.py`.
