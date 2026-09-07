# Bảng Kết quả Tinh chỉnh Siêu tham số Random Forest

> **Mục đích**: Nội dung này có thể copy-paste trực tiếp vào bản báo cáo Word.  
> **Nguồn số liệu**: `outputs/04_best_model_and_feature_importance/10_best_model_tuning_results.csv`  
> (Chạy `python pipeline.py` để cập nhật số liệu mới nhất.)

---

## Bảng 1: 4 Cấu hình Tinh chỉnh Random Forest

| # | n_estimators | min_samples_leaf | max_features | max_depth | CV MAE (mean) | CV MAE (std) |
|---|-------------|-----------------|-------------|-----------|--------------|-------------|
| 1 | 80 | 1 | 0.8 | None (unlimited) | — | — |
| 2 | 100 | 2 | 0.8 | None (unlimited) | — | — |
| 3 | 200 | 1 | 0.7 | 20 | — | — |
| 4 | 300 | 2 | 0.7 | 20 | — | — |

> ℹ️ Điền số liệu CV MAE từ file CSV sau khi chạy pipeline. Mô hình có CV MAE thấp nhất được chọn.

---

## Phân tích Chi tiết từng Cấu hình

### Candidate 1: n_estimators=80, min_samples_leaf=1, max_features=0.8, max_depth=None

- **Đặc điểm**: Rừng nhỏ, không giới hạn độ sâu, mỗi cây dùng 80% features.
- **min_samples_leaf=1**: Cho phép leaf node chứa 1 mẫu → dễ overfit hơn.
- **Ưu điểm**: Huấn luyện nhanh.
- **Nhược điểm**: Cây có thể phát triển quá sâu, tốn bộ nhớ.

### Candidate 2: n_estimators=100, min_samples_leaf=2, max_features=0.8, max_depth=None

- **Đặc điểm**: Cấu hình mặc định điển hình, tăng regularization nhẹ với `min_samples_leaf=2`.
- **min_samples_leaf=2**: Bắt buộc mỗi lá có ít nhất 2 mẫu → giảm overfit.
- **Đây là cấu hình baseline hợp lý** cho bộ dữ liệu ~1,200 dòng.

### Candidate 3: n_estimators=200, min_samples_leaf=1, max_features=0.7, max_depth=20

- **Đặc điểm**: Rừng lớn hơn, giới hạn độ sâu tối đa 20, giảm tỉ lệ features mỗi split.
- **max_depth=20**: Kiểm soát độ phức tạp mô hình.
- **max_features=0.7**: Tăng đa dạng giữa các cây.
- **Ưu điểm**: Cân bằng tốt giữa bias và variance.

### Candidate 4: n_estimators=300, min_samples_leaf=2, max_features=0.7, max_depth=20

- **Đặc điểm**: Cấu hình regularized nhất trong 4 lựa chọn.
- **Kết hợp**: rừng lớn + max_depth giới hạn + min_samples_leaf cao + max_features thấp.
- **Ưu điểm**: Khả năng tổng quát hoá tốt nhất.
- **Nhược điểm**: Thời gian huấn luyện lâu hơn.

---

## Bảng 2: Kết quả Locked-Test (tháng 3/2026) của Mô hình Tốt nhất

| Chỉ số | Giá trị | Ý nghĩa |
|--------|---------|---------|
| MAE | ~15,054 USD | Sai số tuyệt đối trung bình |
| RMSE | ~29,993 USD | Căn bậc hai của sai số bình phương trung bình |
| R² | ~0.801 | Hệ số xác định (~80.1% variance được giải thích) |
| MedAE | ~4,440 USD | Sai số tuyệt đối trung vị (robust với outliers) |

> **Lưu ý diễn giải**: R² cao phản ánh "fit tốt trên bộ dữ liệu này", không phải "AI salary được xác định bởi..." — không được đưa ra tuyên bố nhân quả.

---

## Bảng 3: Top Feature Importance (Encoded)

> Lấy từ `outputs/04_best_model_and_feature_importance/10_encoded_feature_importance.csv`

| Rank | Encoded Feature | Importance | Nhóm |
|------|----------------|-----------|------|
| 1 | job_category_* | — | Categorical |
| 2 | years_of_experience | — | Numeric |
| 3 | skills_* | — | Text/Multi-hot |
| ... | ... | ... | ... |

> ⚠️ **Cảnh báo quan trọng**: Top-2 feature chiếm >90% tổng importance → mô hình thực chất là "2-feature model". Feature importance phản ánh mức độ phụ thuộc của mô hình, không phải quan hệ nhân quả trong thực tế thị trường lao động.

---

## Kết luận

Sau khi so sánh 4 cấu hình bằng 5-fold temporal cross-validation:

- Cấu hình tốt nhất (CV MAE thấp nhất) được chọn tự động.
- Mô hình được huấn luyện lại trên toàn bộ DEV set với cấu hình tốt nhất.
- Locked test (298 dòng, tháng 3/2026) chỉ được mở **một lần duy nhất** để đánh giá cuối cùng.
- Kết quả được lưu tại `outputs/04_best_model_and_feature_importance/` và `artifacts/`.
