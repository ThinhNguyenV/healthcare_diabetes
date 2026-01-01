# Báo Cáo Dự Án Phân Tích Dữ Liệu Y Tế

## Dự Đoán Nguy Cơ Tái Nhập Viện Của Bệnh Nhân Tiểu Đường

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Dự án này nhằm mục đích xây dựng một mô hình phân loại để dự đoán nguy cơ **tái nhập viện trong vòng 30 ngày** của bệnh nhân tiểu đường. Bản cập nhật này tập trung vào việc **cải thiện hiệu suất mô hình** bằng cách áp dụng kỹ thuật **SMOTE** để xử lý mất cân bằng dữ liệu và **so sánh bốn thuật toán Machine Learning** khác nhau.

**Kết quả chính (Cập nhật):**
*   **Xử lý Mất Cân Bằng:** Việc áp dụng SMOTE đã giúp các mô hình có khả năng nhận diện các trường hợp tái nhập viện (Recall) tốt hơn đáng kể so với mô hình ban đầu (Recall tăng từ 0.00 lên khoảng 0.30 - 0.43).
*   **Mô hình Tốt nhất:** Mô hình **LightGBM** cho kết quả ROC AUC tốt nhất (0.5451), trong khi **Logistic Regression** cho Recall cao nhất (0.4333).
*   **Kết luận:** Mặc dù hiệu suất đã được cải thiện, khả năng dự đoán vẫn còn thấp (ROC AUC ~0.54), cho thấy sự phức tạp của bài toán và cần Feature Engineering chuyên sâu hơn (ví dụ: mã hóa ICD-9).

---

## 2. Giới Thiệu

(Nội dung giữ nguyên: Tầm quan trọng của việc dự đoán tái nhập viện sớm.)

---

## 3. Dữ Liệu và Phương Pháp Luận

### 3.1. Nguồn Dữ Liệu

(Nội dung giữ nguyên: Dữ liệu từ UCI Machine Learning Repository [1].)

### 3.2. Làm Sạch và Tiền Xử Lý Dữ Liệu

(Nội dung giữ nguyên: Xử lý giá trị thiếu, tạo biến mục tiêu nhị phân.)

### 3.3. Phân Tích Khám Phá Dữ Liệu (EDA)

(Nội dung giữ nguyên: Phân phối biến mục tiêu, tương quan, Age và HbA1c. Các biểu đồ: `eda_readmission_dist.png`, `viz_correlation_matrix.png`, `eda_age_readmission.png`, `viz_a1c_readmission.png`.)

---

## 4. Xây Dựng và So Sánh Mô Hình Dự Báo (Cập Nhật)

### 4.1. Xử Lý Mất Cân Bằng Dữ Liệu

Do sự mất cân bằng nghiêm trọng của biến mục tiêu (chỉ 11.3% trường hợp tái nhập viện), kỹ thuật **SMOTE (Synthetic Minority Over-sampling Technique)** đã được áp dụng trên tập huấn luyện để tạo ra các mẫu tổng hợp cho lớp thiểu số, giúp cân bằng lại dữ liệu.

### 4.2. So Sánh Hiệu Suất Mô Hình

Bốn mô hình phân loại đã được huấn luyện trên tập dữ liệu đã được cân bằng bằng SMOTE: **Logistic Regression**, **Random Forest**, **XGBoost**, và **LightGBM**.

**Bảng 1: So sánh Hiệu suất Mô hình trên Tập Kiểm tra**

| Mô hình | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.1288 | **0.4333** | 0.1986 | 0.5419 |
| Random Forest | 0.1309 | 0.3122 | 0.1845 | 0.5395 |
| XGBoost | 0.1340 | 0.3131 | 0.1877 | 0.5439 |
| **LightGBM** | **0.1337** | 0.3041 | **0.1858** | **0.5451** |

*Lưu ý: Precision thấp do mô hình dự đoán nhiều trường hợp tái nhập viện hơn sau khi áp dụng SMOTE, nhưng tỷ lệ dự đoán đúng trong số đó vẫn thấp.*

**Phân tích Kết quả:**
*   **Recall (Độ nhạy):** Mô hình **Logistic Regression** cho Recall cao nhất (0.4333), nghĩa là nó nhận diện được 43.33% các trường hợp tái nhập viện thực tế. Đây là một cải thiện lớn so với Recall 0.00 của mô hình Random Forest ban đầu.
*   **ROC AUC:** Mô hình **LightGBM** có khả năng phân loại tốt nhất với ROC AUC là 0.5451. Tuy nhiên, giá trị này vẫn gần với 0.5 (ngẫu nhiên), cho thấy mô hình vẫn chưa thực sự mạnh mẽ.

#### Biểu đồ So sánh ROC Curve

Biểu đồ dưới đây minh họa khả năng phân loại của các mô hình, cho thấy tất cả các mô hình đều có đường cong gần sát với đường chéo (ngẫu nhiên).

![So sánh ROC Curve giữa các mô hình](viz_model_comparison_roc.png)

---

## 5. Kết Luận và Khuyến Nghị

### 5.1. Kết Luận

Việc áp dụng SMOTE và so sánh nhiều mô hình đã giúp dự án đạt được mục tiêu nâng cao khả năng nhận diện các trường hợp tái nhập viện sớm. Mô hình **LightGBM** là mô hình tốt nhất dựa trên chỉ số ROC AUC. Tuy nhiên, hiệu suất tổng thể vẫn còn hạn chế.

### 5.2. Khuyến Nghị

Để đạt được hiệu suất dự đoán ở cấp độ chuyên nghiệp hơn, cần tập trung vào:

1.  **Feature Engineering Chuyên sâu:** Đây là bước quan trọng nhất. Cần mã hóa các biến chẩn đoán ICD-9 (`diag_1`, `diag_2`, `diag_3`) thành các nhóm bệnh lý có ý nghĩa lâm sàng.
2.  **Tối ưu hóa Siêu Tham số:** Sử dụng các kỹ thuật như Grid Search hoặc Bayesian Optimization để tinh chỉnh các tham số của LightGBM và XGBoost.
3.  **Sử dụng Metric Phù hợp:** Trong y tế, **Recall** (để không bỏ sót bệnh nhân có nguy cơ) và **F1-Score** (cân bằng giữa Precision và Recall) thường quan trọng hơn Accuracy.

---

## Tài liệu tham khảo

[1] Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 \[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
