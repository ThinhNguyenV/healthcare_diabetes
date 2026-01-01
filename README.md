# Dự Án Phân Tích Y Tế: Dự Đoán Nguy Cơ Tái Nhập Viện

Dự án này là một bài tập thực hành Phân tích Dữ liệu (Data Analyst), tập trung vào việc dự đoán nguy cơ tái nhập viện trong vòng 30 ngày của bệnh nhân tiểu đường.

## Mục tiêu Dự án
1.  Làm sạch và khám phá bộ dữ liệu y tế phức tạp (UCI Diabetes 130-US Hospitals).
2.  Xây dựng và so sánh hiệu suất của nhiều mô hình Machine Learning (Logistic Regression, Random Forest, XGBoost, LightGBM) sau khi áp dụng kỹ thuật cân bằng dữ liệu SMOTE.
3.  Xác định các yếu tố nguy cơ quan trọng nhất (Feature Importance) và phân tích ý nghĩa lâm sàng.
4.  Đề xuất các chiến lược can thiệp y tế dựa trên phân tích Chi phí - Lợi ích (Cost-Benefit Analysis).

## Cấu trúc Thư mục

```text
healthcare_diabetes_project/
├── data/
│   ├── raw/                # Dữ liệu thô (diabetes_data_raw.csv)
│   └── processed/          # Dữ liệu đã làm sạch và kết quả phân tích (diabetes_cleaned.csv, cba_results.csv)
├── models/                 # Các mô hình đã huấn luyện (.pkl)
├── reports/
│   ├── Bao_cao_Phan_tich.md# Báo cáo phân tích chi tiết
│   └── figures/            # Các biểu đồ trực quan hóa (.png)
├── src/                    # Mã nguồn Python
│   ├── data_prep/          # Scripts tải và làm sạch dữ liệu
│   ├── models/             # Scripts huấn luyện và so sánh mô hình
│   └── visualization/      # Scripts tạo biểu đồ
└── README.md               # File này
```

## Hướng dẫn Thực thi

Để chạy lại dự án, bạn cần thực hiện các bước sau (trong môi trường ảo đã được thiết lập):

1.  **Tải dữ liệu:**
    ```bash
    python src/data_prep/download_data.py
    ```
2.  **Làm sạch và EDA:**
    ```bash
    python src/data_prep/eda_and_cleaning.py
    ```
3.  **Huấn luyện và So sánh Mô hình (Bao gồm SMOTE):**
    ```bash
    python src/models/compare_models.py
    ```
4.  **Phân tích Feature Importance và CBA:**
    ```bash
    python src/models/analyze_lgbm_importance.py
    python src/models/cba_analysis.py
    ```
5.  **Tạo các biểu đồ trực quan hóa:**
    ```bash
    python src/visualization/visualizations.py
    python src/visualization/visualize_comparison.py
    python src/visualization/visualize_top_features.py
    ```

## Kết quả Chính
*   **Mô hình tốt nhất:** LightGBM (ROC AUC 0.5451)
*   **Yếu tố nguy cơ hàng đầu:** Thay đổi đơn thuốc (`change`) và việc sử dụng thuốc tiểu đường (`diabetesMed`).
*   **Chiến lược kinh tế nhất:** "Gói Chăm sóc Chuyển tiếp" (ROI 69.5%).
