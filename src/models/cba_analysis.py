import pandas as pd
import numpy as np

def perform_cba():
    # Giả định các thông số kinh tế (Dựa trên dữ liệu y tế Mỹ)
    COST_PER_READMISSION = 12000  # Chi phí trung bình cho 1 ca tái nhập viện ($)
    TOTAL_PATIENTS = 10000        # Giả định quy mô 10,000 bệnh nhân/năm
    READMISSION_RATE = 0.113      # Tỷ lệ tái nhập viện thực tế (từ dữ liệu của chúng ta)
    
    # 1. Chiến lược 1: Gói Chăm sóc Chuyển tiếp (Dựa trên LightGBM - Tập trung nhóm đổi thuốc)
    # Giả định: Nhắm vào 20% bệnh nhân có nguy cơ cao nhất, giảm được 25% số ca tái nhập viện trong nhóm này.
    cost_s1_per_patient = 200     # Chi phí gọi điện, tư vấn dược sĩ
    coverage_s1 = 0.20            # Tỷ lệ bệnh nhân được can thiệp
    effectiveness_s1 = 0.25       # Hiệu quả giảm tái nhập viện
    
    total_cost_s1 = TOTAL_PATIENTS * coverage_s1 * cost_s1_per_patient
    prevented_s1 = (TOTAL_PATIENTS * READMISSION_RATE * coverage_s1) * effectiveness_s1
    savings_s1 = prevented_s1 * COST_PER_READMISSION
    roi_s1 = (savings_s1 - total_cost_s1) / total_cost_s1
    
    # 2. Chiến lược 2: Quản lý ca bệnh cho người cao tuổi (Dựa trên Logistic Regression - Diện rộng)
    # Giả định: Nhắm vào 40% bệnh nhân (do Recall cao nhưng Precision thấp), giảm được 15% số ca tái nhập viện.
    cost_s2_per_patient = 500     # Chi phí quản lý ca bệnh chuyên sâu
    coverage_s2 = 0.40
    effectiveness_s2 = 0.15
    
    total_cost_s2 = TOTAL_PATIENTS * coverage_s2 * cost_s2_per_patient
    prevented_s2 = (TOTAL_PATIENTS * READMISSION_RATE * coverage_s2) * effectiveness_s2
    savings_s2 = prevented_s2 * COST_PER_READMISSION
    roi_s2 = (savings_s2 - total_cost_s2) / total_cost_s2
    
    # 3. Chiến lược 3: Tối ưu hóa quy trình xuất viện (Dựa trên Quy trình & Bảng kiểm)
    # Giả định: Nhắm vào tất cả bệnh nhân, chi phí thấp nhưng hiệu quả cũng khiêm tốn hơn (5%).
    cost_s3_per_patient = 50      # Chi phí đào tạo, in ấn bảng kiểm, thêm bước kiểm tra
    coverage_s3 = 1.0
    effectiveness_s3 = 0.05
    
    total_cost_s3 = TOTAL_PATIENTS * coverage_s3 * cost_s3_per_patient
    prevented_s3 = (TOTAL_PATIENTS * READMISSION_RATE * coverage_s3) * effectiveness_s3
    savings_s3 = prevented_s3 * COST_PER_READMISSION
    roi_s3 = (savings_s3 - total_cost_s3) / total_cost_s3
    
    # Tổng hợp kết quả
    results = pd.DataFrame({
        'Chiến lược': ['Gói Chăm sóc Chuyển tiếp', 'Quản lý ca bệnh (Cao tuổi)', 'Tối ưu quy trình xuất viện'],
        'Tổng chi phí ($)': [total_cost_s1, total_cost_s2, total_cost_s3],
        'Số ca giảm được': [prevented_s1, prevented_s2, prevented_s3],
        'Tổng tiết kiệm ($)': [savings_s1, savings_s2, savings_s3],
        'Lợi nhuận ròng ($)': [savings_s1 - total_cost_s1, savings_s2 - total_cost_s2, savings_s3 - total_cost_s3],
        'ROI (%)': [roi_s1 * 100, roi_s2 * 100, roi_s3 * 100]
    })
    
    print(results)
    results.to_csv('data/processed/cba_results.csv', index=False)

if __name__ == "__main__":
    perform_cba()
