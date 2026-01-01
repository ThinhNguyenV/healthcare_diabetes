import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_visualizations():
    print("Đang tạo các trực quan hóa chuyên sâu...")
    df = pd.read_csv('data/processed/diabetes_cleaned.csv')
    
    # Style
    sns.set_theme(style="whitegrid")
    
    # 1. Heatmap tương quan giữa các biến số
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 
                    'number_inpatient', 'number_diagnoses']
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Ma trận tương quan giữa các chỉ số lâm sàng')
    plt.savefig('reports/figures/viz_correlation_matrix.png')
    plt.close()
    
    # 2. Tỷ lệ tái nhập viện theo Race và Gender
    plt.figure(figsize=(12, 6))
    sns.barplot(x='race', y='readmitted_binary', hue='gender', data=df, errorbar=None)
    plt.title('Tỷ lệ tái nhập viện theo Chủng tộc và Giới tính')
    plt.ylabel('Tỷ lệ tái nhập viện (<30 ngày)')
    plt.savefig('reports/figures/viz_race_gender_readmission.png')
    plt.close()
    
    # 3. Phân tích HbA1c Result và Readmission
    # Lưu ý: A1Cresult có nhiều giá trị thiếu, chỉ lấy các giá trị có sẵn
    a1c_df = df[df['A1Cresult'].notnull()]
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='A1Cresult', y='readmitted_binary', data=a1c_df, linestyles='none')
    plt.title('Mối liên hệ giữa kết quả HbA1c và Tỷ lệ tái nhập viện')
    plt.savefig('reports/figures/viz_a1c_readmission.png')
    plt.close()
    
    # 4. Thời gian nằm viện vs Số lượng xét nghiệm (Lab Procedures)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='num_lab_procedures', y='time_in_hospital', hue='readmitted_binary', 
                    data=df.sample(2000), alpha=0.5)
    plt.title('Thời gian nằm viện vs Số lượng xét nghiệm (Mẫu 2000 bản ghi)')
    plt.savefig('reports/figures/viz_lab_vs_time.png')
    plt.close()

    print("Đã tạo xong các file hình ảnh trực quan hóa.")

if __name__ == "__main__":
    create_visualizations()
