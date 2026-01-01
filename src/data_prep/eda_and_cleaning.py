import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Bắt đầu EDA và làm sạch dữ liệu...")
    df = pd.read_csv('data/raw/diabetes_data_raw.csv', low_memory=False)
    
    # 1. Kiểm tra thông tin cơ bản
    print("\n--- Thông tin cơ bản ---")
    print(df.info())
    
    # 2. Xử lý giá trị thiếu
    # Trong bộ dữ liệu này, giá trị thiếu thường được ký hiệu là '?'
    df.replace('?', np.nan, inplace=True)
    missing_values = df.isnull().sum()
    print("\n--- Giá trị thiếu ---")
    print(missing_values[missing_values > 0])
    
    # Loại bỏ các cột có quá nhiều giá trị thiếu (ví dụ: weight, payer_code, medical_specialty)
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty']
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Loại bỏ các dòng có giá trị thiếu ở các cột quan trọng nhưng ít thiếu
    df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'], inplace=True)
    
    # 3. Xử lý biến mục tiêu 'readmitted'
    # Biến mục tiêu có 3 giá trị: '<30', '>30', 'NO'
    # Chuyển thành bài toán phân loại nhị phân: Tái nhập viện trong 30 ngày (1) và ngược lại (0)
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # 4. Phân tích phân phối biến mục tiêu
    plt.figure(figsize=(8, 6))
    sns.countplot(x='readmitted_binary', data=df)
    plt.title('Phân phối Tái nhập viện (1: <30 ngày, 0: Ngược lại)')
    plt.savefig('eda_readmission_dist.png')
    plt.close()
    
    # 5. Phân tích mối quan hệ giữa Age và Readmission
    plt.figure(figsize=(12, 6))
    sns.countplot(x='age', hue='readmitted_binary', data=df)
    plt.title('Tái nhập viện theo độ tuổi')
    plt.xticks(rotation=45)
    plt.savefig('eda_age_readmission.png')
    plt.close()
    
    # 6. Phân tích số lượng thuốc và thời gian nằm viện
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='readmitted_binary', y='num_medications', data=df)
    plt.title('Số lượng thuốc vs Tái nhập viện')
    plt.savefig('eda_meds_readmission.png')
    plt.close()
    
    # 7. Lưu dữ liệu đã làm sạch
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    df.to_csv('data/processed/diabetes_cleaned.csv', index=False)
    print(f"\nĐã làm sạch dữ liệu. Số lượng bản ghi còn lại: {len(df)}")
    print("Dữ liệu đã được lưu vào data/processed/diabetes_cleaned.csv")

if __name__ == "__main__":
    perform_eda()
