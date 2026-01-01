import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def analyze_importance():
    print("Đang phân tích Feature Importance của LightGBM...")
    
    # Tải mô hình
    model_path = 'models/diabetes_lightgbm_model.pkl'
    if not os.path.exists(model_path):
        print("Không tìm thấy mô hình LightGBM. Vui lòng chạy compare_models.py trước.")
        return
    
    model = joblib.load(model_path)
    
    # Danh sách đặc trưng
    features = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 
                'num_procedures', 'num_medications', 'number_outpatient', 
                'number_emergency', 'number_inpatient', 'number_diagnoses', 
                'change', 'diabetesMed']
    
    # 1. Importance by Split (Số lần sử dụng để chia nhánh)
    importance_split = model.booster_.feature_importance(importance_type='split')
    
    # 2. Importance by Gain (Mức độ đóng góp vào việc giảm loss)
    importance_gain = model.booster_.feature_importance(importance_type='gain')
    
    # DataFrame kết quả
    importance_df = pd.DataFrame({
        'Feature': features,
        'Split': importance_split,
        'Gain': importance_gain
    })
    
    # Sắp xếp theo Gain
    importance_df = importance_df.sort_values(by='Gain', ascending=False)
    
    # Trực quan hóa Gain
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Gain', y='Feature', data=importance_df, palette='viridis')
    plt.title('LightGBM Feature Importance (Gain) - Mức độ đóng góp vào dự đoán')
    plt.xlabel('Tổng mức tăng thông tin (Total Gain)')
    plt.tight_layout()
    plt.savefig('reports/figures/lgbm_importance_gain.png')
    plt.close()
    
    # Trực quan hóa Split
    importance_df_split = importance_df.sort_values(by='Split', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Split', y='Feature', data=importance_df_split, palette='magma')
    plt.title('LightGBM Feature Importance (Split) - Tần suất sử dụng để chia nhánh')
    plt.xlabel('Số lần chia (Number of Splits)')
    plt.tight_layout()
    plt.savefig('reports/figures/lgbm_importance_split.png')
    plt.close()
    
    print("\n--- Top 5 đặc trưng quan trọng nhất (theo Gain) ---")
    print(importance_df[['Feature', 'Gain']].head(5))
    
    importance_df.to_csv('data/processed/lgbm_importance_analysis.csv', index=False)
    print("\nĐã lưu kết quả phân tích vào data/processed/lgbm_importance_analysis.csv")

if __name__ == "__main__":
    analyze_importance()
