import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def compare_models():
    print("Bắt đầu so sánh các mô hình với SMOTE...")
    df = pd.read_csv('data/processed/diabetes_cleaned.csv')
    
    # 1. Feature Selection & Encoding
    features = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 
                'num_procedures', 'num_medications', 'number_outpatient', 
                'number_emergency', 'number_inpatient', 'number_diagnoses', 
                'change', 'diabetesMed']
    
    X = df[features].copy()
    y = df['readmitted_binary']
    
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Apply SMOTE to training data
    print("Đang áp dụng SMOTE để cân bằng dữ liệu")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Kích thước tập huấn luyện sau SMOTE: {X_train_res.shape}")
    
    # 4. Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    }
    
    results = []
    
    # 5. Train and Evaluate
    for name, model in models.items():
        print(f"Đang huấn luyện {name}...")
        model.fit(X_train_res, y_train_res)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        })
        
        # Lưu mô hình tốt nhất 
        joblib.dump(model, f'models/diabetes_{name.lower().replace(" ", "_")}_model.pkl')

    # 6. Visualize Results
    results_df = pd.DataFrame(results)
    print("\n--- Kết quả so sánh mô hình ---")
    print(results_df)
    
    # Vẽ biểu đồ so sánh
    results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Score', hue='Model', data=results_melted)
    plt.title('So sánh hiệu suất các mô hình (Sau khi dùng SMOTE)')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png')
    plt.close()
    
    results_df.to_csv('data/processed/model_comparison_results.csv', index=False)
    print("\nĐã lưu kết quả so sánh vào data/processed/model_comparison_results.csv")

if __name__ == "__main__":
    compare_models()
