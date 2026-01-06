import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_model():
    print("Train model")
    df = pd.read_csv('data/processed/diabetes_cleaned.csv')
    
    # 1. Feature Engineering & Selection
    # Chọn các cột quan trọng
    features = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 
                'num_procedures', 'num_medications', 'number_outpatient', 
                'number_emergency', 'number_inpatient', 'number_diagnoses', 
                'change', 'diabetesMed']
    
    X = df[features].copy()
    y = df['readmitted_binary']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Random Forest
    print("Training Random Forest model")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\n--- Kết quả đánh giá ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 5. Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Tầm quan trọng của các đặc trưng (Feature Importance)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('model_feature_importance.png')
    plt.close()
    
    # 6. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('model_roc_curve.png')
    plt.close()
    
    # 7. Lưu mô hình
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(rf, 'models/diabetes_rf_model.pkl')
    print("\nĐã lưu mô hình vào models/diabetes_rf_model.pkl")

if __name__ == "__main__":
    train_model()
