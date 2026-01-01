import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def visualize_comparison():
    print("Đang tạo biểu đồ so sánh ROC Curve...")
    df = pd.read_csv('data/processed/diabetes_cleaned.csv')
    
    features = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 
                'num_procedures', 'num_medications', 'number_outpatient', 
                'number_emergency', 'number_inpatient', 'number_diagnoses', 
                'change', 'diabetesMed']
    
    X = df[features].copy()
    y = df['readmitted_binary']
    
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
        
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model_files = {
        'Logistic Regression': 'models/diabetes_logistic_regression_model.pkl',
        'Random Forest': 'models/diabetes_random_forest_model.pkl',
        'XGBoost': 'models/diabetes_xgboost_model.pkl',
        'LightGBM': 'models/diabetes_lightgbm_model.pkl'
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, path in model_files.items():
        model = joblib.load(path)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('So sánh ROC Curve giữa các mô hình (Sau SMOTE)')
    plt.legend(loc="lower right")
    plt.savefig('reports/figures/viz_model_comparison_roc.png')
    plt.close()
    print("Đã lưu biểu đồ viz_model_comparison_roc.png")

if __name__ == "__main__":
    visualize_comparison()
