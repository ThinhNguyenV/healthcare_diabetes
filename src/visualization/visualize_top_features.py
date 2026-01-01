import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_top_features():
    print("Đang trực quan hóa mối quan hệ giữa Top 5 Features và Readmission...")
    df = pd.read_csv('data/processed/diabetes_cleaned.csv')
    
    # Thiết lập style
    sns.set_theme(style="whitegrid")
    top_features = ['diabetesMed', 'change', 'race', 'gender', 'age']
    
    # Tạo một figure lớn với nhiều subplot
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        # Tính toán tỷ lệ tái nhập viện cho từng nhóm
        feature_readmission = df.groupby(feature)['readmitted_binary'].mean().reset_index()
        feature_readmission = feature_readmission.sort_values(by='readmitted_binary', ascending=False)
        
        sns.barplot(x=feature, y='readmitted_binary', data=feature_readmission, ax=axes[i], palette='coolwarm', hue=feature, legend=False)
        axes[i].set_title(f'Tỷ lệ tái nhập viện theo {feature}', fontsize=15)
        axes[i].set_ylabel('Tỷ lệ tái nhập viện (<30 ngày)')
        axes[i].set_xlabel(feature)
        
        # Xoay nhãn trục x nếu là 'age' hoặc 'race'
        if feature in ['age', 'race']:
            axes[i].tick_params(axis='x', rotation=45)
            
    # Xóa subplot cuối cùng không dùng đến
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('reports/figures/viz_top_5_features_relationship.png')
    plt.close()
    
    # Tạo thêm biểu đồ chi tiết cho Age vì nó có xu hướng rõ rệt
    plt.figure(figsize=(12, 6))
    age_order = sorted(df['age'].unique())
    sns.lineplot(x='age', y='readmitted_binary', data=df, marker='o', sort=True)
    plt.title('Xu hướng Tái nhập viện theo Độ tuổi', fontsize=15)
    plt.ylabel('Tỷ lệ tái nhập viện')
    plt.xticks(rotation=45)
    plt.savefig('reports/figures/viz_age_trend_line.png')
    plt.close()

    print("Đã lưu các biểu đồ: viz_top_5_features_relationship.png và viz_age_trend_line.png")

if __name__ == "__main__":
    visualize_top_features()
