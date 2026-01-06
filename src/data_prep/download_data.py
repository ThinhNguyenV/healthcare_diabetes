import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def download_diabetes_data():
    print("Bộ dữ liệu Diabetes 130-US hospitals")
    # fetch dataset 
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
    
    # data (as pandas dataframes) 
    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
    
    # Kết hợp features và targets
    df = pd.concat([X, y], axis=1)
    
    # Tạo thư mục data nếu chưa có
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Lưu vào file csv
    df.to_csv('data/raw/diabetes_data_raw.csv', index=False)
    print(f"Đã tải xong và lưu vào data/diabetes_data_raw.csv. Số lượng record: {len(df)}")
    
    # Lưu thông tin biến
    variables = diabetes_130_us_hospitals_for_years_1999_2008.variables
    variables.to_csv('data/raw/variables_info.csv', index=False)
    print("Đã lưu thông tin vào data/raw/variables_info.csv")

if __name__ == "__main__":
    download_diabetes_data()
