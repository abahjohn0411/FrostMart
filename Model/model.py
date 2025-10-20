import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score





## Configuration and Path Setup
# Define the directory and paths for saving artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_best_regressor.pkl')
SCHEMA_PATH = os.path.join(MODEL_DIR, 'feature_schema.json')

#1. CORE DATA FUNCTIONS


## Data Loading and Merging
def load_and_merge_data(sales_file, product_file, store_file, weather_file, supplier_file):
    """Loads and merges all raw data files into a single DataFrame."""
    
    # Load all datasets
    sales_df = pd.read_csv(sales_file)
    product_df = pd.read_csv(product_file)
    store_df = pd.read_csv(store_file)
    weather_df = pd.read_csv(weather_file)
    supplier_df = pd.read_csv(supplier_file)

    # Standardize ID columns to string for robust merging
    for df in [sales_df, product_df]:
        if 'Product_ID' in df.columns: df['Product_ID'] = df['Product_ID'].astype(str)
    for df in [sales_df, store_df]:
        if 'Store_ID' in df.columns: df['Store_ID'] = df['Store_ID'].astype(str)
    if 'Supplier_ID' in product_df.columns: product_df['Supplier_ID'] = product_df['Supplier_ID'].astype(str)
    if 'Supplier_ID' in supplier_df.columns: supplier_df['Supplier_ID'] = supplier_df['Supplier_ID'].astype(str)

    # Perform the sequential merges (left joins ensure all sales records are kept)
    df = sales_df.merge(product_df, on='Product_ID', how='left')
    df = df.merge(store_df, on='Store_ID', how='left')
    df = df.merge(supplier_df, on='Supplier_ID', how='left')
    df = df.merge(weather_df, on=['Week_Number', 'Region'], how='left')
    
    df = df.drop(columns=['Supplier_ID'])
    
    return df


## Feature Engineering and Encoding
def feature_engineer_and_encode(df):
    """Applies feature engineering (cyclic encoding) and categorical encoding (OHE/Frequency)."""
    
    # Create temporal index for time-based splitting
    week_map = {week: i for i, week in enumerate(df['Week_Number'].unique().tolist())}
    df['Time_Index'] = df['Week_Number'].map(week_map)

    df_ml = df.copy()
    df_ml = df_ml.drop(columns=['Product_Name', 'Supplier_Name'])

    # 1. Cyclic Encoding for Week_Number
    df_ml['Week_Num'] = df_ml['Week_Number'].str.split('-W').str[1].astype(int)
    MAX_WEEKS = 53
    df_ml['Week_sin'] = np.sin(2 * np.pi * df_ml['Week_Num'] / MAX_WEEKS)
    df_ml['Week_cos'] = np.cos(2 * np.pi * df_ml['Week_Num'] / MAX_WEEKS)
    df_ml = df_ml.drop(columns=['Week_Number', 'Week_Num'])

    # 2. One-Hot Encoding
    ohe_cols = ['Store_ID', 'Region', 'Product_Category', 'Holiday_Flag']
    for col in ohe_cols:
        df_ml[col] = df_ml[col].astype(str)
    df_ml = pd.get_dummies(df_ml, columns=ohe_cols, drop_first=True)

    # 3. Frequency Encoding (High Cardinality: Product_ID)
    product_counts = df_ml['Product_ID'].value_counts()
    df_ml['Product_ID_FreqEncoded'] = df_ml['Product_ID'].map(product_counts)
    df_ml = df_ml.drop(columns=['Product_ID'])
    
    return df_ml


## Feature Selection and Scaling
def select_and_scale_features(df_ml, k_features=15):
    """Scales numerical features and applies SelectKBest to select top features."""
    
    X_base = df_ml.drop(columns=['Units_Sold'])
    y_base = df_ml['Units_Sold']
    time_index = X_base['Time_Index']
    X_base = X_base.drop(columns=['Time_Index'])

    # Scaling numerical features
    numerical_cols = X_base.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_final = X_base.copy()
    X_final[numerical_cols] = scaler.fit_transform(X_base[numerical_cols])

    # SelectKBest with mutual_info_regression
    selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
    selector.fit(X_final, y_base)
    
    # Get selected features
    selected_features_mask = selector.get_support()
    X_selected = X_final.iloc[:, selected_features_mask]
    
    return X_selected, y_base, time_index, scaler, selector # Return scaler/selector for deployment


# 2. UTILITY AND MODELING FUNCTIONS


## Temporal Data Splitting
def temporal_split(X, y, time_index, train_ratio=0.8):
    """Splits data temporally based on a ratio of unique time steps."""
    
    unique_time_steps = time_index.unique()
    split_point = int(train_ratio * len(unique_time_steps))
    split_index_value = unique_time_steps[split_point]

    X_train = X[time_index < split_index_value]
    X_val = X[time_index >= split_index_value]
    y_train = y[time_index < split_index_value]
    y_val = y[time_index >= split_index_value]
    
    return X_train, X_val, y_train, y_val



## Model Training and Evaluation
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """Trains a model, predicts, and returns a dictionary of metrics."""
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return {
        'Model': model_name, 
        'RMSE': round(rmse, 4), 
        'MAE': round(mae, 4), 
        'R2': round(r2, 4),
        'model_object': model # Return model object for saving
    }


# 3. MAIN ORCHESTRATION FUNCTION




## Main Orchestration Function
def main():
    """Orchestrates the entire data processing, modeling, and evaluation pipeline."""

    # 1. Data Preparation
    raw_files = {
        'sales_file': 'weekly_sales.csv',
        'product_file': 'product_details.csv',
        'store_file': 'store_info.csv',
        'weather_file': 'weather_data.csv',
        'supplier_file': 'supplier_info.csv'
    }
    
    merged_df = load_and_merge_data(**raw_files)
    df_processed = feature_engineer_and_encode(merged_df)
    
    X_selected, y_base, time_index, scaler, selector = select_and_scale_features(df_processed, k_features=15)
    
    # 2. Data Split
    X_train, X_val, y_train, y_val = temporal_split(X_selected, y_base, time_index, train_ratio=0.8)
    
    print("\n--- Training and Evaluating Models ---")
    print(f"Features used for training: {X_train.columns.tolist()}")
    
    # 3. Model Training & Evaluation (Panel Data)
    all_metrics = []

    # Model 1: Ridge Regression (Linear Baseline)
    ridge_results = train_and_evaluate_model(
        Ridge(alpha=1.0), X_train, y_train, X_val, y_val, 'Ridge Regression'
    )
    all_metrics.append({k:v for k,v in ridge_results.items() if k!='model_object'})

    # Model 2: Random Forest Regressor (Non-linear Model)
    rf_results = train_and_evaluate_model(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), 
        X_train, y_train, X_val, y_val, 'Random Forest Regressor'
    )
    all_metrics.append({k:v for k,v in rf_results.items() if k!='model_object'})

    # Model 3: Gradient Boosting Regressor (Boosting Model - replacing SARIMA)
    gbr_results = train_and_evaluate_model(
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        X_train, y_train, X_val, y_val, 'Gradient Boosting Regressor'
    )
    all_metrics.append({k:v for k,v in gbr_results.items() if k!='model_object'})
    
    # 4. Final Summary
    df_metrics = pd.DataFrame(all_metrics)
    
    print("\n==============================================")
    print("FINAL MODEL COMPARISON (Panel Data)")
    print("==============================================")
    print(df_metrics.T)

    # 5. Model Persistence (Saving the best model, e.g., Random Forest)
    best_model = rf_results['model_object']
    feature_columns = X_selected.columns.tolist()

    os.makedirs(MODEL_DIR, exist_ok = True)
    
    # Save Model (Random Forest is usually the best performer in initial tests)
    with open (BEST_MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved to: {BEST_MODEL_PATH}")

    # Save Schema
    schema = {'feature_columns': feature_columns}
    with open (SCHEMA_PATH, 'w') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"Schema saved to: {SCHEMA_PATH}")



## Execution Block
if __name__ == '__main__':
    main()