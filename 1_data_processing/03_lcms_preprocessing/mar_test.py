"""
Missing At Random (MAR) Analysis for Metabolomics Data
---------------------------------------------------

This script tests whether missing values in metabolomics data follow a Missing At Random (MAR) 
pattern by analyzing the relationship between missingness and observed variables using 
logistic regression.

Statistical Methodology:
- For each metabolite, creates a binary indicator (0/1) for missing values
- Uses logistic regression to model the relationship between missingness and predictors
- Significant relationships with predictors suggest MAR pattern
- Non-significant relationships suggest MCAR (Missing Completely At Random)

Predictors used:
- Genotype
- TMT (Treatment)
- Day
- Replication

Output:
- CSV file containing regression coefficients for each metabolite
- Coefficients indicate strength and direction of relationships with missingness
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load metabolomics data from CSV file.
    
    Args:
        file_path (str): Path to input CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(file_path)

def create_missingness_indicator(data, column):
    """
    Create binary indicator for missing values in specified column.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
        
    Returns:
        pd.Series: Binary indicator (1 for missing, 0 for present)
    """
    return data[column].isnull().astype(int)

def prepare_predictors(data, predictor_columns):
    """
    Prepare predictor variables with one-hot encoding.
    
    Args:
        data (pd.DataFrame): Input dataframe
        predictor_columns (list): List of predictor column names
        
    Returns:
        pd.DataFrame: One-hot encoded predictors
    """
    return pd.get_dummies(data[predictor_columns], drop_first=True)

def fit_logistic_model(X_train, X_test, y_train, y_test):
    """
    Fit logistic regression model with standardized features.
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target vectors
        
    Returns:
        tuple: (fitted model, scaled training data, scaled test data)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, X_train_scaled, X_test_scaled

def create_result_entry(column, model=None, coef_length=0):
    """
    Create DataFrame entry for regression results.
    
    Args:
        column (str): Metabolite column name
        model (LogisticRegression, optional): Fitted model
        coef_length (int): Length of coefficient vector
        
    Returns:
        pd.DataFrame: Single row of results
    """
    if model is None:
        return pd.DataFrame({
            'Metabolite': [column],
            'Intercept': [None],
            'Genotype_G2': [None],
            'TMT_1': [None],
            'Day_2': [None],
            'Rep': [None],
        })
    
    return pd.DataFrame({
        'Metabolite': [column],
        'Intercept': [model.intercept_[0]],
        'Genotype_G2': [model.coef_[0][0]] if coef_length > 0 else [None],
        'TMT_1': [model.coef_[0][1]] if coef_length > 1 else [None],
        'Day_2': [model.coef_[0][2]] if coef_length > 2 else [None],
        'Rep': [model.coef_[0][3]] if coef_length > 3 else [None],
    })

def analyze_metabolite(data, column, predictors):
    """
    Analyze missing value patterns for a single metabolite.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Metabolite column to analyze
        predictors (list): List of predictor variables
        
    Returns:
        pd.DataFrame: Analysis results for the metabolite
    """
    # Create missingness indicator
    y = create_missingness_indicator(data, column)
    
    # Skip if no variation in missingness
    if len(y.unique()) <= 1:
        return create_result_entry(column)
    
    # Prepare predictors
    X = prepare_predictors(data, predictors)
    
    try:
        # Split data and fit model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model, X_train_scaled, X_test_scaled = fit_logistic_model(
            X_train, X_test, y_train, y_test
        )
        
        return create_result_entry(column, model, len(model.coef_[0]))
        
    except Exception as e:
        print(f"Error analyzing {column}: {e}")
        return None

def main():
    # Input and output paths
    input_path = r'C:\Users\ms\Desktop\data_chem\combine\p_column_data_r_clean.csv'
    output_path = r'C:\Users\ms\Desktop\data_chem\combine\missing_data_analysis_results.csv'
    
    # Load data
    data = load_data(input_path)
    
    # Define predictors
    predictors = ['Genotype', 'TMT', 'day', 'Rep']
    
    # Analyze each metabolite
    results = []
    for column in data.columns:
        if column.startswith('P_Cluster_'):
            result = analyze_metabolite(data, column, predictors)
            if result is not None and not result.isnull().all().all():
                results.append(result)
    
    # Combine and save results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        final_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("No valid results to save.")

if __name__ == "__main__":
    main()