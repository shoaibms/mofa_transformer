"""
Outlier Detection and Removal Using Isolation Forest
-------------------------------------------------

This script implements outlier detection and removal for metabolomics data using
the Isolation Forest algorithm. Isolation Forest is particularly effective for
high-dimensional data and does not make assumptions about the data distribution.

Algorithm details:
- Uses Isolation Forest to identify outliers in metabolomic features
- Performs column-wise outlier detection after standardisation
- Replaces identified outliers with NaN for subsequent imputation
- Uses decision function scores to determine outlier threshold

Parameters:
- contamination: Expected proportion of outliers (set to 0.05 or 5%)
- threshold: Uses 5th percentile of decision scores as cutoff

Input:
- CSV file containing metabolomics data with P_Cluster columns
- Non-cluster columns are preserved unchanged

Output:
- CSV file with outliers replaced by NaN values
- Original column structure and non-outlier values are preserved
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def load_data(file_path):
    """
    Load the dataset and identify P_Cluster columns.
    
    Args:
        file_path (str): Path to the input CSV file
        
    Returns:
        tuple: (DataFrame, list of P_Cluster columns)
    """
    df = pd.read_csv(file_path)
    cluster_columns = [col for col in df.columns if 'P_Cluster' in col]
    return df, cluster_columns

def initialise_isolation_forest(contamination=0.05, random_state=42):
    """
    Initialise the Isolation Forest model with specified parameters.
    
    Args:
        contamination (float): Expected proportion of outliers (default: 0.05)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        IsolationForest: Initialised model
    """
    return IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

def detect_outliers(data, model, scaler):
    """
    Detect outliers in a single feature using Isolation Forest.
    
    Args:
        data (pd.Series): Feature values to analyse
        model (IsolationForest): Initialised Isolation Forest model
        scaler (StandardScaler): Initialised standard scaler
        
    Returns:
        tuple: (boolean mask of outliers, decision function scores)
    """
    # Reshape and scale the data
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Fit model and predict outliers
    model.fit(scaled_data)
    outliers = model.predict(scaled_data) == -1
    
    # Get decision scores
    decision_scores = model.decision_function(scaled_data)
    
    return outliers, decision_scores

def process_outliers(df, columns, model, threshold_percentile=5):
    """
    Process outliers for all specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process
        model (IsolationForest): Initialised model
        threshold_percentile (int): Percentile for decision score threshold
        
    Returns:
        pd.DataFrame: DataFrame with outliers replaced by NaN
    """
    df_cleaned = df.copy()
    scaler = StandardScaler()
    
    for col in columns:
        # Detect outliers for current column
        outliers, decision_scores = detect_outliers(df[col], model, scaler)
        
        # Calculate threshold based on decision scores
        threshold = np.percentile(decision_scores, threshold_percentile)
        
        # Replace outliers with NaN based on threshold
        mask = outliers & (decision_scores < threshold)
        df_cleaned.loc[mask, col] = np.nan
        
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        output_path (str): Path for output file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

def main():
    # File paths
    input_path = 'C:/Users/ms/Desktop/data_chem/data/outlier/p_l_rf.csv'
    output_path = 'C:/Users/ms/Desktop/data_chem/data/outlier/p_l_rf_outliers_removed.csv'
    
    # Load data
    df, cluster_columns = load_data(input_path)
    
    # Initialise model
    model = initialise_isolation_forest()
    
    # Process outliers
    df_cleaned = process_outliers(df, cluster_columns, model)
    
    # Save results
    save_cleaned_data(df_cleaned, output_path)

if __name__ == "__main__":
    main()