"""
Relative Median Absolute Deviation (rMAD) Based Variable Selection
---------------------------------------------------------------

This script performs variable selection in metabolomics data based on the relative Median 
Absolute Deviation (rMAD) criterion. Variables with high rMAD values, indicating high 
variability relative to their median, are removed from the dataset.

Key features:
- Calculates rMAD for metabolite clusters (N_Cluster and P_Cluster variables)
- Removes variables above a specified rMAD threshold
- Preserves non-cluster variables
- Generates both cleaned datasets and lists of removed variables

Input:
- CSV files containing metabolomics data with N_Cluster or P_Cluster columns
- rMAD threshold for variable removal (default: 30%)

Output:
- Cleaned CSV files with high-rMAD variables removed
- CSV files listing removed variables for documentation
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List

def calculate_rmad(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Relative Median Absolute Deviation (rMAD) for each column.
    
    rMAD = (median(|x - median(x)|) / median(x)) * 100
    
    Args:
        data: DataFrame containing numeric columns
        
    Returns:
        Series containing rMAD values for each column
    """
    median = data.median()
    mad = (np.abs(data - median)).median()
    rmad = (mad / median) * 100
    return rmad

def identify_cluster_columns(data: pd.DataFrame) -> List[str]:
    """
    Identify metabolite cluster columns in the dataset.
    
    Args:
        data: Input DataFrame
        
    Returns:
        List of column names starting with 'N_Cluster' or 'P_Cluster'
    """
    return [col for col in data.columns if col.startswith(('N_Cluster', 'P_Cluster'))]

def filter_variables(data: pd.DataFrame, rmad_values: pd.Series, 
                    threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter variables based on rMAD threshold.
    
    Args:
        data: Input DataFrame
        rmad_values: Series containing rMAD values for each column
        threshold: Maximum acceptable rMAD value
        
    Returns:
        Tuple containing:
        - DataFrame with high-rMAD variables removed
        - List of removed column names
    """
    columns_to_keep = rmad_values[rmad_values <= threshold].index.tolist()
    columns_to_remove = rmad_values[rmad_values > threshold].index.tolist()
    
    # Keep non-cluster columns and acceptable cluster columns
    non_cluster_cols = [col for col in data.columns if not col.startswith(('N_Cluster', 'P_Cluster'))]
    final_columns = non_cluster_cols + columns_to_keep
    
    clean_data = data[final_columns].copy()
    return clean_data, columns_to_remove

def save_results(clean_data: pd.DataFrame, removed_vars: List[str], 
                original_path: str) -> Tuple[str, str]:
    """
    Save cleaned data and list of removed variables.
    
    Args:
        clean_data: Filtered DataFrame
        removed_vars: List of removed column names
        original_path: Path to original input file
        
    Returns:
        Tuple containing paths to saved files
    """
    directory = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    # Save cleaned data
    clean_data_path = os.path.join(directory, f"cleaned_{filename}")
    clean_data.to_csv(clean_data_path, index=False)
    
    # Save list of removed variables
    removed_vars_path = os.path.join(directory, f"removed_variables_{filename}")
    pd.DataFrame(removed_vars, columns=["Removed_Variables"]).to_csv(removed_vars_path, index=False)
    
    return clean_data_path, removed_vars_path

def process_file(file_path: str, rmad_threshold: float = 30) -> None:
    """
    Process a single file to remove high-rMAD variables.
    
    Args:
        file_path: Path to input CSV file
        rmad_threshold: Maximum acceptable rMAD value (default: 30)
    """
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Process cluster columns
        cluster_columns = identify_cluster_columns(data)
        cluster_data = data[cluster_columns]
        
        # Calculate rMAD and filter variables
        rmad_values = calculate_rmad(cluster_data)
        clean_data, removed_variables = filter_variables(data, rmad_values, rmad_threshold)
        
        # Save results
        clean_path, removed_path = save_results(clean_data, removed_variables, file_path)
        
        print(f"\nResults for {os.path.basename(file_path)}:")
        print(f"Cleaned data saved to: {clean_path}")
        print(f"List of removed variables saved to: {removed_path}")
        print(f"Number of variables removed: {len(removed_variables)}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Input file paths
    file_paths = [
        r'C:\Users\ms\Desktop\data_chem\data\CV_rMAD\n_l_if_asinh.csv',
        r'C:\Users\ms\Desktop\data_chem\data\CV_rMAD\n_r_if_asinh.csv',
        r'C:\Users\ms\Desktop\data_chem\data\CV_rMAD\p_l_if_asinh.csv',
        r'C:\Users\ms\Desktop\data_chem\data\CV_rMAD\p_r_if_asinh.csv'
    ]
    
    # Process each file
    for file_path in file_paths:
        process_file(file_path)

if __name__ == "__main__":
    main()