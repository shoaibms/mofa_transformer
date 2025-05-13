"""
Little's MCAR (Missing Completely At Random) Test Implementation
-------------------------------------------------------------

This script implements Little's MCAR test to evaluate the randomness of missing data
in metabolomics datasets. The test helps determine if data is Missing Completely
At Random (MCAR), which is a crucial assumption for many imputation methods.

Little's MCAR Test:
- Null hypothesis (H0): Data is Missing Completely At Random
- Alternative hypothesis (H1): Data is not Missing Completely At Random
- Interpretation: 
  * Low p-value (< 0.05): Reject H0, data is not MCAR
  * High p-value (≥ 0.05): Fail to reject H0, no evidence against MCAR

Input:
- CSV file containing metabolomics data with N_Cluster columns
- Only numeric N_Cluster columns are analyzed

Output:
- CSV file containing the p-value from Little's MCAR test
- Console output of the test result
"""

import pandas as pd
import numpy as np
from pyampute.exploration.mcar_statistical_tests import MCARTest

def load_metabolomics_data(file_path):
    """
    Load metabolomics data and extract numeric N_Cluster columns.
    
    Args:
        file_path (str): Path to the CSV file containing metabolomics data
    
    Returns:
        pd.DataFrame: DataFrame containing only numeric N_Cluster columns
    """
    # Load the complete dataset
    data = pd.read_csv(file_path)
    
    # Filter for numeric N_Cluster columns
    n_cluster_columns = [
        col for col in data.columns 
        if col.startswith('N_Cluster_') and np.issubdtype(data[col].dtype, np.number)
    ]
    
    return data[n_cluster_columns]

def perform_littles_mcar_test(data):
    """
    Perform Little's MCAR test on the provided data.
    
    Args:
        data (pd.DataFrame): DataFrame containing only the columns to be tested
    
    Returns:
        float: p-value from Little's MCAR test
    """
    # Initialize and perform Little's MCAR test
    mcar_tester = MCARTest(method="little")
    return mcar_tester.little_mcar_test(data)

def save_test_results(p_value, output_path):
    """
    Save the MCAR test results to a CSV file.
    
    Args:
        p_value (float): p-value from Little's MCAR test
        output_path (str): Path where the results should be saved
    """
    result_df = pd.DataFrame({'P-Value': [p_value]})
    result_df.to_csv(output_path, index=False)

def main():
    # File paths
    data_path = r'C:\Users\ms\Desktop\data_chem\data\n_column_data_l_deducted.csv'
    result_path = r'C:\Users\ms\Desktop\data_chem\data\littles_mcar_test_results.csv'
    
    # Load and process data
    n_cluster_data = load_metabolomics_data(data_path)
    
    # Perform Little's MCAR test
    p_value = perform_littles_mcar_test(n_cluster_data)
    
    # Save and display results
    save_test_results(p_value, result_path)
    
    # Print results with interpretation
    print("\nLittle's MCAR Test Results:")
    print(f"P-value: {p_value}")
    print("\nInterpretation:")
    if p_value < 0.05:
        print("The data is likely NOT Missing Completely At Random (MCAR)")
    else:
        print("No evidence against the MCAR assumption")

if __name__ == "__main__":
    main()