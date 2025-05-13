"""
Normality Testing for Metabolomics Data
-------------------------------------

This script performs comprehensive normality testing on metabolomics data using two
complementary approaches:

1. Shapiro-Wilk Test:
   - Tests the null hypothesis that the data was drawn from a normal distribution
   - More powerful for small to medium sample sizes (n < 5000)
   - Sensitive to slight deviations from normality in large samples

2. Anderson-Darling Test:
   - Tests normality with greater weight on tails of distribution
   - More sensitive to deviations in the tails than Shapiro-Wilk
   - Generally more powerful than Shapiro-Wilk for detecting non-normality

Input:
- CSV files containing metabolomics data with N_Cluster columns
- Each file represents different transformation methods

Output:
- CSV files with normality test results for each variable
- Test statistics and p-values for both methods
- Warning if number of variables doesn't match expected count (807)

Note: Results are used by transformation_normality_test_plot.py for visualization
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, anderson
import os

def load_data(file_path):
    """
    Load CSV data file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    return pd.read_csv(file_path)

def extract_n_cluster_data(df):
    """
    Extract N_Cluster columns from the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Subset containing only N_Cluster columns
    """
    return df.filter(regex="N_Cluster")

def shapiro_wilk_test(data):
    """
    Perform Shapiro-Wilk normality test on each column.
    
    Args:
        data (pandas.DataFrame): Input data
        
    Returns:
        list: Tuples of (column_name, statistic, p_value)
        
    Note:
        H0: Data follows normal distribution
        H1: Data does not follow normal distribution
        Reject H0 if p-value < alpha (typically 0.05)
    """
    shapiro_results = []
    for col in data.columns:
        stat, p = shapiro(data[col].dropna())
        shapiro_results.append((col, stat, p))
    return shapiro_results

def anderson_darling_test(data):
    """
    Perform Anderson-Darling normality test on each column.
    
    Args:
        data (pandas.DataFrame): Input data
        
    Returns:
        list: Tuples of (column_name, statistic, p_value)
        
    Note:
        - Uses 5% significance level for critical value
        - P-value approximation: 
          * 0.05 if statistic > critical_value (reject H0)
          * 1.0 if statistic ≤ critical_value (fail to reject H0)
    """
    anderson_results = []
    for col in data.columns:
        result = anderson(data[col].dropna())
        stat = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level
        try:
            # Find critical value for 5% significance level
            index_5_percent = significance_levels.tolist().index(5.0)
            critical_value_5_percent = critical_values[index_5_percent]
            # Simplified p-value approximation
            p_value = 0.05 if stat > critical_value_5_percent else 1.0
        except ValueError:
            critical_value_5_percent = None
            p_value = None
        anderson_results.append((col, stat, p_value))
    return anderson_results

def combine_results(shapiro_results, anderson_results):
    """
    Combine results from both normality tests.
    
    Args:
        shapiro_results (list): Results from Shapiro-Wilk test
        anderson_results (list): Results from Anderson-Darling test
        
    Returns:
        list: Combined results with both test statistics and p-values
    """
    combined_results = []
    for shapiro_result, anderson_result in zip(shapiro_results, anderson_results):
        variable = shapiro_result[0]
        shapiro_stat = shapiro_result[1]
        shapiro_p = shapiro_result[2]
        anderson_stat = anderson_result[1]
        anderson_p = anderson_result[2]
        combined_results.append([variable, shapiro_stat, shapiro_p, anderson_stat, anderson_p])
    return combined_results

def save_combined_results_to_csv(results, file_path):
    """
    Save combined test results to CSV file.
    
    Args:
        results (list): Combined results from both tests
        file_path (str): Output file path
    """
    df = pd.DataFrame(results, columns=[
        'Variable', 
        'Shapiro_Statistic', 
        'Shapiro_p_value', 
        'Anderson_Statistic', 
        'Anderson_p_value'
    ])
    df.to_csv(file_path, index=False)
    print(f"Combined test results saved to {file_path}")

def main():
    """
    Main execution function.
    
    Process each transformation file:
    1. Load data and extract N_Cluster columns
    2. Perform both normality tests
    3. Combine and save results
    4. Verify expected number of variables
    """
    file_paths = [
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_anscombe.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_asinh.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_boxcox.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_glog.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_log.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_sqrt.csv",
        r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_yeojohnson.csv"
    ]
    
    for file_path in file_paths:
        # Load and process data
        data = load_data(file_path)
        n_cluster_data = extract_n_cluster_data(data)

        # Perform normality tests
        shapiro_results = shapiro_wilk_test(n_cluster_data)
        anderson_results = anderson_darling_test(n_cluster_data)

        # Save results
        combined_results = combine_results(shapiro_results, anderson_results)
        combined_results_file = file_path.replace('.csv', '_normality_results.csv')
        save_combined_results_to_csv(combined_results, combined_results_file)

        # Validate number of variables
        if len(n_cluster_data.columns) != 807:
            print(f"Warning: Number of processed variables ({len(n_cluster_data.columns)}) "
                  f"does not match expected count (807) in {file_path}")

if __name__ == "__main__":
    main()