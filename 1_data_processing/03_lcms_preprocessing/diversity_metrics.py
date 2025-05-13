"""
Imputation Quality Assessment Using Diversity Metrics
--------------------------------------------------

This script evaluates the quality of different imputation methods (median and RF)
using ecological diversity metrics. These metrics help assess how well the imputation
preserves the data's statistical properties and distribution characteristics.

Metrics calculated:
1. Richness: Number of unique values (assesses value diversity)
2. Shannon Entropy: Measures uncertainty/randomness in the distribution
3. Simpson's Diversity Index: Probability that two randomly selected values are different
4. Sparsity: Proportion of values equal to the mode (measures imputation uniformity)

Input:
- Two CSV files containing imputed data (median and RF methods)
- Files should contain N_Cluster columns for analysis

Output:
- CSV file with comparative metrics for both imputation methods
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy

def load_imputed_datasets(median_path, rf_path):
    """
    Load and prepare imputed datasets for analysis.
    
    Args:
        median_path: Path to median-imputed data
        rf_path: Path to RF-imputed data
        
    Returns:
        tuple: Numeric data from both datasets (median, rf)
    """
    # Load datasets with low_memory=False to handle mixed types
    median_data = pd.read_csv(median_path, low_memory=False)
    rf_data = pd.read_csv(rf_path, low_memory=False)
    
    # Extract N_Cluster columns and ensure numeric type
    n_cluster_cols = [col for col in rf_data.columns if col.startswith('N_Cluster')]
    
    return (
        median_data[n_cluster_cols].select_dtypes(include=[np.number]),
        rf_data[n_cluster_cols].select_dtypes(include=[np.number])
    )

def calculate_richness(data):
    """
    Calculate average number of unique values across columns.
    
    Higher richness indicates more diverse value distribution.
    """
    return data.apply(lambda x: len(x.unique()), axis=0).mean()

def calculate_shannon_entropy(data):
    """
    Calculate Shannon entropy to measure distribution uncertainty.
    
    Higher entropy indicates more even distribution of values.
    Higher values suggest better preservation of data variability.
    """
    def entropy_of_series(series):
        counts = series.value_counts()
        return entropy(counts)
    return data.apply(entropy_of_series).mean()

def calculate_simpsons_index(data):
    """
    Calculate Simpson's diversity index (1 - D).
    
    Ranges from 0 (no diversity) to 1 (infinite diversity).
    Represents probability that two randomly chosen values are different.
    """
    def simpsons_index_of_series(series):
        counts = series.value_counts()
        n = sum(counts)
        return 1 - sum((count/n)**2 for count in counts)
    return data.apply(simpsons_index_of_series).mean()

def calculate_sparsity(imputed_data):
    """
    Calculate average proportion of mode values in each column.
    
    Lower sparsity suggests better preservation of data variability.
    High sparsity might indicate over-reliance on central tendencies.
    """
    modes = imputed_data.mode().iloc[0]
    sparsity = imputed_data.eq(modes, axis=1).mean()
    return sparsity.mean()

def evaluate_imputation_quality(median_data, rf_data):
    """
    Calculate all diversity metrics for both imputation methods.
    
    Returns:
        DataFrame: Comparison of all metrics for both methods
    """
    metrics = {
        'Metric': ['Richness', 'Shannon Entropy', 'Simpson\'s Diversity Index', 'Sparsity'],
        'Median Imputed': [
            calculate_richness(median_data),
            calculate_shannon_entropy(median_data),
            calculate_simpsons_index(median_data),
            calculate_sparsity(median_data)
        ],
        'RF Imputed': [
            calculate_richness(rf_data),
            calculate_shannon_entropy(rf_data),
            calculate_simpsons_index(rf_data),
            calculate_sparsity(rf_data)
        ]
    }
    
    return pd.DataFrame(metrics)

def main():
    # Define input file paths
    median_path = r'C:\Users\ms\Desktop\data_chem\imputated\complete imputed files\n_column_data_r_deducted_imputed_median2.csv'
    rf_path = r'C:\Users\ms\Desktop\data_chem\imputated\complete imputed files\n_column_data_r_deducted_imputed_rf2.csv'
    
    # Load and prepare data
    median_data, rf_data = load_imputed_datasets(median_path, rf_path)
    
    # Calculate and compile all metrics
    results = evaluate_imputation_quality(median_data, rf_data)
    
    # Save results
    output_path = r'C:\Users\ms\Desktop\data_chem\imputated\complete imputed files\imputation_comparison_results.csv'
    results.to_csv(output_path, index=False)
    
    # Print results for immediate review
    print("\nImputation Quality Metrics:")
    print(results)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()