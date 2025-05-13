"""
Batch Effect Validation for Molecular Feature Data Augmentation

This module provides functionality to validate batch effects in molecular feature data augmentation.
It analyzes original and augmented molecular feature datasets to calculate how well batch effects
are preserved during the augmentation process. The module generates comparison metrics
and visualizations to help assess the quality of different augmentation methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Color definitions for molecular features
COLORS = {
    'MolecularFeature': '#41ab5d',       # Medium-Dark Yellow-Green
    'MolecularFeature_PCluster': '#006837', # Darkest Yellow-Green
    'MolecularFeature_NCluster': '#ffffd4', # Very Light Yellow
    'MolecularFeature_Other': '#bdbdbd',     # Light Grey
    'Original': 'blue',
    'Augmented': 'green'
}


def validate_batch_effects(original_path, augmented_path, output_dir, batch_col='Batch'):
    """
    Simple batch effect validation for molecular feature data augmentation.
    
    Parameters:
    -----------
    original_path : str
        Path to original molecular feature data CSV
    augmented_path : str
        Path to augmented molecular feature data CSV
    output_dir : str
        Directory to save results
    batch_col : str
        Name of batch column
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    print("Loading data...")
    original = pd.read_csv(original_path)
    augmented = pd.read_csv(augmented_path)
    
    # Identify molecular feature columns
    n_cluster_cols = [col for col in original.columns if col.startswith('N_Cluster_')]
    p_cluster_cols = [col for col in original.columns if col.startswith('P_Cluster_')]
    molecular_feature_cols = n_cluster_cols + p_cluster_cols
    
    # Extract augmentation methods
    augmented_only = augmented[~augmented['Row_names'].isin(original['Row_names'])]
    methods = set()
    for row_name in augmented_only['Row_names']:
        if '_' in row_name:
            methods.add(row_name.split('_')[-1])
    
    # Add "original" as a method
    methods.add('original')
    methods = list(methods)
    print(f"Found {len(methods)} methods: {', '.join(methods)}")
    
    # Create dictionary of datasets by method
    datasets = {'original': original}
    for method in methods:
        if method != 'original':
            mask = augmented['Row_names'].str.endswith(f'_{method}')
            datasets[method] = augmented[mask]
    
    # Calculate batch effect metrics
    result_rows = []
    
    # Variance explained by batch for original data
    original_var_explained = calculate_variance_explained_by_batch(
        datasets['original'], batch_col, molecular_feature_cols)
    
    for method in methods:
        data = datasets[method]
        
        # Calculate variance explained by batch
        var_explained = calculate_variance_explained_by_batch(
            data, batch_col, molecular_feature_cols)
        
        # Calculate preservation ratio (how close to original)
        if method == 'original':
            preservation = 1.0
        else:
            # Closer to 1.0 is better (neither too high nor too low)
            ratio = var_explained / original_var_explained if original_var_explained > 0 else 0
            preservation = 1.0 - min(1.0, abs(ratio - 1.0))
        
        result_rows.append({
            'Method': method,
            'Variance_Explained_By_Batch': var_explained,
            'Preservation_Score': preservation
        })
    
    # Create results table
    results = pd.DataFrame(result_rows)
    
    # Save results
    results.to_csv(os.path.join(output_dir, 'batch_effect_validation.csv'), index=False)
    
    # Create simple bar chart of preservation scores
    plt.figure(figsize=(10, 5))
    plt.bar(
        results['Method'], 
        results['Preservation_Score'],
        color=[COLORS['Original'] if m == 'original' else COLORS['Augmented'] for m in results['Method']]
    )
    plt.axhline(y=0.9, linestyle='--', color='green', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.title('Batch Effect Preservation by Method')
    plt.ylabel('Preservation Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_preservation.png'), dpi=300)
    
    print(f"Results saved to {output_dir}")
    print(results)
    
    return results


def calculate_variance_explained_by_batch(data, batch_col, molecular_feature_cols):
    """
    Calculate the average variance explained by batch across all molecular features.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset to analyze
    batch_col : str
        Column name containing batch information
    molecular_feature_cols : list
        List of column names for molecular feature features
        
    Returns:
    --------
    float
        Mean variance explained by batch across molecular features
    """
    if len(data[batch_col].unique()) < 2:
        return 0
    
    # Calculate variance explained for each molecular feature
    var_explained_values = []
    
    for molecular_feature in molecular_feature_cols:
        if molecular_feature in data.columns:
            # Total variance
            total_var = data[molecular_feature].var()
            
            if total_var > 0:
                # Between-batch variance
                batch_means = data.groupby(batch_col)[molecular_feature].mean()
                between_var = batch_means.var() * len(data) / len(batch_means)
                
                # Variance explained ratio
                var_ratio = between_var / total_var
                var_explained_values.append(var_ratio)
    
    # Return mean variance explained
    return np.mean(var_explained_values) if var_explained_values else 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate batch effect preservation in augmented data.'
    )
    parser.add_argument(
        '--original', 
        type=str, 
        default=r"C:\Users\ms\Desktop\hyper\data\n_p_r2.csv",
        help='Path to original molecular feature data CSV file'
    )
    parser.add_argument(
        '--augmented', 
        type=str, 
        default=r"C:\Users\ms\Desktop\hyper\output\augment\molecular_feature\root\augmented_molecular_feature_data.csv",
        help='Path to augmented molecular feature data CSV file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=r"C:\Users\ms\Desktop\hyper\output\augment\molecular_feature\root\batch_validation",
        help='Directory to save validation results'
    )
    parser.add_argument(
        '--batch-col', 
        type=str, 
        default='Batch',
        help='Name of batch column (default: Batch)'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validate_batch_effects(args.original, args.augmented, args.output, args.batch_col)