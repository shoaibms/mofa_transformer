"""
Outlier Imputation Impact Analysis
--------------------------------

This script analyzes and visualizes the impact of Random Forest (RF) outlier imputation
on metabolomics data by comparing pre- and post-imputation values for metabolite clusters.

The analysis:
1. Calculates the relative impact of imputation on each metabolite
2. Identifies the most significantly affected metabolites
3. Visualizes the changes with error bars for the top 30 most impacted variables

Key metrics:
- Impact = (post_mean - pre_mean) / pre_mean
- Error bars represent standard deviation
- Separate colors distinguish pre- and post-imputation values

Input files:
- Pre-imputation data: Original metabolomics data
- Post-imputation data: Data after RF outlier imputation

Output:
- Error bar plot showing the top 30 most impacted variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_datasets(pre_path, post_path):
    """
    Load pre- and post-imputation datasets.
    
    Args:
        pre_path (str): Path to pre-imputation data
        post_path (str): Path to post-imputation data
    
    Returns:
        tuple: (pre-imputation DataFrame, post-imputation DataFrame)
    """
    data_before = pd.read_csv(pre_path)
    data_after = pd.read_csv(post_path)
    return data_before, data_after

def get_cluster_columns(data, prefix='P_Cluster'):
    """
    Extract cluster column names from the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        prefix (str): Prefix for cluster columns
    
    Returns:
        list: Column names matching the prefix
    """
    return [col for col in data.columns if col.startswith(prefix)]

def calculate_impact_metrics(data_before, data_after, cluster_cols):
    """
    Calculate impact metrics for each cluster variable.
    
    Args:
        data_before (pd.DataFrame): Pre-imputation data
        data_after (pd.DataFrame): Post-imputation data
        cluster_cols (list): List of cluster column names
    
    Returns:
        pd.DataFrame: DataFrame containing impact metrics
    """
    metrics = {
        'Variable': cluster_cols,
        'Impact': [],
        'Before': [],
        'After': [],
        'Before_Std': [],
        'After_Std': []
    }
    
    for col in cluster_cols:
        before_mean = data_before[col].mean()
        after_mean = data_after[col].mean()
        
        # Calculate relative impact, handling zero values
        impact = (after_mean - before_mean) / before_mean if before_mean != 0 else 0
        
        metrics['Impact'].append(impact)
        metrics['Before'].append(before_mean)
        metrics['After'].append(after_mean)
        metrics['Before_Std'].append(data_before[col].std())
        metrics['After_Std'].append(data_after[col].std())
    
    impact_df = pd.DataFrame(metrics)
    impact_df['AbsImpact'] = impact_df['Impact'].abs()
    return impact_df.sort_values('AbsImpact', ascending=False).reset_index(drop=True)

def plot_impact_comparison(impact_df, n_top=30):
    """
    Create error bar plot comparing pre- and post-imputation values.
    
    Args:
        impact_df (pd.DataFrame): DataFrame with impact metrics
        n_top (int): Number of top impacted variables to plot
    """
    top_impact = impact_df.head(n_top)
    
    plt.figure(figsize=(12, 8))
    
    # Plot pre-imputation values with error bars
    plt.errorbar(range(n_top), top_impact['Before'], yerr=top_impact['Before_Std'],
                fmt='o', color='#65c792', ecolor='#7fcdb0', capsize=5, 
                label='Pre', alpha=0.7)
    
    # Plot post-imputation values with error bars
    plt.errorbar(range(n_top), top_impact['After'], yerr=top_impact['After_Std'],
                fmt='o', color='#0e3823', ecolor='#2f5240', capsize=5, 
                label='Post', alpha=0.7)
    
    # Add reference line and styling
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.ylabel('Value', fontsize=18)
    plt.xlabel('Variables', fontsize=18)
    plt.title('Pre and Post RF Imputation of Outliers', fontsize=16)
    plt.xticks(range(n_top), top_impact['Variable'], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)
    
    plt.tight_layout()
    plt.show()

def main():
    # File paths
    pre_imputation_path = r'C:\Users\ms\Desktop\data_chem\data\impute_missing_value\p_l_rf.csv'
    post_imputation_path = r'C:\Users\ms\Desktop\data_chem\data\outlier_fixed\p_l_if.csv'
    
    # Load data
    data_before, data_after = load_datasets(pre_imputation_path, post_imputation_path)
    
    # Get cluster columns
    cluster_cols = get_cluster_columns(data_before)
    
    # Calculate impact metrics
    impact_df = calculate_impact_metrics(data_before, data_after, cluster_cols)
    
    # Create visualization
    plot_impact_comparison(impact_df)

if __name__ == "__main__":
    main()