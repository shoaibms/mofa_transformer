"""
Metabolomics Data Transformation Evaluation Script
-----------------------------------------------

This script evaluates the effectiveness of different data transformations in metabolomics analysis
using MA plots and Relative Standard Deviation (RSD) analysis.

Key Visualizations:
1. MA Plots: Assess intensity-dependent ratio variations
   - M: log ratio (measure of differential expression)
   - A: average log intensity (measure of overall expression)
   
2. RSD Analysis: Evaluate measurement precision and reproducibility
   - Boxplots comparing RSD distribution before/after transformation
   - Histograms showing RSD frequency distribution

The script processes multiple transformation methods (anscombe, asinh, boxcox, glog, log, sqrt, 
yeojohnson) and generates comparative visualizations for each.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def extract_n_cluster_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract N_Cluster columns from dataframe using regex."""
    return df.filter(regex="N_Cluster")

def plot_ma_transform(before_df: pd.DataFrame, 
                     after_df: pd.DataFrame, 
                     title: str, 
                     cmap: str = 'BuGn', 
                     title_size: int = 16, 
                     label_size: int = 14, 
                     tick_size: int = 12):
    """
    Generate MA plots comparing data before and after transformation.
    
    MA plots are used to visualize intensity-dependent ratio of changes:
    - M (y-axis): log2 ratio between two samples (log fold change)
    - A (x-axis): average log2 intensity of two samples
    
    Args:
        before_df: Data before transformation
        after_df: Data after transformation
        title: Plot title
        cmap: Colormap for scatter points
        title_size: Font size for title
        label_size: Font size for axis labels
        tick_size: Font size for tick labels
    """
    # Add 1 to avoid log(0)
    before = before_df + 1
    after = after_df + 1

    # Calculate M and A values
    M_before = np.log2(before.iloc[:, 0]) - np.log2(before.iloc[:, 1])
    A_before = 0.5 * (np.log2(before.iloc[:, 0]) + np.log2(before.iloc[:, 1]))
    M_after = np.log2(after.iloc[:, 0]) - np.log2(after.iloc[:, 1])
    A_after = 0.5 * (np.log2(after.iloc[:, 0]) + np.log2(after.iloc[:, 1]))

    # Create subplot figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot before transformation
    scatter1 = axes[0].scatter(A_before, M_before, alpha=0.5, c=M_before, cmap=cmap)
    lowess_before = sm.nonparametric.lowess(M_before, A_before, frac=0.3)
    axes[0].plot(lowess_before[:, 0], lowess_before[:, 1], color='red')
    axes[0].set_title(f'MA-transform Before Transformation - {title}', fontsize=title_size)
    axes[0].set_xlabel('A (average log-intensity)', fontsize=label_size)
    axes[0].set_ylabel('M (log ratio)', fontsize=label_size)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
    fig.colorbar(scatter1, ax=axes[0], label='M Before')

    # Plot after transformation
    scatter2 = axes[1].scatter(A_after, M_after, alpha=0.5, c=M_after, cmap=cmap)
    lowess_after = sm.nonparametric.lowess(M_after, A_after, frac=0.3)
    axes[1].plot(lowess_after[:, 0], lowess_after[:, 1], color='red')
    axes[1].set_title(f'MA-transform After Transformation - {title}', fontsize=title_size)
    axes[1].set_xlabel('A (average log-intensity)', fontsize=label_size)
    axes[1].set_ylabel('M (log ratio)', fontsize=label_size)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
    fig.colorbar(scatter2, ax=axes[1], label='M After')

    plt.tight_layout()
    plt.show()

def calculate_rsd(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate Relative Standard Deviation (RSD) for each column.
    
    RSD = (Standard Deviation / Mean) * 100
    Used to assess measurement precision and reproducibility.
    """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    rsd = (std_dev / mean) * 100
    return rsd

def plot_rsd(before_df: pd.DataFrame, 
             after_df: pd.DataFrame, 
             title: str, 
             before_color: str = '#8ae391', 
             after_color: str = '#0c3b10', 
             title_size: int = 16, 
             label_size: int = 14, 
             tick_size: int = 12) -> tuple:
    """
    Generate RSD comparison plots before and after transformation.
    
    Creates two subplots:
    1. Boxplot showing RSD distribution
    2. Histogram showing RSD frequency distribution
    
    Args:
        before_df: Data before transformation
        after_df: Data after transformation
        title: Plot title
        before_color/after_color: Colors for before/after data
        title_size/label_size/tick_size: Font sizes
        
    Returns:
        tuple: (RSD values before transformation, RSD values after transformation)
    """
    rsd_before = calculate_rsd(before_df)
    rsd_after = calculate_rsd(after_df)

    results_df = pd.DataFrame({
        'RSD_Before': rsd_before,
        'RSD_After': rsd_after
    })

    # Handle infinite and NA values
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    results_df.dropna(inplace=True)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot
    axes[0].boxplot([results_df['RSD_Before'], results_df['RSD_After']], 
                    labels=['Before', 'After'], 
                    patch_artist=True,
                    boxprops=dict(facecolor=before_color, color=before_color))
    axes[0].set_title(f'Boxplot of Relative Standard Deviation (RSD) - {title}', 
                      fontsize=title_size)
    axes[0].set_ylabel('RSD (%)', fontsize=label_size)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_size)

    # Histogram
    axes[1].hist([results_df['RSD_Before'], results_df['RSD_After']], 
                 bins=30, alpha=0.5, label=['Before', 'After'], 
                 color=[before_color, after_color])
    axes[1].set_title(f'Histogram of Relative Standard Deviation (RSD) - {title}', 
                      fontsize=title_size)
    axes[1].set_ylabel('Frequency', fontsize=label_size)
    axes[1].legend(fontsize=label_size)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_size)

    plt.tight_layout()
    plt.show()

    return rsd_before, rsd_after

def save_metrics_to_csv(metrics: dict, file_path: str):
    """Save computed metrics to CSV file."""
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index').transpose()
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")

def main():
    """Main execution function."""
    # Define file paths for original and transformed data
    file_paths = {
        'original': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if.csv",
        'anscombe': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_anscombe.csv",
        'asinh': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_asinh.csv",
        'boxcox': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_boxcox.csv",
        'glog': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_glog.csv",
        'log': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_log.csv",
        'sqrt': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_sqrt.csv",
        'yeojohnson': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_yeojohnson.csv"
    }

    # Load and process original data
    original_df = load_data(file_paths['original'])
    original_n_cluster = extract_n_cluster_data(original_df)
    original_median = original_n_cluster.median().mean()

    # Initialize metrics dictionary
    metrics = {
        'Transformation': [],
        'RSD_Before': [],
        'RSD_After': [],
        'rMAD_Percentage_of_Median_Before': [],
        'rMAD_Percentage_of_Median_After': []
    }

    # Process each transformation
    for key, path in file_paths.items():
        if key == 'original':
            continue
        
        transformed_df = load_data(path)
        transformed_n_cluster = extract_n_cluster_data(transformed_df)
        title = key.replace('_', ' ').title()

        # Generate plots and calculate metrics
        plot_ma_transform(original_n_cluster, transformed_n_cluster, title)
        rsd_before, rsd_after = plot_rsd(original_n_cluster, transformed_n_cluster, title)

        # Store metrics
        metrics['Transformation'].append(title)
        metrics['RSD_Before'].append(rsd_before.mean())
        metrics['RSD_After'].append(rsd_after.mean())
        metrics['rMAD_Percentage_of_Median_Before'].append(rsd_before.mean())
        metrics['rMAD_Percentage_of_Median_After'].append(rsd_after.mean())

    # Save results
    save_metrics_to_csv(metrics, os.path.join(os.path.dirname(file_paths['original']), 
                                            'transformation_metrics.csv'))

if __name__ == "__main__":
    main()