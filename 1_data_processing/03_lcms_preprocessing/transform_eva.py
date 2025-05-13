"""
Data Transformation Evaluation Script for Metabolomics Analysis
-----------------------------------------------------------

This script creates facet grid visualisations to compare original and transformed 
metabolomics data using Coefficient of Variation (CV) and relative Median Absolute 
Deviation (rMAD) metrics.

The script generates:
1. Violin plots comparing CV distributions before/after transformations
2. Violin plots comparing rMAD distributions before/after transformations
3. Facet grid layouts for multiple transformation methods

Key metrics:
- CV (Coefficient of Variation): Measures relative variability, calculated as std/mean
- rMAD (relative Median Absolute Deviation): Robust measure of variability, 
  calculated as median(|x - median(x)|)/median(x)

Each transformation method is compared against the original data in a separate facet,
allowing for easy visual comparison of their effects on data distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_n_cluster_data(df):
    """
    Extract N_Cluster columns from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Subset containing only N_Cluster columns
    """
    return df.filter(regex="N_Cluster")

def calculate_cv(series):
    """
    Calculate Coefficient of Variation for a data series.
    
    CV = (standard deviation / mean) * 100
    
    Args:
        series (pd.Series): Input data series
        
    Returns:
        float: Coefficient of Variation as percentage
    """
    return np.std(series) / np.mean(series)

def calculate_rmad(series):
    """
    Calculate relative Median Absolute Deviation for a data series.
    
    rMAD = median(|x - median(x)|) / median(x)
    
    Args:
        series (pd.Series): Input data series
        
    Returns:
        float: relative Median Absolute Deviation ratio
    """
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    return mad / median

def calculate_metrics(df):
    """
    Calculate CV and rMAD for all columns in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (CV values, rMAD values) for all columns
    """
    cv_values = df.apply(calculate_cv)
    rmad_values = df.apply(calculate_rmad)
    return cv_values, rmad_values

def load_data(original_path, transformed_paths):
    """
    Load original and transformed datasets.
    
    Args:
        original_path (str): Path to original data file
        transformed_paths (dict): Dictionary of transformation names and file paths
        
    Returns:
        tuple: (original dataframe, dictionary of transformed dataframes)
    """
    original_df = pd.read_csv(original_path)
    transformed_dfs = {name: pd.read_csv(path) for name, path in transformed_paths.items()}
    return original_df, transformed_dfs

def prepare_data_for_plotting(original_df, transformed_dfs):
    """
    Prepare data for visualisation by calculating metrics and formatting for plotting.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        transformed_dfs (dict): Dictionary of transformed datasets
        
    Returns:
        tuple: (CV dataframe for plotting, rMAD dataframe for plotting)
    """
    orig_n_cluster = extract_n_cluster_data(original_df)
    orig_cv, orig_rmad = calculate_metrics(orig_n_cluster)
    
    cv_dfs = []
    rmad_dfs = []
    
    for name, df in transformed_dfs.items():
        trans_n_cluster = extract_n_cluster_data(df)
        trans_cv, trans_rmad = calculate_metrics(trans_n_cluster)
        
        # Prepare CV data
        cv_df = pd.DataFrame({
            'Original': orig_cv,
            'Transformed': trans_cv
        }).reset_index().melt(id_vars='index', var_name='Type', value_name='CV')
        cv_df['Transformation'] = name
        
        # Prepare rMAD data
        rmad_df = pd.DataFrame({
            'Original': orig_rmad,
            'Transformed': trans_rmad
        }).reset_index().melt(id_vars='index', var_name='Type', value_name='rMAD')
        rmad_df['Transformation'] = name
        
        cv_dfs.append(cv_df)
        rmad_dfs.append(rmad_df)
    
    return pd.concat(cv_dfs, ignore_index=True), pd.concat(rmad_dfs, ignore_index=True)

def plot_facet_grid(df, metric, font_size):
    """
    Create facet grid plot comparing original and transformed data distributions.
    
    Args:
        df (pd.DataFrame): Prepared data for plotting
        metric (str): Metric name ('CV' or 'rMAD')
        font_size (dict): Dictionary containing font size specifications
    """
    g = sns.FacetGrid(df, col="Transformation", col_wrap=3, sharex=False, sharey=False, height=4)
    g.map_dataframe(sns.violinplot, x='Type', y=metric, palette='BuGn', split=True)
    g.set_axis_labels("Data Type", metric)
    g.set_titles("{col_name}")
    
    # Customise font sizes
    for ax in g.axes.flat:
        ax.set_xlabel('Data Type', fontsize=font_size['xlabel'])
        ax.set_ylabel(metric, fontsize=font_size['ylabel'])
        ax.tick_params(axis='x', labelsize=font_size['ticks'])
        ax.tick_params(axis='y', labelsize=font_size['ticks'])
    
    g.fig.subplots_adjust(top=0.9, hspace=0.4)
    g.fig.suptitle(f'Violin Plot of {metric}', fontsize=font_size['title'])
    plt.show()

def main():
    # Define file paths
    original_path = r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if.csv"
    transformed_paths = {
        'anscombe': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_anscombe.csv",
        'asinh': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_asinh.csv",
        'boxcox': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_boxcox.csv",
        'glog': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_glog.csv",
        'log': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_log.csv",
        'sqrt': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_sqrt.csv",
        'yeojohnson': r"C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_yeojohnson.csv"
    }

    # Define font sizes for plots
    font_size = {
        'title': 16,
        'xlabel': 14,
        'ylabel': 14,
        'ticks': 12,
        'legend': 12
    }

    # Load and prepare data
    original_df, transformed_dfs = load_data(original_path, transformed_paths)
    cv_df, rmad_df = prepare_data_for_plotting(original_df, transformed_dfs)

    # Generate visualisation plots
    plot_facet_grid(cv_df, 'CV', font_size)
    plot_facet_grid(rmad_df, 'rMAD', font_size)

if __name__ == "__main__":
    main()