"""
Imputation Validation Visualization Script
----------------------------------------

This script creates pairwise box plots to compare original and imputed metabolomics data.
It visualizes the distribution of values for randomly selected N_Cluster variables,
allowing assessment of how well the imputation preserves the original data characteristics.

Features:
- Randomly selects 20 N_Cluster variables for comparison
- Handles missing values in original data using median imputation
- Creates publication-ready box plots with customized aesthetics
- Maintains reproducibility through fixed random seed

Input:
- Original dataset CSV with missing values
- Imputed dataset CSV with filled values
- Both datasets must contain N_Cluster columns

Output:
- Publication-quality box plot comparing original vs imputed distributions
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_datasets(original_path, imputed_path):
    """
    Load original and imputed datasets from CSV files.
    
    Args:
        original_path (str): Path to original dataset
        imputed_path (str): Path to imputed dataset
    
    Returns:
        tuple: Original and imputed DataFrames
    """
    original_data = pd.read_csv(original_path)
    imputed_data = pd.read_csv(imputed_path)
    return original_data, imputed_data

def select_random_clusters(data, n_clusters=20, random_seed=42):
    """
    Randomly select N_Cluster columns from the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        n_clusters (int): Number of clusters to select
        random_seed (int): Random seed for reproducibility
    
    Returns:
        list: Selected column names
    """
    np.random.seed(random_seed)
    cluster_columns = [col for col in data if 'N_Cluster' in col]
    return np.random.choice(cluster_columns, size=n_clusters, replace=False)

def prepare_data_for_plotting(original_data, imputed_data, selected_columns):
    """
    Prepare data for visualization by handling missing values and melting DataFrames.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        imputed_data (pd.DataFrame): Imputed dataset
        selected_columns (list): Columns to process
    
    Returns:
        pd.DataFrame: Combined data ready for plotting
    """
    # Extract and process original data
    original_subset = original_data[selected_columns].copy()
    original_subset.fillna(original_subset.median(), inplace=True)
    original_melted = original_subset.melt(var_name='Variable', value_name='Value')
    original_melted['Type'] = 'Original'
    
    # Process imputed data
    imputed_subset = imputed_data[selected_columns].copy()
    imputed_melted = imputed_subset.melt(var_name='Variable', value_name='Value')
    imputed_melted['Type'] = 'Imputed'
    
    # Combine datasets
    return pd.concat([original_melted, imputed_melted])

def create_comparison_boxplot(data, palette, plot_params):
    """
    Create and customize boxplot comparing original and imputed values.
    
    Args:
        data (pd.DataFrame): Prepared data for plotting
        palette (dict): Color palette for different data types
        plot_params (dict): Plot customization parameters
    """
    plt.figure(figsize=plot_params['figsize'])
    
    # Create boxplot
    sns.boxplot(
        data=data,
        x='Variable',
        y='Value',
        hue='Type',
        palette=palette,
        boxprops=dict(edgecolor='none')
    )
    
    # Customize plot appearance
    plt.xticks(rotation=90, fontsize=plot_params['tick_fontsize'])
    plt.yticks(fontsize=plot_params['tick_fontsize'])
    plt.xlabel('Variable', fontsize=plot_params['label_fontsize'])
    plt.ylabel('Value', fontsize=plot_params['label_fontsize'])
    plt.title(plot_params['title'], fontsize=plot_params['title_fontsize'])
    
    # Customize legend
    plt.legend(
        title='Data Type',
        title_fontsize=plot_params['legend_title_fontsize'],
        fontsize=plot_params['legend_fontsize']
    )
    
    plt.tight_layout()

def main():
    # File paths
    original_path = r"C:\Users\ms\Desktop\data_chem\data\old_2\n_column_data_l_deducted.csv"
    imputed_path = r"C:\Users\ms\Desktop\data_chem\imputated\complete imputed files\n_column_data_l_deducted_imputed_rf2.csv"
    
    # Color palette
    palette = {
        "Original": "#F0CACA",
        "Imputed": "#F3E2C4"
    }
    
    # Plot parameters
    plot_params = {
        'figsize': (20, 10),
        'tick_fontsize': 12,
        'label_fontsize': 14,
        'title_fontsize': 16,
        'legend_fontsize': '12',
        'legend_title_fontsize': '13',
        'title': 'Pairwise Box Plot of 20 Random N_Cluster Variables from Original and Imputed Data'
    }
    
    # Load and process data
    original_data, imputed_data = load_datasets(original_path, imputed_path)
    selected_columns = select_random_clusters(original_data)
    combined_data = prepare_data_for_plotting(original_data, imputed_data, selected_columns)
    
    # Create visualization
    create_comparison_boxplot(combined_data, palette, plot_params)
    plt.show()

if __name__ == "__main__":
    main()