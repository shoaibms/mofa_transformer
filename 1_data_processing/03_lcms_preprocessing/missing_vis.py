"""
Missing Data Visualisation for Metabolomics Data
----------------------------------------------

This script creates a visual representation of missing values in metabolomics data,
specifically focusing on N_Cluster variables. It generates a heatmap where:
- Missing values are shown in green
- Non-missing values are shown in light gray
- Percentages of missing/non-missing values are displayed in the legend

The visualisation helps in:
- Understanding the pattern of missing data
- Identifying potential systematic missingness
- Quantifying the extent of missing data across variables and entries

Color Scheme:
- Non-missing: Light gray (#e1e6eb)
- Missing: Green (#2ca02c)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_missing_data_matrix(df, font_size):
    """
    Create a heatmap visualization of missing data patterns.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing metabolomics data
        font_size (int): Base font size for plot text elements
    
    The function creates a heatmap where:
        - Each row represents an entry
        - Each column represents a variable
        - Missing values are highlighted in green
        - Non-missing values are shown in light gray
    """
    # Configure global plot settings
    plt.rcParams.update({'font.size': font_size})
    
    # Extract N_Cluster variables
    n_cluster_data = df[[col for col in df.columns if col.startswith('N_Cluster_')]]
    
    # Create missing value mask (True where value is missing)
    missing = n_cluster_data.isnull()

    # Calculate missing data statistics
    total_elements = missing.size
    missing_elements = missing.sum().sum()
    missing_percentage = (missing_elements / total_elements) * 100
    non_missing_percentage = 100 - missing_percentage

    # Define visualization parameters
    colors = {
        'non_missing': '#e1e6eb',  # Light gray
        'missing': '#2ca02c'       # Green
    }
    custom_cmap = ListedColormap([colors['non_missing'], colors['missing']])

    # Create and configure plot
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    
    # Remove plot borders for cleaner appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create heatmap
    ax.imshow(missing, 
              aspect='auto',
              interpolation='none',
              cmap=custom_cmap)

    # Configure axes
    ax.set_xlabel('Variables (N=807)')
    ax.set_ylabel('Entry (N=168)')
    # Remove tick marks to reduce visual clutter
    ax.set_xticks([])
    ax.set_yticks([])

    # Create and configure legend
    legend_labels = [
        'Non-Missing ({:.2f}%)'.format(non_missing_percentage),
        'Missing ({:.2f}%)'.format(missing_percentage)
    ]
    patches = [Patch(color=colors[key], label=label)
              for key, label in zip(['non_missing', 'missing'], legend_labels)]
    
    leg = ax.legend(handles=patches,
                   loc='upper right',
                   frameon=True)
    leg.get_frame().set_edgecolor('none')

    plt.tight_layout()
    plt.show()

def main():
    """Main function to load data and create visualization."""
    # Load metabolomics data
    data_path = r'C:\Users\ms\Desktop\data_chem\data\old_2\n_column_data_l_deducted.csv'
    data = pd.read_csv(data_path)
    
    # Create visualization with specified font size
    font_size = 15
    plot_missing_data_matrix(data, font_size)

if __name__ == "__main__":
    main()