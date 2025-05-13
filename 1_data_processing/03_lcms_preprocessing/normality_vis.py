"""
Normality Test Visualization Script for Metabolomics Data Transformations
----------------------------------------------------------------------

This script creates density plots to visualize the results of normality tests 
(Shapiro-Wilk and Anderson-Darling) across different data transformation methods.
It helps assess which transformation methods most effectively normalize the data.

Visualizations:
1. Shapiro-Wilk test statistic distribution
2. Anderson-Darling test statistic distribution
3. Shapiro-Wilk p-value distribution
4. Anderson-Darling p-value distribution

Input:
- CSV files containing normality test results for different transformations
- Each file contains: Shapiro_Statistic, Anderson_Statistic, and their p-values

Output:
- Four density plots comparing distributions across transformation methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_transformation_data(file_paths):
    """
    Load and combine normality test results from multiple transformation files.
    
    Args:
        file_paths (list): Paths to CSV files containing normality test results
        
    Returns:
        tuple: Combined DataFrame and list of transformation method names
    """
    dfs = []
    transformation_methods = []
    
    for file_path in file_paths:
        # Load data and extract transformation method name
        df = pd.read_csv(file_path)
        transformation_method = file_path.split('_')[-3]
        
        # Add transformation method as a column
        df['Transformation'] = transformation_method
        
        dfs.append(df)
        transformation_methods.append(transformation_method)
    
    return pd.concat(dfs, ignore_index=True), transformation_methods

def create_density_plot(data, x_column, transformation_methods, hex_colors, font_sizes):
    """
    Create a density plot for a specific normality test metric.
    
    Args:
        data (DataFrame): Combined normality test results
        x_column (str): Column name to plot (e.g., 'Shapiro_Statistic')
        transformation_methods (list): Names of transformation methods
        hex_colors (list): Color palette for different transformations
        font_sizes (dict): Font size specifications
    """
    plt.figure(figsize=(6, 4))
    
    # Create KDE plot
    sns.kdeplot(
        data=data,
        x=x_column,
        hue='Transformation',
        fill=True,
        common_norm=False,
        palette=hex_colors
    )
    
    # Customize plot appearance
    plt.xlabel(x_column, fontsize=font_sizes['xlabel'])
    plt.ylabel('Density', fontsize=font_sizes['ylabel'])
    plt.xticks(fontsize=font_sizes['ticks'])
    plt.yticks(fontsize=font_sizes['ticks'])
    
    # Customize legend
    legend = plt.legend(
        title='Transformation Method',
        title_fontsize=font_sizes['legend_title'],
        fontsize=font_sizes['legend_text'],
        labels=transformation_methods
    )
    legend.get_frame().set_alpha(0.5)
    
    plt.tight_layout()
    plt.show()

def main():
    # File paths for different transformation results
    file_paths = [
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_anscombe_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_asinh_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_boxcox_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_glog_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_log_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_sqrt_normality_results.csv',
        r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if_yeojohnson_normality_results.csv'
    ]
    
    # Color palette for different transformations (green shades)
    hex_colors = ["#006400", "#228B22", "#32CD32", "#66CDAA", 
                 "#8FBC8F", "#98FB98", "#ADFF2F"]
    
    # Font size specifications
    font_sizes = {
        'xlabel': 16,
        'ylabel': 16,
        'ticks': 14,
        'legend_title': 16,
        'legend_text': 15
    }
    
    # Load and prepare data
    all_data, transformation_methods = get_transformation_data(file_paths)
    
    # Create plots for each normality test metric
    metrics = [
        'Shapiro_Statistic',
        'Anderson_Statistic',
        'Shapiro_p_value',
        'Anderson_p_value'
    ]
    
    for metric in metrics:
        create_density_plot(
            all_data,
            metric,
            transformation_methods,
            hex_colors,
            font_sizes
        )

if __name__ == "__main__":
    main()