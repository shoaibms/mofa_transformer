"""
Logistic Regression Results Analysis and Visualization
---------------------------------------------------

This script analyzes logistic regression results from multiple metabolomics datasets
to identify significant associations between metabolites and experimental factors.

The analysis:
1. Processes results from four complementary datasets (n_l, n_r, p_l, p_r)
2. Calculates percentage of significant associations for each predictor
3. Generates visualizations comparing significance patterns across datasets
4. Creates both tabular and graphical summaries of the findings

Input files:
- CSV files containing logistic regression results for each dataset
- Expected columns: Genotype_G2, TMT_1, Day_2, Rep

Output:
- Summary CSV with percentages of significant associations
- Grouped bar plot visualization of the results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def define_file_paths():
    """Define paths for input files and their corresponding dataset names."""
    return {
        r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\n_column_data_l_clean_results.csv': 'n_l',
        r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\n_column_data_r_clean_results.csv': 'n_r',
        r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\p_column_data_l_clean_results.csv': 'p_l',
        r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\p_column_data_r_clean_results.csv': 'p_r'
    }

def calculate_significant_associations(df, column, threshold=0.05):
    """
    Calculate number of significant associations for a predictor column.
    
    Args:
        df: DataFrame containing regression results
        column: Name of the predictor column
        threshold: Significance threshold (default: 0.05)
    
    Returns:
        Number of significant associations
    """
    return np.sum(df[column].apply(lambda x: isinstance(x, (int, float)) and abs(x) > threshold))

def process_dataset(file_path, dataset_name, df_length):
    """
    Process a single dataset and calculate summary statistics.
    
    Args:
        file_path: Path to the dataset file
        dataset_name: Name identifier for the dataset
        df_length: Length of the dataframe for percentage calculation
    
    Returns:
        Dictionary containing summary statistics
    """
    df = pd.read_csv(file_path)
    
    return {
        'Dataset': dataset_name,
        'Genotype': calculate_significant_associations(df, 'Genotype_G2') / df_length * 100,
        'Treatment': calculate_significant_associations(df, 'TMT_1') / df_length * 100,
        'Day': calculate_significant_associations(df, 'Day_2') / df_length * 100,
        'Replicate': calculate_significant_associations(df, 'Rep') / df_length * 100
    }

def create_summary_visualization(summary_df, output_path, font_size=21):
    """
    Create and save a grouped bar plot visualization.
    
    Args:
        summary_df: DataFrame containing summary statistics
        output_path: Path to save the visualization
        font_size: Base font size for the plot (default: 21)
    """
    # Prepare data for plotting
    summary_melted = summary_df.melt(
        id_vars='Dataset',
        var_name='Predictor',
        value_name='Percentage'
    )
    
    # Create color palette
    color_palette = sns.color_palette("Greens", len(summary_df['Dataset']))
    
    # Create plot
    plt.figure(figsize=(8, 5.85))
    sns.barplot(
        x='Dataset',
        y='Percentage',
        hue='Predictor',
        data=summary_melted,
        palette=color_palette
    )
    
    # Customize appearance
    plt.xlabel('Dataset', fontsize=font_size)
    plt.ylabel('% of Significant Associations', fontsize=font_size)
    plt.xticks(fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    plt.legend(
        title='Predictor',
        fontsize=font_size - 2,
        title_fontsize=font_size,
        framealpha=0.6
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    # Define file paths
    file_paths = define_file_paths()
    
    # Process each dataset
    summary_data = []
    df_sample = pd.read_csv(next(iter(file_paths)))  # Get length from first file
    
    for file_path, dataset_name in file_paths.items():
        summary = process_dataset(file_path, dataset_name, len(df_sample))
        summary_data.append(summary)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_file_path = r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\summary_table_significant_associations.csv'
    summary_df.to_csv(summary_file_path, index=False)
    
    # Create and save visualization
    grouped_bar_plot_path = r'C:\Users\ms\Desktop\data_chem\data\data\LogisticRegression\grouped_bar_significant_associations_datasets.png'
    create_summary_visualization(summary_df, grouped_bar_plot_path)
    
    print("Summary Table saved to:", summary_file_path)
    print("Grouped Bar Chart saved to:", grouped_bar_plot_path)

if __name__ == "__main__":
    main()