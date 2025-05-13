"""
Column Filter Script for Metabolomics Data Quality Control
-------------------------------------------------------

This script filters metabolomics data based on replication criteria to ensure data quality.
It removes N_Cluster columns that have insufficient replication, defined as having 3 or more
missing values for any Entry when TMT = 0.

Filtering criteria:
- Focuses on N_Cluster columns (metabolite measurements)
- Checks replication only for control samples (TMT = 0)
- Requires < 3 missing values per Entry to retain a column
- Preserves all necessary metadata columns

Input:
- Excel file (.xlsx) containing metabolomics data
- Must contain N_Cluster columns and specified metadata columns
- Must have TMT and Entry columns for filtering

Output:
- CSV file containing filtered data
- Includes all metadata columns and N_Cluster columns meeting replication criteria
"""

import pandas as pd

def load_metabolomics_data(file_path):
    """
    Load metabolomics data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        pd.DataFrame: Loaded metabolomics data
    """
    return pd.read_excel(file_path, engine='openpyxl')

def identify_metabolite_columns(df):
    """
    Identify N_Cluster columns for filtering.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        list: List of column names starting with 'N_Cluster'
    """
    return [col for col in df.columns if col.startswith('N_Cluster')]

def filter_columns_by_replication(df, columns_to_check):
    """
    Filter columns based on replication criteria.
    
    Criteria:
    - Examines only control samples (TMT = 0)
    - Requires < 3 missing values per Entry
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_check (list): List of N_Cluster columns
    
    Returns:
        list: Columns meeting replication criteria
    """
    columns_to_keep = []
    
    # Subset data for control samples
    control_samples = df[df['TMT'] == 0]
    
    for col in columns_to_check:
        # Count missing values per Entry
        missing_counts = control_samples.groupby('Entry')[col].apply(
            lambda x: x.isnull().sum()
        )
        
        # Keep column if all Entries have < 3 missing values
        if all(missing_counts < 3):
            columns_to_keep.append(col)
    
    return columns_to_keep

def get_final_columns(columns_to_keep):
    """
    Combine filtered metabolite columns with necessary metadata columns.
    
    Args:
        columns_to_keep (list): Filtered N_Cluster columns
    
    Returns:
        list: Complete list of columns for final dataset
    """
    metadata_columns = [
        'Row_names', 'Vac_id', 'Entry', 'Maximum_RT_Shift_Neg',
        'Tissue', 'Batch', 'TMT', 'Rep', 'Day'
    ]
    return metadata_columns + columns_to_keep

def save_filtered_data(filtered_df, output_path):
    """
    Save filtered DataFrame to CSV file.
    
    Args:
        filtered_df (pd.DataFrame): Filtered DataFrame
        output_path (str): Path for output CSV file
    """
    filtered_df.to_csv(output_path, index=False)

def main():
    # Input and output file paths
    input_path = 'C:\\Users\\ms\\Desktop\\data_chem\\combine\\n_column_data_r.xlsx'
    output_path = 'C:\\Users\\ms\\Desktop\\data_chem\\combine\\n_column_data_r_clean.csv'
    
    # Load data
    df = load_metabolomics_data(input_path)
    
    # Identify and filter columns
    metabolite_columns = identify_metabolite_columns(df)
    columns_to_keep = filter_columns_by_replication(df, metabolite_columns)
    
    # Create filtered dataset
    final_columns = get_final_columns(columns_to_keep)
    filtered_df = df[final_columns]
    
    # Save results
    save_filtered_data(filtered_df, output_path)
    print("Data filtering completed successfully.")

if __name__ == "__main__":
    main()