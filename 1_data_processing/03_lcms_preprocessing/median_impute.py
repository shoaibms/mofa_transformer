"""
Median-Based Missing Value Imputation for Metabolomics Data
--------------------------------------------------------

This script performs a three-step imputation process for missing values in metabolomics data:
1. Group-wise median imputation based on experimental factors
2. Forward fill for any remaining missing values
3. Backward fill as a final step to ensure complete data

The imputation strategy preserves the data structure by:
- Considering experimental factors (Day, Batch, Genotype, Treatment)
- Using group-specific medians to maintain biological relevance
- Applying sequential imputation methods for comprehensive coverage

Input:
- CSV file containing metabolomics data with P_Cluster columns
- Categorical variables: Day, Batch, Genotype, Treatment
- Missing values represented as NaN

Output:
- CSV file with all missing values imputed
- Original column structure and data types preserved
"""

import pandas as pd

def load_and_prepare_data(file_path):
    """
    Load the dataset and convert categorical variables to appropriate data types.
    
    Args:
        file_path (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe with properly typed categorical columns
    """
    # Load the raw data
    data = pd.read_csv(file_path)
    
    # Convert experimental factors to categorical type for efficient grouping
    categorical_columns = ['Day', 'Batch', 'Genotype', 'Treatment']
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    
    return data

def identify_metabolite_columns(data):
    """
    Identify metabolite columns (P_Cluster) for imputation.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        list: Column names starting with 'P_Cluster_'
    """
    return [col for col in data.columns if col.startswith('P_Cluster_')]

def impute_missing_values(data, metabolite_columns, grouping_columns):
    """
    Perform three-step imputation for missing values.
    
    Steps:
    1. Group-wise median imputation
    2. Forward fill remaining NaNs
    3. Backward fill any remaining NaNs
    
    Args:
        data (pd.DataFrame): Input dataframe
        metabolite_columns (list): Columns to impute
        grouping_columns (list): Columns to group by for median imputation
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    imputed_data = data.copy()
    
    for column in metabolite_columns:
        # Step 1: Group-wise median imputation
        imputed_data[column] = (imputed_data.groupby(grouping_columns)[column]
                               .transform(lambda x: x.fillna(x.median())))
        
        # Step 2: Forward fill remaining missing values
        imputed_data[column] = imputed_data[column].fillna(method='ffill')
        
        # Step 3: Backward fill any remaining gaps
        imputed_data[column] = imputed_data[column].fillna(method='bfill')
    
    return imputed_data

def save_imputed_data(data, output_path):
    """
    Save the imputed data to CSV file.
    
    Args:
        data (pd.DataFrame): Imputed dataframe
        output_path (str): Path for output CSV file
    """
    data.to_csv(output_path, index=False)

def main():
    # File paths
    input_path = r'C:\Users\ms\Desktop\data_chem\data\old_2\p_column_data_l_deducted.csv'
    output_path = r'C:\Users\ms\Desktop\data_chem\imputated\p_column_data_l_deducted_imputed_median2.csv'
    
    # Define grouping columns for imputation
    grouping_columns = ['Day', 'Batch', 'Genotype', 'Treatment']
    
    # Load and prepare data
    data = load_and_prepare_data(input_path)
    
    # Identify columns for imputation
    metabolite_columns = identify_metabolite_columns(data)
    
    # Perform imputation
    imputed_data = impute_missing_values(data, metabolite_columns, grouping_columns)
    
    # Save results
    save_imputed_data(imputed_data, output_path)
    
    print("Data imputation complete and file saved.")

if __name__ == "__main__":
    main()
