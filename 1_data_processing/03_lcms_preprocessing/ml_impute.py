"""
Advanced Missing Value Imputation for Metabolomics Data
----------------------------------------------------

This script implements multiple sophisticated imputation methods for handling missing values
in metabolomics data, particularly focusing on P_Cluster measurements.

Implemented methods:
1. k-Nearest Neighbors (kNN): Imputes based on similar samples
2. Expectation-Maximization (EM): Iterative algorithm using maximum likelihood
3. Singular Value Decomposition (SVD): Matrix completion approach
4. Gaussian Process Regression (GPR): Probabilistic imputation using kernels

Features:
- Handles categorical variables through one-hot encoding
- Preserves data structure and relationships
- Multiple imputation approaches for comparison
- Maintains original column structure

Input:
- CSV file containing metabolomics data with P_Cluster columns
- Categorical variables: Day, Batch, Genotype, Treatment

Output:
- Separate CSV files for each imputation method
- Preserved column structure with imputed values
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from fancyimpute import SoftImpute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_prepare_data(file_path):
    """
    Load data and prepare it for imputation by encoding categorical variables.
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        tuple: (transformed data, columns to impute, column names)
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Define categorical variable handling
    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Day', 'Batch', 'Genotype', 'Treatment'])
        ],
        remainder='passthrough'
    )
    
    # Transform the data
    data_transformed = transformer.fit_transform(data)
    transformed_cols = transformer.get_feature_names_out()
    
    # Identify columns for imputation
    cols_to_impute = [col for col in transformed_cols 
                      if col.startswith('remainder__P_Cluster')]
    
    return data_transformed, cols_to_impute, transformed_cols

def perform_imputation(data_transformed, columns, transformed_cols, imputer, output_file):
    """
    Perform imputation using specified method and save results.
    
    Args:
        data_transformed: Transformed input data
        columns: Columns to impute
        transformed_cols: All column names
        imputer: Initialized imputer object
        output_file: Path for saving results
    """
    data_frame = pd.DataFrame(data_transformed, columns=transformed_cols)
    print(f"\nStarting imputation with {imputer.__class__.__name__}")
    print("Sample of data before imputation:\n", data_frame[columns].head())
    
    data_frame[columns] = imputer.fit_transform(data_frame[columns])
    print("Sample of data after imputation:\n", data_frame[columns].head())
    
    data_frame.to_csv(output_file, index=False)
    print(f"Imputed data saved to: {output_file}")

def main():
    # Input/output paths
    input_file = r'C:\Users\ms\Desktop\data_chem\data\old_2\p_column_data_r_deducted.csv'
    output_base = r'C:\Users\ms\Desktop\data_chem\imputated\p_column_data_r_deducted_imputed_'
    
    # Prepare data
    data_transformed, cols_to_impute, transformed_cols = load_and_prepare_data(input_file)
    
    # Configure imputation methods
    imputers = {
        'knn': {
            'imputer': KNNImputer(n_neighbors=5),
            'file': f'{output_base}knn2.csv'
        },
        'em': {
            'imputer': IterativeImputer(random_state=0),
            'file': f'{output_base}em2.csv'
        },
        'svd': {
            'imputer': SoftImpute(),
            'file': f'{output_base}svd2.csv'
        },
        'gpr': {
            'imputer': IterativeImputer(
                estimator=GaussianProcessRegressor(
                    kernel=DotProduct() + WhiteKernel(),
                    alpha=1e-2,
                    random_state=0
                ),
                random_state=0,
                max_iter=15,
                tol=0.001
            ),
            'file': f'{output_base}gpr2.csv'
        }
    }
    
    # Perform imputations
    for method, config in imputers.items():
        print(f"\nPerforming {method.upper()} imputation...")
        perform_imputation(
            data_transformed,
            cols_to_impute,
            transformed_cols,
            config['imputer'],
            config['file']
        )

if __name__ == "__main__":
    main()