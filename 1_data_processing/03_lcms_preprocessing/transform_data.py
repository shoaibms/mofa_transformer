"""
Data Transformation Script for Metabolomics Analysis
-------------------------------------------------

This script applies various data transformations commonly used in metabolomics data analysis
to address non-normality, heteroscedasticity, and improve data distribution properties.

Implemented transformations:
1. Log transformation: ln(x + 1) to handle zeros and reduce right skewness
2. Square root: √x to reduce right skewness, less aggressive than log
3. Box-Cox: Parametric power transformation family to normalise data
4. Yeo-Johnson: Similar to Box-Cox but handles negative values
5. Asinh (inverse hyperbolic sine): Similar to log but handles zeros/negative values
6. Generalised log (glog): Handles zeros/small values better than regular log
7. Anscombe: Variance-stabilising transformation, especially for count data

Input:
- CSV file containing metabolomics data with N_Cluster columns
- Non-cluster columns are preserved unchanged

Output:
- Separate CSV files for each transformation method
- Original column names and structure are preserved
"""

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

def load_data(file_path):
    """Load the dataset and identify cluster columns."""
    data = pd.read_csv(file_path)
    cluster_columns = [col for col in data.columns if col.startswith('N_Cluster')]
    return data, cluster_columns

def apply_log_transform(data, columns):
    """
    Natural logarithm transformation: ln(x + 1)
    Adding 1 prevents issues with zero values while preserving order
    """
    transformed = data.copy()
    transformed[columns] = transformed[columns].apply(lambda x: np.log(x + 1))
    return transformed

def apply_sqrt_transform(data, columns):
    """
    Square root transformation: √x
    Less aggressive than log for reducing right skewness
    """
    transformed = data.copy()
    transformed[columns] = transformed[columns].apply(np.sqrt)
    return transformed

def apply_boxcox_transform(data, columns):
    """
    Box-Cox transformation: Parametric transformation to approximate normality
    Adds small constant (1) to handle zero values
    """
    transformed = data.copy()
    for col in columns:
        transformed[col], _ = boxcox(transformed[col] + 1)
    return transformed

def apply_yeojohnson_transform(data, columns):
    """
    Yeo-Johnson transformation: Extension of Box-Cox that handles negative values
    """
    transformed = data.copy()
    pt = PowerTransformer(method='yeo-johnson')
    transformed[columns] = pt.fit_transform(transformed[columns])
    return transformed

def apply_asinh_transform(data, columns):
    """
    Inverse hyperbolic sine transformation
    Similar to log but handles zeros and negative values naturally
    """
    transformed = data.copy()
    transformed[columns] = np.arcsinh(transformed[columns])
    return transformed

def apply_glog_transform(data, columns, a=0.75):
    """
    Generalized logarithm transformation with parameter a
    Better handling of small values compared to regular log
    Args:
        a: Small constant to stabilize variance (default: 0.75)
    """
    transformed = data.copy()
    transformed[columns] = transformed[columns].apply(
        lambda x: np.log(x + np.sqrt(x**2 + a**2))
    )
    return transformed

def apply_anscombe_transform(data, columns):
    """
    Anscombe transformation: 2√(x + 3/8)
    Variance-stabilizing transformation, particularly useful for count data
    """
    transformed = data.copy()
    transformed[columns] = 2 * np.sqrt(transformed[columns] + 3/8)
    return transformed

def save_transformed_data(transformed_data, original_path, suffix):
    """Save transformed data to a new CSV file with appropriate suffix."""
    output_path = original_path.replace('.csv', f'_{suffix}.csv')
    transformed_data.to_csv(output_path, index=False)
    return output_path

def main():
    # Input file path
    file_path = r'C:\Users\ms\Desktop\data_chem\data\transformation\n_l_if.csv'
    
    # Load data and identify columns to transform
    data, cluster_columns = load_data(file_path)
    
    # Apply and save each transformation
    transformations = {
        'log': apply_log_transform,
        'sqrt': apply_sqrt_transform,
        'boxcox': apply_boxcox_transform,
        'yeojohnson': apply_yeojohnson_transform,
        'asinh': apply_asinh_transform,
        'glog': apply_glog_transform,
        'anscombe': apply_anscombe_transform
    }
    
    for name, transform_func in transformations.items():
        transformed = transform_func(data, cluster_columns)
        save_transformed_data(transformed, file_path, name)
    
    print("All transformations completed successfully.")

if __name__ == "__main__":
    main()