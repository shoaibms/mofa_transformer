"""
Comprehensive Outlier Analysis for Metabolomics Data
--------------------------------------------------

This script implements multiple outlier detection methods and their visualization
for metabolomics data analysis. It combines traditional statistical approaches
with modern machine learning techniques for robust outlier detection.

Methods implemented:
1. Z-Score: Traditional statistical method using standard deviations
2. IQR: Interquartile range method for detecting statistical outliers
3. Isolation Forest: Unsupervised learning method for anomaly detection
4. Elliptic Envelope: Assumes Gaussian distribution for outlier detection
5. Mahalanobis Distance: Accounts for covariance structure in multivariate data
6. Robust PCA: Principal Component Analysis with robust covariance estimation
7. Local Outlier Factor: Density-based outlier detection

Visualizations:
- PCA projection with outlier highlighting
- t-SNE projection with outlier highlighting
- Performance evaluation metrics for each method
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import mahalanobis

def load_and_preprocess_data(file_path, categorical_vars):
    """
    Load and preprocess the metabolomics data.
    
    Args:
        file_path: Path to the input CSV file
        categorical_vars: List of categorical variables to encode
    
    Returns:
        Tuple of (processed DataFrame, standardized data, cluster columns)
    """
    df = pd.read_csv(file_path)
    
    # Encode categorical variables
    label_encoders = {var: LabelEncoder().fit(df[var]) for var in categorical_vars}
    for var in categorical_vars:
        df[var] = label_encoders[var].transform(df[var])
    
    # Extract and standardize P_Cluster variables
    p_cluster_columns = [col for col in df.columns if 'P_Cluster' in col]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[p_cluster_columns])
    
    return df, scaled_data, p_cluster_columns

def mahalanobis_distance(data, contamination=0.05):
    """
    Calculate Mahalanobis distance and identify outliers.
    
    Args:
        data: Input data matrix
        contamination: Expected proportion of outliers
    
    Returns:
        Boolean array indicating outlier status for each sample
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = np.sqrt(np.sum((centered_data @ inv_cov_matrix) * centered_data, axis=1))
    
    # Calculate adaptive threshold using median absolute deviation
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    threshold = median_dist + 3 * 1.4826 * mad
    outliers = distances > threshold
    
    print(f"Mahalanobis Distance - Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}, Threshold: {threshold:.2f}")
    print(f"Number of outliers detected: {np.sum(outliers)}")
    return outliers

def robust_pca_outliers(data, contamination=0.05):
    """
    Detect outliers using robust PCA and Elliptic Envelope.
    
    Args:
        data: Input data matrix
        contamination: Expected proportion of outliers
    
    Returns:
        Boolean array indicating outlier status for each sample
    """
    pca = PCA(n_components=2)
    scores = pca.fit_transform(data)
    robust_cov = EllipticEnvelope(contamination=contamination, random_state=42)
    return robust_cov.fit_predict(scores) == -1

def apply_outlier_methods(data, contamination=0.05):
    """
    Apply multiple outlier detection methods.
    
    Args:
        data: Input data matrix
        contamination: Expected proportion of outliers
    
    Returns:
        Dictionary containing outlier detection results for each method
    """
    # Z-Score method
    z_scores = np.abs((data - data.mean()) / data.std())
    z_score_outliers = (z_scores > 5.5).any(axis=1)

    # IQR method
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    iqr_outliers = ((data < (Q1 - 7.0 * IQR)) | (data > (Q3 + 7.0 * IQR))).any(axis=1)

    # Machine learning methods
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    elliptic_envelope = EllipticEnvelope(contamination=contamination, random_state=42)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    
    return {
        'Z-Score': z_score_outliers,
        'IQR': iqr_outliers,
        'Isolation Forest': isolation_forest.fit_predict(data) == -1,
        'Elliptic Envelope': elliptic_envelope.fit_predict(data) == -1,
        'Mahalanobis Distance': mahalanobis_distance(data, contamination),
        'Robust PCA': robust_pca_outliers(data, contamination),
        'Local Outlier Factor': lof.fit_predict(data) == -1
    }

def evaluate_methods(outlier_results):
    """
    Calculate the percentage of samples identified as outliers by each method.
    """
    return {method: np.mean(results) * 100 for method, results in outlier_results.items()}

def plot_dimensionality_reduction(data, outliers, method_name, reduction_type='PCA'):
    """
    Visualize outliers using dimensionality reduction techniques.
    
    Args:
        data: Input data matrix
        outliers: Boolean array indicating outlier status
        method_name: Name of the outlier detection method
        reduction_type: Type of dimensionality reduction ('PCA' or 't-SNE')
    """
    # Choose dimensionality reduction method
    if reduction_type == 'PCA':
        reducer = PCA(n_components=2)
        xlabel = 'Principal Component 1'
        ylabel = 'Principal Component 2'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
    
    reduced_data = reducer.fit_transform(data)
    
    # Create visualization
    plt.figure(figsize=(6, 4))
    colors = ['#bfdbc1', '#425944']  # Light green to dark green
    cmap = mcolors.LinearSegmentedColormap.from_list("Custom", colors, N=100)
    
    color_array = np.zeros(len(data))
    color_array[outliers] = 1
    
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=color_array, 
               cmap=cmap, alpha=0.8, s=50)
    
    plt.title(f"{reduction_type}: Outliers detected by {method_name}", fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=15)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Normal Data',
                  markerfacecolor='#bfdbc1', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Outliers',
                  markerfacecolor='#425944', markersize=12)
    ]
    plt.legend(handles=legend_elements, fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    file_path = r'C:\Users\ms\Desktop\data_chem\data\impute_missing_value\p_l_rf.csv'
    categorical_vars = ['Day', 'Batch', 'Genotype', 'Treatment', 'Replication']
    contamination = 0.05
    
    # Load and process data
    df, scaled_data, p_cluster_columns = load_and_preprocess_data(file_path, categorical_vars)
    data = pd.DataFrame(scaled_data, columns=p_cluster_columns)
    
    # Detect outliers using multiple methods
    outlier_results = apply_outlier_methods(data.values, contamination)
    
    # Evaluate and save results
    evaluation_results = evaluate_methods(outlier_results)
    evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index', 
                                         columns=['Score'])
    evaluation_df.to_csv('outlier_detection_evaluation_results.csv')
    print("\nEvaluation Results:")
    print(evaluation_df)
    
    # Visualize results
    for method, outliers in outlier_results.items():
        plot_dimensionality_reduction(data.values, outliers, method, 'PCA')
        plot_dimensionality_reduction(data.values, outliers, method, 't-SNE')

if __name__ == "__main__":
    main()