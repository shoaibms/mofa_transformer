"""
Imputation Validation Analysis for Metabolomics Data
-------------------------------------------------

This script evaluates the quality of metabolomics data imputation by comparing
original and imputed distributions using multiple statistical approaches:

1. Earth Mover's Distance (Wasserstein): Measures the minimum "work" needed to
   transform one distribution into another
2. Hellinger Distance: Quantifies the similarity between probability distributions
3. Visual comparisons using Q-Q plots, ECDFs, and kernel density estimates

The script generates both numerical metrics and visualization plots to assess
imputation quality comprehensively.

Output:
- CSV file with distance metrics
- Composite figure with three plots:
  * Q-Q plot comparing quantiles
  * Empirical Cumulative Distribution Function (ECDF)
  * Kernel Density Estimation (KDE)
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

def hellinger(p, q):
    """
    Calculate Hellinger distance between two probability distributions.
    
    The Hellinger distance is bounded between 0 (identical distributions) and 
    1 (completely different distributions).
    
    Args:
        p, q (array-like): Input probability distributions
        
    Returns:
        float: Hellinger distance between p and q
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def load_and_prepare_data(original_path, imputed_path):
    """
    Load and prepare original and imputed datasets for comparison.
    
    Args:
        original_path (str): Path to original data CSV
        imputed_path (str): Path to imputed data CSV
        
    Returns:
        tuple: Normalized and flattened original and imputed data
    """
    # Load datasets
    original_data = pd.read_csv(original_path)
    imputed_data = pd.read_csv(imputed_path)
    
    # Extract and process N_Cluster columns
    original_n_cluster = original_data.filter(like='N_Cluster')
    imputed_n_cluster = imputed_data.filter(like='N_Cluster')
    
    # Calculate mean across clusters
    original_flat = original_n_cluster.mean(axis=1)
    imputed_flat = imputed_n_cluster.mean(axis=1)
    
    # Normalize for probability distribution comparison
    original_norm = original_flat / original_flat.sum()
    imputed_norm = imputed_flat / imputed_flat.sum()
    
    return original_flat, imputed_flat, original_norm, imputed_norm

def calculate_distances(original_flat, imputed_flat, original_norm, imputed_norm):
    """
    Calculate distance metrics between original and imputed distributions.
    
    Args:
        original_flat, imputed_flat: Flattened data
        original_norm, imputed_norm: Normalized distributions
        
    Returns:
        tuple: Earth Mover's Distance and Hellinger distance
    """
    emd = wasserstein_distance(original_flat.dropna(), imputed_flat.dropna())
    hellinger_dist = hellinger(original_norm.dropna(), imputed_norm.dropna())
    return emd, hellinger_dist

def create_comparison_plots(original_flat, imputed_flat, cmap_name="BuGn"):
    """
    Create three comparison plots: Q-Q plot, ECDF, and KDE.
    
    Args:
        original_flat, imputed_flat: Data to compare
        cmap_name (str): Name of colormap to use
        
    Returns:
        tuple: Figure and axes objects
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmap = plt.get_cmap(cmap_name)
    
    # Q-Q Plot
    quantiles_original = np.linspace(0, 1, num=len(original_flat.dropna()))
    original_quantiles = np.quantile(original_flat.dropna(), quantiles_original)
    imputed_quantiles = np.quantile(imputed_flat, quantiles_original)
    
    axs[0].scatter(original_quantiles, imputed_quantiles, edgecolor='k', c=cmap(0.6))
    axs[0].plot([original_quantiles.min(), original_quantiles.max()],
                [original_quantiles.min(), original_quantiles.max()], 'r--')
    axs[0].set_xlabel('Quantiles of Original Data (without missing value)')
    axs[0].set_ylabel('Quantiles of Imputed Data')
    axs[0].set_title('Q-Q Plot')
    
    # ECDF Plot
    sns.ecdfplot(original_flat.dropna(), ax=axs[1], label='Original', color=cmap(0.6))
    sns.ecdfplot(imputed_flat.dropna(), ax=axs[1], label='Imputed', color=cmap(0.8))
    axs[1].set_title('Empirical Cumulative Distribution Function')
    axs[1].legend()
    
    # KDE Plot
    sns.kdeplot(original_flat.dropna(), ax=axs[2], label='Original', fill=True, color=cmap(0.6))
    sns.kdeplot(imputed_flat.dropna(), ax=axs[2], label='Imputed', fill=True, color=cmap(0.8))
    axs[2].set_title('Kernel Density Estimate')
    axs[2].legend()
    
    plt.tight_layout()
    return fig, axs

def main():
    # File paths
    original_path = r'C:\Users\ms\Desktop\data_chem\data\old_2\n_column_data_r_deducted.csv'
    imputed_path = r'C:\Users\ms\Desktop\data_chem\imputated\complete imputed files\n_column_data_r_deducted_imputed_rf2.csv'
    output_base_path = r'C:\Users\ms\Desktop\data_chem\imputated\impute_validatiion\n_r_impute_validation_pmm3c'
    
    # Load and prepare data
    original_flat, imputed_flat, original_norm, imputed_norm = load_and_prepare_data(
        original_path, imputed_path
    )
    
    # Calculate distance metrics
    emd, hellinger_dist = calculate_distances(
        original_flat, imputed_flat, original_norm, imputed_norm
    )
    
    # Save distance metrics
    pd.DataFrame({
        'Earth Mover\'s Distance': [emd],
        'Hellinger Distance': [hellinger_dist]
    }).to_csv(f'{output_base_path}.csv', index=False)
    
    # Create and save visualization
    fig, _ = create_comparison_plots(original_flat, imputed_flat)
    fig.savefig(f'{output_base_path}.png')
    plt.show()

if __name__ == "__main__":
    main()