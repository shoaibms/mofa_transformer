"""
Advanced Spectral Data Validation Framework

This module provides comprehensive validation tools for spectral data augmentation. 
It evaluates the quality and fidelity of augmented spectral data compared to original data
through multiple statistical and machine learning approaches including:
1. Distributional divergence measures (Wasserstein distance, Jensen-Shannon divergence)
2. Classification consistency testing
3. Wavelength-specific impact analysis
4. Principal component analysis (PCA)

The validation framework generates detailed reports and visualizations to assess
whether augmented data maintains the essential characteristics of the original data.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Hardcoded colors from colour.py
COLORS = {
    # Core Experimental Variables
    'G1': '#3182bd',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#2ca25f',             # Susceptible Genotype (Medium Teal)
    'T0': '#74c476',             # Control Treatment (Medium Green)
    'T1': '#fe9929',             # Stress Treatment (Muted Orange/Yellow)
    'Leaf': '#006d2c',           # Leaf Tissue (Darkest Green)
    'Root': '#08519c',           # Root Tissue (Darkest Blue)
    'Day1': '#ffffcc',           # Very Light Yellow-Green
    'Day2': '#c2e699',           # Light Yellow-Green
    'Day3': '#78c679',           # Medium Yellow-Green

    # Data Types / Omics / Features
    'Spectral': '#6baed6',       # General Spectral (Medium Blue)
    'Molecular features': '#41ab5d',  # General Molecular features (Medium-Dark Yellow-Green)
    'UnknownFeature': '#969696', # Medium Grey for fallback

    # Specific Spectral Categories
    'Spectral_Water': '#3182bd',     # Medium-Dark Blue
    'Spectral_Pigment': '#238b45',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#08519c',         # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',      # Medium Grey

    # Specific Molecular features Categories
    'Molecular features_PCluster': '#006837', # Darkest Yellow-Green
    'Molecular features_NCluster': '#ffffd4', # Very Light Yellow
    'Molecular features_Other': '#bdbdbd',     # Light Grey

    # Methods & Model Comparison
    'MOFA': '#08519c',            # Dark Blue
    'SHAP': '#006d2c',            # Dark Green
    'Overlap': '#41ab5d',         # Medium-Dark Yellow-Green
    'Transformer': '#6baed6',     # Medium Blue
    'RandomForest': '#74c476',    # Medium Green
    'KNN': '#7fcdbb',             # Medium Teal

    # Network Visualization Elements
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#08519c',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Molecular features': '#41ab5d',   # Default Molecular features Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # Statistical & Difference Indicators
    'Positive_Diff': '#238b45',     # Medium-Dark Green
    'Negative_Diff': '#fe9929',     # Muted Orange/Yellow (Matches T1)
    'Significance': '#08519c',      # Dark Blue (for markers/text)
    'NonSignificant': '#bdbdbd',    # Light Grey
    'Difference_Line': '#636363',   # Dark Grey line

    # Plot Elements & Annotations
    'Background': '#FFFFFF',       # White plot background
    'Panel_Background': '#f7f7f7', # Very Light Gray background for some panels
    'Grid': '#d9d9d9',             # Lighter Gray grid lines
    'Text_Dark': '#252525',        # Darkest Gray / Near Black text
    'Text_Light': '#FFFFFF',       # White text
    'Text_Annotation': '#000000',   # Black text for annotations
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd',# Light Grey border for text boxes
    'Table_Header_BG': '#deebf7',   # Very Light Blue table header
    'Table_Highlight_BG': '#fff7bc',# Pale Yellow for highlighted table cells

    # Temporal Patterns
    'Pattern_Increasing': '#238b45',  # Medium-Dark Green
    'Pattern_Decreasing': '#fe9929',  # Muted Orange/Yellow
    'Pattern_Peak': '#78c679',        # Medium Yellow-Green
    'Pattern_Valley': '#6baed6',      # Medium Blue
    'Pattern_Stable': '#969696',      # Medium Grey
}

def advanced_spectral_validation(original_path, augmented_path, output_dir):
    """
    Perform advanced validation metrics on augmented spectral data.
    
    Parameters:
    -----------
    original_path : str
        Path to the original spectral data CSV file
    augmented_path : str
        Path to the augmented spectral data CSV file
    output_dir : str
        Directory to save validation results
    """
    # Create output directories
    adv_output_dir = os.path.join(output_dir, 'advanced_validation')
    if not os.path.exists(adv_output_dir):
        os.makedirs(adv_output_dir)
    
    plots_dir = os.path.join(adv_output_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    results_dir = os.path.join(adv_output_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load data
    print("Loading data for advanced validation...")
    original_data = pd.read_csv(original_path)
    augmented_data = pd.read_csv(augmented_path)
    
    # Identify wavelength and metadata columns
    wavelength_cols = [
        col for col in original_data.columns if col.startswith('W_')
    ]
    metadata_cols = [
        col for col in original_data.columns if not col.startswith('W_')
    ]
    
    # Extract wavelength values
    wavelengths = np.array(
        [float(col.split('_')[1]) for col in wavelength_cols]
    )
    
    # Separate augmented data by source
    augmented_only = augmented_data[
        ~augmented_data['Row_names'].isin(original_data['Row_names'])
    ]
    
    # Extract spectral data
    original_spectra = original_data[wavelength_cols].values
    augmented_spectra = augmented_only[wavelength_cols].values
    
    # Run validation analyses
    results = {}
    
    # 1. Distributional Divergence Measures
    print("1. Calculating distributional divergence measures...")
    divergence_results = calculate_divergence(
        original_spectra, augmented_spectra, wavelengths, plots_dir, results_dir
    )
    results['divergence'] = divergence_results
    
    # 2. Classification Consistency Test
    print("2. Performing classification consistency test...")
    classification_results = classification_consistency(
        original_data, augmented_data, wavelength_cols, plots_dir, results_dir
    )
    results['classification'] = classification_results
    
    # 3. Wavelength-specific Impact Analysis
    print("3. Analyzing wavelength-specific impacts...")
    impact_results = wavelength_impact_analysis(
        original_spectra, augmented_spectra, wavelengths, plots_dir, results_dir
    )
    results['impact'] = impact_results
    
    # 4. Principal Component Analysis
    print("4. Conducting principal component analysis...")
    pca_results = pca_analysis(
        original_data, augmented_data, wavelength_cols, plots_dir, results_dir
    )
    results['pca'] = pca_results
    
    # Generate summary and return results
    generate_summary(results, adv_output_dir)
    return results


def calculate_divergence(
        original_spectra, augmented_spectra, wavelengths, plots_dir, results_dir
):
    """Calculate distributional divergence measures between original and augmented spectra"""
    # Define key spectral regions
    regions = {
        'Full Spectrum': (350, 2500),
        'Visible': (400, 700),
        'NIR': (700, 1300),
        'SWIR': (1300, 2500)
    }
    
    # Calculate region indices
    region_indices = {}
    for region, (start, end) in regions.items():
        region_indices[region] = np.where(
            (wavelengths >= start) & (wavelengths <= end)
        )[0]
    
    divergence_results = {}
    
    # Calculate overall and region-specific divergence measures
    for region, indices in region_indices.items():
        if len(indices) > 0:
            # Extract region data
            orig_region = original_spectra[:, indices]
            aug_region = augmented_spectra[:, indices]
            
            # Calculate wavelength-specific measures
            wasserstein_distances = []
            js_divergences = []
            
            # For each wavelength in the region
            for i in range(len(indices)):
                orig_values = orig_region[:, i]
                aug_values = aug_region[:, i]
                
                # Wasserstein distance (Earth Mover's Distance)
                wd = wasserstein_distance(orig_values, aug_values)
                wasserstein_distances.append(wd)
                
                # Jensen-Shannon divergence
                # First, create histograms for probability distributions
                bins = 50
                hist_range = (
                    min(np.min(orig_values), np.min(aug_values)),
                    max(np.max(orig_values), np.max(aug_values))
                )
                
                orig_hist, _ = np.histogram(
                    orig_values, bins=bins, range=hist_range, density=True
                )
                aug_hist, _ = np.histogram(
                    aug_values, bins=bins, range=hist_range, density=True
                )
                
                # Add small epsilon to avoid division by zero
                orig_hist = np.maximum(orig_hist, 1e-10)
                aug_hist = np.maximum(aug_hist, 1e-10)
                
                # Normalize
                orig_hist = orig_hist / np.sum(orig_hist)
                aug_hist = aug_hist / np.sum(aug_hist)
                
                # Calculate JS divergence
                js_div = jensenshannon(orig_hist, aug_hist)
                js_divergences.append(js_div)
            
            # Calculate mean values for the region
            mean_wd = np.mean(wasserstein_distances)
            mean_js = np.mean(js_divergences)
            
            # Store results
            divergence_results[f"{region}_wasserstein"] = mean_wd
            divergence_results[f"{region}_jensen_shannon"] = mean_js
            
            # Plot divergence measures across wavelengths
            if region == 'Full Spectrum':
                plt.figure(figsize=(12, 8))
                
                # Plot Wasserstein distance
                plt.subplot(2, 1, 1)
                plt.plot(wavelengths[indices], wasserstein_distances)
                plt.title('Wasserstein Distance by Wavelength')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Wasserstein Distance')
                plt.grid(True, alpha=0.3)
                
                # Plot Jensen-Shannon divergence
                plt.subplot(2, 1, 2)
                plt.plot(wavelengths[indices], js_divergences)
                plt.title('Jensen-Shannon Divergence by Wavelength')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('JS Divergence')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, "divergence_by_wavelength.png"),
                    dpi=300
                )
                plt.close()
    
    # Calculate bootstrap confidence intervals for divergence measures
    print("  Calculating bootstrap confidence intervals...")
    bootstrap_results = bootstrap_divergence(
        original_spectra, augmented_spectra, region_indices
    )
    divergence_results.update(bootstrap_results)
    
    # Save results to CSV
    pd.DataFrame([divergence_results]).to_csv(
        os.path.join(results_dir, "divergence_metrics.csv"), index=False
    )
    
    print("Divergence Results:")
    for key, value in divergence_results.items():
        if not key.endswith('_ci'):
            ci_key = f"{key}_ci"
            ci_value = divergence_results.get(ci_key, "N/A")
            print(f"  {key}: {value:.6f}, 95% CI: {ci_value}")
    
    return divergence_results


def bootstrap_divergence(
        original_spectra, augmented_spectra, region_indices,
        n_bootstrap=1000, confidence=0.95#######################################################1000
):
    """Calculate bootstrap confidence intervals for divergence measures"""
    bootstrap_results = {}
    
    # For each region
    for region, indices in region_indices.items():
        if len(indices) > 0:
            # Extract region data
            orig_region = original_spectra[:, indices]
            aug_region = augmented_spectra[:, indices]
            
            # Bootstrap wasserstein distance
            wd_bootstrap = []
            js_bootstrap = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                orig_bootstrap_idx = np.random.choice(
                    len(orig_region), size=len(orig_region), replace=True
                )
                aug_bootstrap_idx = np.random.choice(
                    len(aug_region), size=len(aug_region), replace=True
                )
                
                orig_bootstrap = orig_region[orig_bootstrap_idx]
                aug_bootstrap = aug_region[aug_bootstrap_idx]
                
                # Calculate mean measures across wavelengths
                wd_values = []
                js_values = []
                
                for i in range(orig_bootstrap.shape[1]):
                    # Wasserstein
                    wd = wasserstein_distance(
                        orig_bootstrap[:, i], aug_bootstrap[:, i]
                    )
                    wd_values.append(wd)
                    
                    # JS divergence
                    bins = 50
                    hist_range = (
                        min(np.min(orig_bootstrap[:, i]),
                            np.min(aug_bootstrap[:, i])),
                        max(np.max(orig_bootstrap[:, i]),
                            np.max(aug_bootstrap[:, i]))
                    )
                    
                    orig_hist, _ = np.histogram(
                        orig_bootstrap[:, i], bins=bins, range=hist_range,
                        density=True
                    )
                    aug_hist, _ = np.histogram(
                        aug_bootstrap[:, i], bins=bins, range=hist_range,
                        density=True
                    )
                    
                    # Add small epsilon to avoid division by zero
                    orig_hist = np.maximum(orig_hist, 1e-10)
                    aug_hist = np.maximum(aug_hist, 1e-10)
                    
                    # Normalize
                    orig_hist = orig_hist / np.sum(orig_hist)
                    aug_hist = aug_hist / np.sum(aug_hist)
                    
                    # Calculate JS divergence
                    js_div = jensenshannon(orig_hist, aug_hist)
                    js_values.append(js_div)
                
                wd_bootstrap.append(np.mean(wd_values))
                js_bootstrap.append(np.mean(js_values))
            
            # Calculate confidence intervals
            alpha = (1 - confidence) / 2
            wd_ci = (
                np.percentile(wd_bootstrap, alpha * 100),
                np.percentile(wd_bootstrap, (1 - alpha) * 100)
            )
            js_ci = (
                np.percentile(js_bootstrap, alpha * 100),
                np.percentile(js_bootstrap, (1 - alpha) * 100)
            )
            
            # Store results
            bootstrap_results[f"{region}_wasserstein_ci"] = (
                f"({wd_ci[0]:.6f}, {wd_ci[1]:.6f})"
            )
            bootstrap_results[f"{region}_jensen_shannon_ci"] = (
                f"({js_ci[0]:.6f}, {js_ci[1]:.6f})"
            )
    
    return bootstrap_results


def classification_consistency(
        original_data, augmented_data, wavelength_cols, plots_dir, results_dir
):
    """Test functional equivalence through classification consistency"""
    # Prepare data
    augmented_only = augmented_data[
        ~augmented_data['Row_names'].isin(original_data['Row_names'])
    ]
    
    # Check if Treatment column exists
    if 'Treatment' not in original_data.columns:
        print(
            "  Warning: 'Treatment' column not found. "
            "Using 'Genotype' for classification."
        )
        target_col = 'Genotype'
    else:
        target_col = 'Treatment'
    
    # Extract features and target
    X_orig = original_data[wavelength_cols].values
    y_orig = original_data[target_col].values
    
    X_aug = augmented_only[wavelength_cols].values
    y_aug = augmented_only[target_col].values
    
    # Result container
    classification_results = {}
    
    # 1. Train on original, test on augmented
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42
    )
    
    rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_orig.fit(X_train, y_train)
    
    # Test on original test set
    y_pred_orig = rf_orig.predict(X_test)
    orig_accuracy = accuracy_score(y_test, y_pred_orig)
    
    # Test on augmented data
    y_pred_aug = rf_orig.predict(X_aug)
    aug_accuracy = accuracy_score(y_aug, y_pred_aug)
    
    classification_results['orig_to_orig_accuracy'] = orig_accuracy
    classification_results['orig_to_aug_accuracy'] = aug_accuracy
    classification_results['accuracy_retention'] = (
        aug_accuracy / orig_accuracy if orig_accuracy > 0 else 0
    )
    
    # 2. Train on augmented, test on original
    X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
        X_aug, y_aug, test_size=0.3, random_state=42
    )
    
    rf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_aug.fit(X_train_aug, y_train_aug)
    
    # Test on augmented test set
    y_pred_aug_test = rf_aug.predict(X_test_aug)
    aug_test_accuracy = accuracy_score(y_test_aug, y_pred_aug_test)
    
    # Test on original data
    y_pred_orig_from_aug = rf_aug.predict(X_orig)
    orig_from_aug_accuracy = accuracy_score(y_orig, y_pred_orig_from_aug)
    
    classification_results['aug_to_aug_accuracy'] = aug_test_accuracy
    classification_results['aug_to_orig_accuracy'] = orig_from_aug_accuracy
    classification_results['reverse_accuracy_retention'] = (
        orig_from_aug_accuracy / aug_test_accuracy if aug_test_accuracy > 0 else 0
    )
    
    # 3. Train on combined, test on both
    X_combined = np.vstack([X_orig, X_aug])
    y_combined = np.concatenate([y_orig, y_aug])
    
    X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
        X_combined, y_combined, test_size=0.3, random_state=42
    )
    
    rf_comb = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_comb.fit(X_train_comb, y_train_comb)
    
    # Test on combined test set
    y_pred_comb = rf_comb.predict(X_test_comb)
    comb_accuracy = accuracy_score(y_test_comb, y_pred_comb)
    
    classification_results['combined_accuracy'] = comb_accuracy
    
    # Save results to CSV
    pd.DataFrame([classification_results]).to_csv(
        os.path.join(results_dir, "classification_results.csv"), index=False
    )
    
    # Plot confusion matrices
    # Original model on augmented data
    cm_orig_aug = confusion_matrix(y_aug, y_pred_aug)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_orig_aug, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: Original to Augmented (Acc: {aug_accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(
        os.path.join(plots_dir, "confusion_orig_to_aug.png"), dpi=300
    )
    plt.close()
    
    # Augmented model on original data
    cm_aug_orig = confusion_matrix(y_orig, y_pred_orig_from_aug)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_aug_orig, annot=True, fmt='d', cmap='Blues')
    plt.title(
        f'Confusion Matrix: Augmented to Original (Acc: {orig_from_aug_accuracy:.4f})'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(
        os.path.join(plots_dir, "confusion_aug_to_orig.png"), dpi=300
    )
    plt.close()
    
    print("Classification Results:")
    print(f"  Original to Original Accuracy: {orig_accuracy:.4f}")
    print(f"  Original to Augmented Accuracy: {aug_accuracy:.4f}")
    print(
        f"  Accuracy Retention: "
        f"{classification_results['accuracy_retention']:.4f}"
    )
    print(f"  Augmented to Augmented Accuracy: {aug_test_accuracy:.4f}")
    print(f"  Augmented to Original Accuracy: {orig_from_aug_accuracy:.4f}")
    print(
        f"  Reverse Accuracy Retention: "
        f"{classification_results['reverse_accuracy_retention']:.4f}"
    )
    print(f"  Combined Data Accuracy: {comb_accuracy:.4f}")
    
    return classification_results


def wavelength_impact_analysis(
        original_spectra, augmented_spectra, wavelengths, plots_dir, results_dir
):
    """Analyze the impact of augmentation at each wavelength"""
    # Calculate absolute differences
    abs_diff = np.abs(
        np.mean(original_spectra, axis=0) - np.mean(augmented_spectra, axis=0)
    )
    rel_diff = abs_diff / np.mean(original_spectra, axis=0)
    
    # Calculate variance ratio
    orig_var = np.var(original_spectra, axis=0)
    aug_var = np.var(augmented_spectra, axis=0)
    var_ratio = aug_var / orig_var
    
    # Calculate impact metrics
    impact_results = {
        'mean_abs_diff': np.mean(abs_diff),
        'max_abs_diff': np.max(abs_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'max_rel_diff': np.max(rel_diff),
        'mean_var_ratio': np.mean(var_ratio),
        'max_var_ratio': np.max(var_ratio)
    }
    
    # Find most affected wavelengths
    top_abs_indices = np.argsort(abs_diff)[-10:]
    top_rel_indices = np.argsort(rel_diff)[-10:]
    
    # Add to results
    impact_results['most_affected_wavelengths_abs'] = ', '.join(
        [str(int(wavelengths[i])) for i in top_abs_indices]
    )
    impact_results['most_affected_wavelengths_rel'] = ', '.join(
        [str(int(wavelengths[i])) for i in top_rel_indices]
    )
    
    # Save results to CSV
    pd.DataFrame([impact_results]).to_csv(
        os.path.join(results_dir, "wavelength_impact.csv"), index=False
    )
    
    # Plot absolute difference
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(wavelengths, abs_diff)
    plt.title('Absolute Difference Between Original and Augmented Spectra')
    plt.ylabel('Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    # Plot relative difference
    plt.subplot(3, 1, 2)
    plt.plot(wavelengths, rel_diff)
    plt.title('Relative Difference Between Original and Augmented Spectra')
    plt.ylabel('Relative Difference')
    plt.grid(True, alpha=0.3)
    
    # Plot variance ratio
    plt.subplot(3, 1, 3)
    plt.plot(wavelengths, var_ratio)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Variance Ratio (Augmented / Original)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Variance Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "wavelength_impact.png"), dpi=300)
    plt.close()
    
    # Plot residuals with confidence interval
    plt.figure(figsize=(12, 6))
    
    # Calculate mean and std of spectra
    orig_mean = np.mean(original_spectra, axis=0)
    aug_mean = np.mean(augmented_spectra, axis=0)
    residuals = orig_mean - aug_mean
    
    # Calculate confidence interval
    orig_std = np.std(original_spectra, axis=0)
    n_orig = original_spectra.shape[0]
    n_aug = augmented_spectra.shape[0]
    
    # Standard error of the difference between means
    se_diff = np.sqrt((orig_std**2 / n_orig) + (orig_std**2 / n_aug))
    
    # 95% confidence interval
    ci_lower = residuals - 1.96 * se_diff
    ci_upper = residuals + 1.96 * se_diff
    
    plt.plot(wavelengths, residuals, 'b-', label='Residuals (Original - Augmented)')
    plt.fill_between(
        wavelengths, ci_lower, ci_upper, color='blue', alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Analysis with 95% Confidence Interval')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residual_analysis.png"), dpi=300)
    plt.close()
    
    print("Wavelength Impact Results:")
    print(f"  Mean Absolute Difference: {impact_results['mean_abs_diff']:.6f}")
    print(f"  Mean Relative Difference: {impact_results['mean_rel_diff']:.6f}")
    print(f"  Mean Variance Ratio: {impact_results['mean_var_ratio']:.6f}")
    print(
        f"  Most Affected Wavelengths (Absolute): "
        f"{impact_results['most_affected_wavelengths_abs']}"
    )
    
    return impact_results


def pca_analysis(
        original_data, augmented_data, wavelength_cols, plots_dir, results_dir
):
    """Perform PCA analysis on original and augmented data"""
    # Extract data
    original_spectra = original_data[wavelength_cols].values
    
    # Extract different categories of augmented data based on Row_names patterns
    categories = {
        'original': original_data,
        'augmented': augmented_data[
            ~augmented_data['Row_names'].isin(original_data['Row_names'])
        ]
    }
    
    # Try to identify augmentation methods
    method_patterns = {
        'gaussian_process': '_GP_',
        'mixup': '_MIX_',
        'warp': '_WARP_',
        'scale': '_SCALE_',
        'noise': '_NOISE_',
        'additive': '_ADD_',
        'multiplicative': '_MULT_'
    }
    
    for method, pattern in method_patterns.items():
        method_data = augmented_data[
            augmented_data['Row_names'].str.contains(pattern, regex=False)
        ]
        if len(method_data) > 0:
            categories[method] = method_data
    
    # Combine all spectra for PCA
    all_spectra = []
    labels = []
    
    for category, data in categories.items():
        spectra = data[wavelength_cols].values
        # Limit to 500 samples per category to prevent overcrowding
        if len(spectra) > 500:
            indices = np.random.choice(len(spectra), 500, replace=False)
            spectra = spectra[indices]
        
        all_spectra.append(spectra)
        labels.extend([category] * len(spectra))
    
    all_spectra = np.vstack(all_spectra)
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(all_spectra)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'PC3': pca_result[:, 2],
        'Category': labels
    })
    
    # Calculate metrics
    pca_metrics = {}
    
    # Calculate average distance from original centroid
    original_mask = pca_df['Category'] == 'original'
    original_centroid = pca_df.loc[
        original_mask, ['PC1', 'PC2', 'PC3']
    ].mean().values
    
    for category in pca_df['Category'].unique():
        if category != 'original':
            category_mask = pca_df['Category'] == category
            category_points = pca_df.loc[
                category_mask, ['PC1', 'PC2', 'PC3']
            ].values
            
            # Calculate Euclidean distances to original centroid
            distances = np.sqrt(
                np.sum((category_points - original_centroid)**2, axis=1)
            )
            
            # Store metrics
            pca_metrics[f'{category}_mean_distance'] = np.mean(distances)
            pca_metrics[f'{category}_std_distance'] = np.std(distances)
    
    # Save metrics to CSV
    pd.DataFrame([pca_metrics]).to_csv(
        os.path.join(results_dir, "pca_metrics.csv"), index=False
    )
    
    # Plot PCA results
    # 2D plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {
        'original': 'blue', 'augmented': 'red',
        'gaussian_process': 'green', 'mixup': 'purple', 'warp': 'orange',
        'scale': 'brown', 'noise': 'pink', 'additive': 'cyan',
        'multiplicative': 'magenta'
    }
    
    for category in pca_df['Category'].unique():
        mask = pca_df['Category'] == category
        plt.scatter(
            pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
            c=colors.get(category, 'gray'), label=category, alpha=0.7
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA of Original and Augmented Spectra')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_visualization.png"), dpi=300)
    plt.close()
    
    # 3D plot if available
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for category in pca_df['Category'].unique():
            mask = pca_df['Category'] == category
            ax.scatter(
                pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                pca_df.loc[mask, 'PC3'],
                c=colors.get(category, 'gray'), label=category, alpha=0.7
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
        ax.set_title('3D PCA of Original and Augmented Spectra')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pca_3d_visualization.png"), dpi=300)
        plt.close()
    except ImportError: # Catch specific ImportError for mplot3d
        print("  Warning: 3D plotting not available (mpl_toolkits.mplot3d "
              "not found). Skipping 3D PCA plot.")
    except Exception as e: # Catch other potential errors during 3D plotting
        print(f"  Warning: 3D plotting failed ({e}). Skipping 3D PCA plot.")

    
    print("PCA Analysis Results:")
    for key, value in pca_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    return pca_metrics


def generate_summary(results, output_dir):
    """Generate a summary of advanced validation results"""
    summary_path = os.path.join(output_dir, "advanced_validation_summary.html")
    
    # Use UTF-8 encoding explicitly when opening the file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>Advanced Spectral Validation Summary</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        .good {{ color: green; }}
        .moderate {{ color: orange; }}
        .poor {{ color: red; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>Advanced Spectral Validation Summary</h1>
    <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>Overview</h2>
        <p>This report presents advanced statistical validation of the augmented spectral data.</p>
    </div>
''')
        
        # Distributional Divergence section
        if 'divergence' in results:
            f.write('''
    <h2>Distributional Divergence Measures</h2>
    <p>These metrics quantify the statistical difference between original and augmented data distributions.</p>
    <table>
        <tr>
            <th>Region</th>
            <th>Wasserstein Distance</th>
            <th>Jensen-Shannon Divergence</th>
            <th>Interpretation</th>
        </tr>
''')
            
            divergence_results = results['divergence']
            for region in ['Full Spectrum', 'Visible', 'NIR', 'SWIR']:
                wd_key = f"{region}_wasserstein"
                js_key = f"{region}_jensen_shannon"
                wd_ci_key = f"{region}_wasserstein_ci"
                js_ci_key = f"{region}_jensen_shannon_ci"
                
                if wd_key in divergence_results and js_key in divergence_results:
                    wd_value = divergence_results[wd_key]
                    js_value = divergence_results[js_key]
                    wd_ci = divergence_results.get(wd_ci_key, "N/A")
                    js_ci = divergence_results.get(js_ci_key, "N/A")
                    
                    # Determine quality class
                    quality_class = ""
                    interpretation = ""
                    if js_value < 0.1:
                        quality_class = "good"
                        interpretation = "Excellent similarity between distributions"
                    elif js_value < 0.2:
                        quality_class = "moderate"
                        interpretation = "Good similarity between distributions"
                    else:
                        quality_class = "poor"
                        interpretation = "Moderate similarity between distributions"
                    
                    f.write(f'''
        <tr>
            <td>{region}</td>
            <td class="{quality_class}">{wd_value:.6f} <br><small>CI: {wd_ci}</small></td>
            <td class="{quality_class}">{js_value:.6f} <br><small>CI: {js_ci}</small></td>
            <td>{interpretation}</td>
        </tr>
''')
            
            f.write("</table>")
            
            # Add divergence plots
            f.write('''
    <h3>Divergence by Wavelength</h3>
    <p>These plots show how divergence measures vary across the spectral range:</p>
    <div>
        <img src="plots/divergence_by_wavelength.png" alt="Divergence by Wavelength">
    </div>
''')
        
        # Classification Consistency section
        if 'classification' in results:
            f.write('''
    <h2>Classification Consistency</h2>
    <p>This analysis tests the functional equivalence of original and augmented data for machine learning purposes.</p>
    <table>
        <tr>
            <th>Test</th>
            <th>Accuracy</th>
            <th>Interpretation</th>
        </tr>
''')
            
            classification_results = results['classification']
            
            # Original to original
            orig_to_orig = classification_results.get('orig_to_orig_accuracy', 0)
            quality_class = ("good" if orig_to_orig > 0.8 else
                             "moderate" if orig_to_orig > 0.6 else "poor")
            f.write(f'''
        <tr>
            <td>Original to Original</td>
            <td class="{quality_class}">{orig_to_orig:.4f}</td>
            <td>Baseline accuracy on original data</td>
        </tr>
''')
            
            # Original to augmented
            orig_to_aug = classification_results.get('orig_to_aug_accuracy', 0)
            retention = classification_results.get('accuracy_retention', 0)
            quality_class = ("good" if retention > 0.9 else
                             "moderate" if retention > 0.7 else "poor")
            f.write(f'''
        <tr>
            <td>Original to Augmented</td>
            <td class="{quality_class}">{orig_to_aug:.4f} <br><small>({retention:.2f}x retention)</small></td>
            <td>How well models trained on original data perform on augmented data</td>
        </tr>
''')
            
            # Augmented to augmented
            aug_to_aug = classification_results.get('aug_to_aug_accuracy', 0)
            quality_class = ("good" if aug_to_aug > 0.8 else
                             "moderate" if aug_to_aug > 0.6 else "poor")
            f.write(f'''
        <tr>
            <td>Augmented to Augmented</td>
            <td class="{quality_class}">{aug_to_aug:.4f}</td>
            <td>Baseline accuracy on augmented data</td>
        </tr>
''')
            
            # Augmented to original
            aug_to_orig = classification_results.get('aug_to_orig_accuracy', 0)
            reverse_retention = classification_results.get(
                'reverse_accuracy_retention', 0
            )
            quality_class = ("good" if reverse_retention > 0.9 else
                             "moderate" if reverse_retention > 0.7 else "poor")
            f.write(f'''
        <tr>
            <td>Augmented to Original</td>
            <td class="{quality_class}">{aug_to_orig:.4f} <br><small>({reverse_retention:.2f}x retention)</small></td>
            <td>How well models trained on augmented data perform on original data</td>
        </tr>
''')
            
            # Combined
            combined = classification_results.get('combined_accuracy', 0)
            quality_class = ("good" if combined > 0.8 else
                             "moderate" if combined > 0.6 else "poor")
            f.write(f'''
        <tr>
            <td>Combined Training</td>
            <td class="{quality_class}">{combined:.4f}</td>
            <td>Accuracy when training on combined original and augmented data</td>
        </tr>
''')
            
            f.write("</table>")
            
            # Add confusion matrix images
            f.write('''
    <h3>Confusion Matrices</h3>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 50%; padding: 10px;">
            <img src="plots/confusion_orig_to_aug.png" alt="Original to Augmented Confusion Matrix">
        </div>
        <div style="flex: 50%; padding: 10px;">
            <img src="plots/confusion_aug_to_orig.png" alt="Augmented to Original Confusion Matrix">
        </div>
    </div>
''')
        
        # Wavelength Impact section
        if 'impact' in results:
            f.write('''
    <h2>Wavelength-specific Impact Analysis</h2>
    <p>This analysis shows how the augmentation process affects different parts of the spectrum.</p>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Interpretation</th>
        </tr>
''')
            
            impact_results = results['impact']
            
            # Mean absolute difference
            mean_abs_diff = impact_results.get('mean_abs_diff', 0)
            quality_class = ("good" if mean_abs_diff < 0.01 else
                             "moderate" if mean_abs_diff < 0.05 else "poor")
            f.write(f'''
        <tr>
            <td>Mean Absolute Difference</td>
            <td class="{quality_class}">{mean_abs_diff:.6f}</td>
            <td>Average absolute difference across all wavelengths</td>
        </tr>
''')
            
            # Mean relative difference
            mean_rel_diff = impact_results.get('mean_rel_diff', 0)
            quality_class = ("good" if mean_rel_diff < 0.05 else
                             "moderate" if mean_rel_diff < 0.1 else "poor")
            f.write(f'''
        <tr>
            <td>Mean Relative Difference</td>
            <td class="{quality_class}">{mean_rel_diff:.6f}</td>
            <td>Average relative difference across all wavelengths</td>
        </tr>
''')
            
            # Variance ratio
            mean_var_ratio = impact_results.get('mean_var_ratio', 0)
            quality_class = ("good" if 0.8 < mean_var_ratio < 1.2 else
                             "moderate" if 0.5 < mean_var_ratio < 1.5 else "poor")
            f.write(f'''
        <tr>
            <td>Mean Variance Ratio</td>
            <td class="{quality_class}">{mean_var_ratio:.6f}</td>
            <td>Ratio of variances between augmented and original data (ideal: ~1.0)</td>
        </tr>
''')
            
            # Most affected wavelengths
            most_affected_abs = impact_results.get(
                'most_affected_wavelengths_abs', ''
            )
            f.write(f'''
        <tr>
            <td>Most Affected Wavelengths</td>
            <td>{most_affected_abs}</td>
            <td>Wavelengths with the largest absolute differences</td>
        </tr>
''')
            
            f.write("</table>")
            
            # Add impact plots
            f.write('''
    <h3>Wavelength Impact Visualization</h3>
    <div>
        <img src="plots/wavelength_impact.png" alt="Wavelength Impact Analysis">
    </div>
    
    <h3>Residual Analysis</h3>
    <div>
        <img src="plots/residual_analysis.png" alt="Residual Analysis">
    </div>
''')
        
        # PCA Analysis section
        if 'pca' in results:
            f.write('''
    <h2>Principal Component Analysis</h2>
    <p>This visualization shows how augmented data compares to original data in reduced dimensionality space.</p>
    <div>
        <img src="plots/pca_visualization.png" alt="PCA Visualization">
    </div>
''')
            
            # 3D PCA if available
            if os.path.exists(
                os.path.join(output_dir, "plots", "pca_3d_visualization.png")
            ):
                f.write('''
    <h3>3D PCA Visualization</h3>
    <div>
        <img src="plots/pca_3d_visualization.png" alt="3D PCA Visualization">
    </div>
''')
            
            # PCA distance metrics
            f.write('''
    <h3>PCA Distance Metrics</h3>
    <p>These metrics quantify how far augmented data points are from the original data centroid in PCA space.</p>
    <table>
        <tr>
            <th>Category</th>
            <th>Mean Distance</th>
            <th>Standard Deviation</th>
        </tr>
''')
            
            pca_results = results['pca']
            sorted_categories = sorted([
                k.replace('_mean_distance', '') for k in pca_results.keys()
                if k.endswith('_mean_distance')
            ])
            for category in sorted_categories:
                mean_key = f"{category}_mean_distance"
                std_key = f"{category}_std_distance"
                
                if mean_key in pca_results and std_key in pca_results:
                    mean_distance = pca_results[mean_key]
                    std_distance = pca_results[std_key]
                    
                    # Determine quality class based on distance
                    quality_class = ("good" if mean_distance < 10 else
                                     "moderate" if mean_distance < 20 else "poor")
                    
                    f.write(f'''
        <tr>
            <td>{category}</td>
            <td class="{quality_class}">{mean_distance:.4f}</td>
            <td>{std_distance:.4f}</td>
        </tr>
''')
            
            f.write("</table>")
        
        # Overall assessment
        f.write('''
    <h2>Overall Assessment</h2>
    <p>Based on the advanced validation metrics, the augmented spectral data demonstrates:</p>
    <ul>
''')
        
        # Generate conclusions based on available results
        if 'divergence' in results:
            avg_js = np.mean(
                [v for k, v in results['divergence'].items()
                 if k.endswith('_jensen_shannon')]
            )
            if avg_js < 0.1:
                f.write("<li><strong>Excellent statistical similarity</strong> "
                        "to the original data distribution</li>")
            elif avg_js < 0.2:
                f.write("<li><strong>Good statistical similarity</strong> "
                        "to the original data distribution</li>")
            else:
                f.write("<li><strong>Acceptable statistical similarity</strong> "
                        "to the original data distribution</li>")
        
        if 'classification' in results:
            avg_retention = np.mean([
                results['classification'].get('accuracy_retention', 0),
                results['classification'].get('reverse_accuracy_retention', 0)
            ])
            if avg_retention > 0.9:
                f.write("<li><strong>Excellent functional equivalence</strong> "
                        "for machine learning applications</li>")
            elif avg_retention > 0.7:
                f.write("<li><strong>Good functional equivalence</strong> "
                        "for machine learning applications</li>")
            else:
                f.write("<li><strong>Acceptable functional equivalence</strong> "
                        "for machine learning applications</li>")
        
        if 'impact' in results:
            mean_rel_diff = results['impact'].get('mean_rel_diff', 0)
            mean_var_ratio = results['impact'].get('mean_var_ratio', 0)
            
            if mean_rel_diff < 0.05 and 0.8 < mean_var_ratio < 1.2:
                f.write("<li><strong>Excellent preservation</strong> of spectral "
                        "characteristics across all wavelengths</li>")
            elif mean_rel_diff < 0.1 and 0.5 < mean_var_ratio < 1.5:
                f.write("<li><strong>Good preservation</strong> of spectral "
                        "characteristics across all wavelengths</li>")
            else:
                f.write("<li><strong>Acceptable preservation</strong> of spectral "
                        "characteristics across all wavelengths</li>")
        
        # Final recommendation
        f.write('''
    </ul>
    <h3>Publication Recommendation</h3>
''')
        
        # Calculate overall score based on available metrics
        scores = []
        
        if 'divergence' in results:
            # Lower is better for divergence, so convert to a 0-1 scale where 1 is perfect
            avg_js = np.mean(
                [v for k, v in results['divergence'].items()
                 if k.endswith('_jensen_shannon')]
            )
            scores.append(max(0, 1 - (avg_js * 5)))  # Scale JS (0-0.2) to 0-1
        
        if 'classification' in results:
            avg_retention = np.mean([
                results['classification'].get('accuracy_retention', 0),
                results['classification'].get('reverse_accuracy_retention', 0)
            ])
            scores.append(avg_retention)
        
        if 'impact' in results:
            mean_rel_diff = results['impact'].get('mean_rel_diff', 0)
            # Scale relative diff (0-0.1) to 0-1
            impact_score = max(0, 1 - (mean_rel_diff * 10))
            scores.append(impact_score)
            
            # Variance ratio score (1.0 is ideal)
            mean_var_ratio = results['impact'].get('mean_var_ratio', 0)
            var_ratio_score = max(0, 1 - abs(mean_var_ratio - 1))
            scores.append(var_ratio_score)
        
        if scores:
            overall_score = np.mean(scores)
            
            if overall_score > 0.9:
                recommendation = '''
        <p class="good">The augmented data is of <strong>exceptional quality</strong> and meets the highest standards for publication in high-impact journals. The statistical properties and physical characteristics of the original spectra are preserved with remarkable fidelity, making the augmented dataset a valuable resource for advancing plant stress response research.</p>
'''
            elif overall_score > 0.8:
                recommendation = '''
        <p class="good">The augmented data is of <strong>high quality</strong> and suitable for publication in high-impact journals. The statistical and physical properties of the original data are well-preserved, providing a reliable foundation for robust analysis and modeling of plant stress responses.</p>
'''
            elif overall_score > 0.7:
                recommendation = '''
        <p class="moderate">The augmented data is of <strong>good quality</strong> and suitable for most research purposes, including publication. While some minor deviations from the original data exist, they are unlikely to significantly impact analysis results or scientific conclusions.</p>
'''
            else:
                recommendation = '''
        <p class="poor">The augmented data shows <strong>acceptable quality</strong> but may benefit from refinement before publication in high-impact journals. Consider adjusting the augmentation parameters to better preserve key spectral characteristics.</p>
'''
            
            f.write(recommendation)
            f.write(f"<p><strong>Overall validation score: {overall_score:.4f}</strong></p>")
        
        f.write('''
    <hr>
    <p><em>Report generated by Advanced Spectral Validator</em></p>
</body>
</html>
''')
    
    print(f"\nAdvanced validation summary saved to: {summary_path}")


if __name__ == "__main__":
    # File paths
    original_path = r"C:\\Users\\ms\\Desktop\\hyper\\data\\hyper_full_w.csv"
    augmented_path = r"C:\\Users\\ms\\Desktop\\hyper\\output\\augment\\augmented_spectral_data.csv"
    output_dir = r"C:\\Users\\ms\\Desktop\\hyper\\output\\augment\\hyper"
    
    # Run advanced validation
    start_time = time.time()
    results = advanced_spectral_validation(original_path, augmented_path, output_dir)
    end_time = time.time()
    
    print(f"\nAdvanced validation completed in {end_time - start_time:.2f} seconds")