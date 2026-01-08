#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated MOFA+ Visualization Script for Figure 1

This script creates a comprehensive Figure 1 with 6 panels (a-f) by integrating
the variance explained plot, factor-condition correlation heatmap, factor scatter plots,
and factor distribution box plots. It dynamically identifies key factors based on
correlation results.

Usage:
    python Figure1_Integrated.py /path/to/mofa_model.hdf5 /path/to/metadata.csv /path/to/correlations.csv /path/to/output_dir
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, Patch
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
from scipy.stats import spearmanr, mannwhitneyu
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
import traceback
from datetime import datetime
from matplotlib.lines import Line2D

# --- Plotting Parameter Setup ---
# Set up plotting parameters for publication quality
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22

# --- Color Definitions ---
VIEW_COLORS = {
    'leaf_spectral': '#c2e699',
    'root_spectral': '#636363',
    'leaf_molecular_feature': '#006837',
    'root_molecular_feature': '#ffffd4',
    'leaf_metabolite': '#006837',
    'root_metabolite': '#ffffd4'
}
GENOTYPE_COLORS = {
    'G1': '#00FA9A', 'G2': '#48D1CC',
    '1': '#00FA9A', '2': '#48D1CC'
}
TREATMENT_COLORS = {
    'T0': '#4682B4', 'T1': '#BDB76B',
    '0': '#4682B4', '1': '#BDB76B'
}
TISSUE_COLORS = {'L': '#00FF00', 'R': '#40E0D0'}
TIMEPOINT_COLORS = {1: "#d0d05a", 2: "#2BA9E8", 3: "#116537"}

# --- Helper Functions ---

def safe_decode(byte_string):
    """Safely decodes byte strings, returns original if not bytes."""
    if isinstance(byte_string, bytes):
        try:
            return byte_string.decode('utf-8')
        except UnicodeDecodeError:
            return byte_string.decode('latin-1', errors='replace')
    return byte_string

# --- Data Loading ---

def load_data(mofa_file, metadata_file):
    """Load data from MOFA+ HDF5 file and metadata CSV"""
    print(f"Loading data from {mofa_file} and {metadata_file}")
    results = {}
    metadata = None

    # --- Metadata Loading ---
    if not os.path.exists(metadata_file):
        print(f"FATAL ERROR: Metadata file {metadata_file} does not exist!")
        return None
    try:
        metadata = pd.read_csv(metadata_file)
        print(f"Metadata loaded: {len(metadata)} rows, {len(metadata.columns)} columns")
        
        required_meta_cols = ['Genotype', 'Treatment', 'Time Point', 'Tissue.type', 'Batch']

        if 'Time Point' not in metadata.columns and 'Day' in metadata.columns:
            print("INFO: Renaming metadata column 'Day' to 'Time Point' for consistency.")
            metadata.rename(columns={'Day': 'Time Point'}, inplace=True)
            if 'Day' in required_meta_cols:
                 required_meta_cols.remove('Day')
            if 'Time Point' not in required_meta_cols:
                 required_meta_cols.append('Time Point')

        missing_cols = [col for col in required_meta_cols if col not in metadata.columns]
        if missing_cols:
            print(f"WARNING: Metadata missing required columns: {missing_cols}")
        else:
            print("All essential metadata columns found.")
            for col in required_meta_cols:
                unique_values = metadata[col].unique()
                print(f"  Column '{col}' unique values ({len(unique_values)}): {unique_values}")
        results['metadata'] = metadata
    except Exception as e:
        print(f"FATAL ERROR reading metadata file: {e}")
        traceback.print_exc()
        return None

    # --- MOFA+ Model Loading ---
    if not os.path.exists(mofa_file):
        print(f"FATAL ERROR: MOFA+ model file {mofa_file} not found")
        return None
    try:
        with h5py.File(mofa_file, 'r') as f:
            # Extract views
            if 'views' in f and 'views' in f['views']:
                views_data = f['views']['views'][()]
                views = [safe_decode(v) for v in views_data]
                view_map = {
                    'leaf_spectral': 'leaf_spectral',
                    'root_spectral': 'root_spectral',
                    'leaf_metabolite': 'leaf_molecular_feature',
                    'root_metabolite': 'root_molecular_feature'
                }
                results['views'] = [view_map.get(v, v) for v in views]
                print(f"Found and mapped views: {results['views']}")
            else:
                print("ERROR: 'views/views' dataset not found in HDF5 file.")
                results['views'] = []

            # Extract samples
            if 'samples' in f and 'group0' in f['samples']:
                sample_data = f['samples']['group0'][()]
                results['samples'] = {'group0': [safe_decode(s) for s in sample_data]}
                num_samples = len(results['samples']['group0'])
                print(f"Loaded {num_samples} sample names for group 'group0'")
                if metadata is not None and len(metadata) != num_samples:
                    print(f"FATAL ERROR: Metadata row count ({len(metadata)}) does not match MOFA sample count ({num_samples})!")
                    return None
            else:
                print("ERROR: 'samples/group0' not found in HDF5 file.")
                return None

            # Extract factors (Z)
            if 'expectations' in f and 'Z' in f['expectations'] and 'group0' in f['expectations']['Z']:
                z_data = f['expectations']['Z']['group0'][()]
                results['factors'] = z_data.T
                print(f"Loaded factors (Z): shape {results['factors'].shape} (samples x factors)")
                if results['factors'].shape[0] != num_samples:
                    print(f"FATAL ERROR: Factor matrix row count ({results['factors'].shape[0]}) does not match sample count ({num_samples})!")
                    return None
            else:
                print("ERROR: 'expectations/Z/group0' not found.")
                results['factors'] = None

            # Extract weights (W)
            if 'expectations' in f and 'W' in f['expectations']:
                weights = {}
                expected_factors = results['factors'].shape[1] if results.get('factors') is not None else None
                for view_key in f['expectations']['W']:
                    view_name = safe_decode(view_key)
                    mapped_view_name = view_map.get(view_name, view_name)
                    w_data = f['expectations']['W'][view_key][()]
                    weights[mapped_view_name] = w_data
                    if expected_factors is not None and w_data.shape[0] != expected_factors:
                        print(f"WARNING: Weight matrix for view '{mapped_view_name}' (orig: {view_name}) has {w_data.shape[0]} factors, expected {expected_factors}.")
                results['weights'] = weights
                print(f"Loaded weights (W) for {len(weights)} views (using mapped names).")
            else:
                print("ERROR: 'expectations/W' not found.")
                results['weights'] = {}

            # Extract feature names
            if 'features' in f:
                features = {}
                for view_key in f['features']:
                    view_name = safe_decode(view_key)
                    mapped_view_name = view_map.get(view_name, view_name)
                    try:
                        feature_data = f['features'][view_key][()]
                        cleaned_features = [safe_decode(feat).replace('P_Cluster_', 'P_').replace('N_Cluster_', 'N_') for feat in feature_data]
                        features[mapped_view_name] = cleaned_features
                        if mapped_view_name in results['weights'] and len(features[mapped_view_name]) != results['weights'][mapped_view_name].shape[1]:
                            print(f"WARNING: Feature count for view '{mapped_view_name}' ({len(features[mapped_view_name])}) doesn't match weight matrix dimension ({results['weights'][mapped_view_name].shape[1]}).")
                    except Exception as e:
                        print(f"Error extracting features for {mapped_view_name} (orig: {view_name}): {e}")
                results['features'] = features
                print(f"Loaded and cleaned feature names for {len(features)} views (using mapped names).")
            else:
                 print("ERROR: 'features' group not found.")
                 results['features'] = {}

            # Extract variance explained
            variance_dict = {}
            if 'variance_explained' in f:
                r2_total_path = 'variance_explained/r2_total/group0'
                if r2_total_path in f:
                    try:
                        r2_total_data = f[r2_total_path][()]
                        if len(r2_total_data) == len(results.get('views', [])):
                            variance_dict['r2_total_per_view'] = r2_total_data
                            print(f"Loaded r2_total_per_view: {r2_total_data}")
                        else:
                            print(f"Warning: r2_total shape {r2_total_data.shape} mismatch with view count {len(results.get('views', []))}. Storing raw.")
                            variance_dict['r2_total_raw'] = r2_total_data
                    except Exception as e:
                        print(f"Error extracting r2_total from {r2_total_path}: {e}")
                else:
                     print(f"Dataset {r2_total_path} not found.")

                r2_pf_path = 'variance_explained/r2_per_factor/group0'
                if r2_pf_path in f:
                    try:
                        r2_per_factor_data = f[r2_pf_path][()]
                        n_views = len(results.get('views', []))
                        n_factors_expected = results.get('factors', np.array([])).shape[1]

                        if r2_per_factor_data.shape == (n_views, n_factors_expected):
                            variance_dict['r2_per_factor'] = r2_per_factor_data
                            print(f"Loaded r2_per_factor: shape {r2_per_factor_data.shape} (views x factors)")
                        else:
                            print(f"WARNING: r2_per_factor shape {r2_per_factor_data.shape} mismatch with expected ({n_views}, {n_factors_expected}). Check model output.")
                            if r2_per_factor_data.shape == (n_factors_expected, n_views):
                                variance_dict['r2_per_factor'] = r2_per_factor_data.T
                                print(f"  -> Transposed to expected shape ({n_views}, {n_factors_expected})")
                            else:
                                variance_dict['r2_per_factor_raw'] = r2_per_factor_data
                                print(f"  -> Stored raw r2_per_factor data.")

                    except Exception as e:
                        print(f"Error extracting r2_per_factor from {r2_pf_path}: {e}")
                        traceback.print_exc()
                else:
                    print(f"Dataset {r2_pf_path} not found.")

            results['variance'] = variance_dict

        # --- Post-Loading Processing ---

        if results.get('factors') is not None and metadata is not None:
            factors = results['factors']
            factor_cols = [f"Factor{i+1}" for i in range(factors.shape[1])]
            factors_df = pd.DataFrame(factors, columns=factor_cols)

            if 'Time Point' not in metadata.columns and 'Day' in metadata.columns:
                 print("WARNING: 'Day' column found but not renamed to 'Time Point' earlier. Attempting rename now.")
                 metadata.rename(columns={'Day': 'Time Point'}, inplace=True)

            meta_cols_to_add = [col for col in metadata.columns if col not in factors_df.columns]
            for col in meta_cols_to_add:
                try:
                    factors_df[col] = metadata[col].values
                except ValueError as ve:
                    print(f"ERROR adding metadata column '{col}': {ve}. Length mismatch likely.")
                    print(f"  Factor df length: {len(factors_df)}, Metadata column length: {len(metadata[col])}")
                    return None

            results['factors_df'] = factors_df
            print(f"Created combined factors + metadata DataFrame: {factors_df.shape}")
            print(f"  factors_df columns: {factors_df.columns.tolist()}")

        if results.get('weights') and results.get('features'):
            feature_importance = {}
            weights = results['weights']
            features = results['features']

            for view_name, view_weights in weights.items():
                if view_name not in features:
                    print(f"Warning: No features found for view '{view_name}', cannot calculate importance.")
                    continue

                view_features = features[view_name]

                if len(view_features) != view_weights.shape[1]:
                    print(f"ERROR: Feature count ({len(view_features)}) != weight dim ({view_weights.shape[1]}) for view '{view_name}'. Skipping importance.")
                    continue

                importance = np.sum(np.abs(view_weights), axis=0)

                importance_df = pd.DataFrame({
                    'Feature': view_features,
                    'Importance': importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                feature_importance[view_name] = importance_df

            results['feature_importance'] = feature_importance
            print(f"Calculated feature importance for {len(feature_importance)} views.")

    except Exception as e:
        print(f"FATAL ERROR during HDF5 loading or processing: {e}")
        traceback.print_exc()
        return None

    print("-" * 30 + " Data Loading Summary " + "-" * 30)
    print(f"Views loaded: {results.get('views')}")
    print(f"Samples loaded: {len(results.get('samples', {}).get('group0', []))}")
    print(f"Factors loaded shape: {results.get('factors', np.array([])).shape}")
    print(f"Weights loaded for views: {list(results.get('weights', {}).keys())}")
    print(f"Features loaded for views: {list(results.get('features', {}).keys())}")
    print(f"Variance data keys: {list(results.get('variance', {}).keys())}")
    print(f"Factors + Metadata DF shape: {results.get('factors_df', pd.DataFrame()).shape}")
    print(f"Feature Importance calculated for views: {list(results.get('feature_importance', {}).keys())}")
    print("-" * 80)

    essential = ['views', 'samples', 'factors', 'weights', 'features', 'metadata', 'factors_df']
    missing_essential = []
    for item in essential:
        if item not in results or results[item] is None:
            missing_essential.append(item)
            continue

        value = results[item]
        is_empty = False
        if isinstance(value, (dict, list)):
            if not value:
                is_empty = True
        elif isinstance(value, pd.DataFrame):
            if value.empty:
                is_empty = True
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                is_empty = True

        if is_empty:
            missing_essential.append(f"{item} (empty)")

    if missing_essential:
         print(f"FATAL ERROR: Missing or empty essential data components after loading: {missing_essential}")
         return None

    print("SUCCESS: All essential data components loaded successfully.")
    return results

# --- Plotting Helpers ---

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Create a plot of the covariance confidence ellipse of *x* and *y*."""
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if x.size < 2 :
        return None, (np.mean(x), np.mean(y))

    cov = np.cov(x, y)
    if np.isscalar(cov):
        return None, (np.mean(x), np.mean(y))
    if cov[0, 0] < 1e-9 or cov[1, 1] < 1e-9:
        return None, (np.mean(x), np.mean(y))

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = np.clip(pearson, -1.0, 1.0)

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
         return None, (np.mean(x), np.mean(y))

    if np.any(eigenvalues <= 1e-9):
        return None, (np.mean(x), np.mean(y))

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale_x = np.sqrt(eigenvalues[0]) * n_std
    scale_y = np.sqrt(eigenvalues[1]) * n_std

    mean_x, mean_y = np.mean(x), np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse, (mean_x, mean_y)

def verify_variance_explained(r2_per_factor, views):
    """Verify variance explained calculations are correct"""
    print("\n---- VARIANCE EXPLAINED VERIFICATION ----")
    
    print(f"r2_per_factor shape: {r2_per_factor.shape}")
    expected_shape = (len(views), r2_per_factor.shape[1])
    print(f"Expected shape (views × factors): {expected_shape}")
    
    df_var = pd.DataFrame(r2_per_factor, index=views, columns=[f"Factor{i+1}" for i in range(r2_per_factor.shape[1])])
    print("\nVariance per view (%): ")
    for view in views:
        row_sum = df_var.loc[view].sum()
        print(f"  {view}: sum = {row_sum:.2f}%")
    
    total_per_factor = df_var.sum(axis=0)
    print("\nTotal variance per factor (%):")
    for factor, value in total_per_factor.items():
        print(f"  {factor}: {value:.2f}%")
    
    if np.any(r2_per_factor < 0):
        print("WARNING: Negative variance detected!")
    if np.any(r2_per_factor > 100):
        print("WARNING: Variance values > 100% detected!")
        
    print("------------------------------------------------")
    return df_var

def verify_treatment_factor_plot(merged_df, factor_name):
    """Verify treatment factor distribution calculations"""
    print("\n---- TREATMENT FACTOR VERIFICATION ----")
    
    print(f"Selected treatment/batch factor: {factor_name}")
    
    if factor_name not in merged_df.columns:
        print(f"ERROR: Factor '{factor_name}' not found in dataframe!")
        return
    
    factor_stats = merged_df[factor_name].describe()
    print(f"\nFactor statistics: \n{factor_stats}")
    
    print("\nGroup statistics (should match boxplot):")
    for treat in sorted(merged_df['Treatment'].unique()):
        treat_label = 'T0' if treat == '0' else 'T1' if treat == '1' else treat
        for geno in sorted(merged_df['Genotype'].unique()):
            if geno not in ['G1', 'G2', '1', '2']: continue
            mask = (merged_df['Genotype'] == geno) & (merged_df['Treatment'] == treat)
            count = sum(mask)
            if count > 0:
                values = merged_df.loc[mask, factor_name]
                q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                print(f"  {treat_label}, {geno}: n={count}, median={median:.3f}, Q1={q1:.3f}, Q3={q3:.3f}")
    
    if 'Treatment' in merged_df.columns:
        treat_numeric = pd.to_numeric(merged_df['Treatment'], errors='coerce')
        correlation = np.corrcoef(merged_df[factor_name], treat_numeric)[0, 1]
        print(f"\nCorrelation with Treatment: {correlation:.3f}")
        
    print("------------------------------------------------")

# --- Main Visualization Function ---

def create_integrated_figure1(data, correlation_file, output_dir, export_data=False):
    """Create a comprehensive Figure 1 with 6 panels a-f using dynamically identified factors."""

    print("\n--- Creating Integrated Figure 1 (Dynamically Identified Factors) ---")

    # --- Data Validation ---
    if 'factors_df' not in data or data['factors_df'].empty: print("ERROR: factors_df not available."); return None
    if 'variance' not in data or not data['variance']: print("ERROR: Variance explained data not available."); return None
    if 'r2_per_factor' not in data['variance']:
        if 'r2_per_factor_raw' in data['variance']: print("ERROR: Using raw r2_per_factor due to shape mismatch."); r2_per_factor = data['variance']['r2_per_factor_raw']
        else: print("ERROR: Variance per factor not found."); return None
    else: r2_per_factor = data['variance']['r2_per_factor']

    factors_df = data['factors_df']
    views = data.get('views', [])
    factor_cols = [col for col in factors_df.columns if col.startswith('Factor')]

    # --- Load Correlation Data ---
    if not os.path.exists(correlation_file):
        print(f"FATAL ERROR: Correlation file '{correlation_file}' not found!")
        return None
    try:
        corr_results_df = pd.read_csv(correlation_file)
        print(f"Loaded correlation results from: {correlation_file}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load correlation file '{correlation_file}': {e}")
        return None

    # --- Dynamically Identify Key Factors ---
    def find_top_factor(corr_df, meta_var, fdr_thresh=0.05, sign_filter=None):
        """Finds the top factor significantly correlated with a metadata variable."""
        filtered = corr_df[
            (corr_df['Metadata'].str.startswith(meta_var)) &
            (corr_df['Significant_FDR'] == True) &
            (corr_df['P_value_FDR'] < fdr_thresh)
        ].copy()

        if filtered.empty:
            print(f"  WARNING: No significant factor found for '{meta_var}' at FDR < {fdr_thresh}.")
            return None

        if sign_filter == 'positive':
            filtered = filtered[filtered['Correlation'] > 0]
        elif sign_filter == 'negative':
            filtered = filtered[filtered['Correlation'] < 0]

        if filtered.empty:
            print(f"  WARNING: No significant factor found for '{meta_var}' with sign filter '{sign_filter}'.")
            return None

        filtered['Abs_Correlation'] = filtered['Correlation'].abs()
        filtered = filtered.sort_values(by=['P_value_FDR', 'Abs_Correlation'], ascending=[True, False])

        top_factor = filtered['Factor'].iloc[0]
        top_corr = filtered['Correlation'].iloc[0]
        top_fdr = filtered['P_value_FDR'].iloc[0]
        print(f"  Identified Factor for '{meta_var}': {top_factor} (Corr={top_corr:.3f}, FDR={top_fdr:.2e})")
        return top_factor

    print("Identifying key factors from correlations:")
    geno_factor_name = find_top_factor(corr_results_df, 'Genotype')
    time_factor_name = find_top_factor(corr_results_df, 'Time Point')
    if time_factor_name is None:
         time_factor_name = find_top_factor(corr_results_df, 'Day')
         if time_factor_name: print("  -> Used 'Day' as fallback for Time Point.")

    treat_factor_name = find_top_factor(corr_results_df, 'Treatment', sign_filter='negative')
    batch_factor_name = find_top_factor(corr_results_df, 'Batch', sign_filter='negative')

    treat_batch_factor_name = treat_factor_name
    if treat_batch_factor_name is None:
        treat_batch_factor_name = batch_factor_name
        print(f"  WARNING: No significant negative Treatment factor found, using Batch factor '{batch_factor_name}' for Panel D/F.")
    elif batch_factor_name is not None and treat_batch_factor_name != batch_factor_name:
        print(f"  NOTE: Top Treatment factor ({treat_factor_name}) differs from top Batch factor ({batch_factor_name}). Using Treatment factor for plots.")
    elif batch_factor_name is None:
         print(f"  NOTE: No significant negative Batch factor found. Using Treatment factor '{treat_batch_factor_name}' for plots.")

    # --- Validation ---
    if not all([geno_factor_name, time_factor_name, treat_batch_factor_name]):
        print("\nFATAL ERROR: Could not identify all required factors (Genotype, Time, Treatment/Batch) from the correlation file.")
        print("Please check 'mofa_factor_metadata_associations_spearman.csv'.")
        return None

    print(f"-> Using: Genotype={geno_factor_name}, Time={time_factor_name}, Treat/Batch={treat_batch_factor_name}")

    geno_desc = "Drought Tolerance (G1 vs G2)"
    time_desc = "Temporal Response"
    treat_batch_desc = "Treatment/Batch Effect"

    try:
        # --- Set up Figure Layout ---
        fig = plt.figure(figsize=(20, 22))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 1], width_ratios=[1, 1],
                              wspace=0.3, hspace=0.5)

        # --- PANEL A: Variance Explained ---
        var_df = verify_variance_explained(r2_per_factor, views)
        
        ax_var = fig.add_subplot(gs[0, 0])
        n_views = len(views)
        n_factors = r2_per_factor.shape[1]
        factor_labels = [f"Factor{i+1}" for i in range(n_factors)]
        df_var = var_df
        df_plot_var = df_var.T
        bottom = np.zeros(n_factors)
        view_colors_actual = {view: VIEW_COLORS.get(view, plt.cm.viridis(i / max(1, n_views - 1)))
                              for i, view in enumerate(df_var.index)}

        for view in df_var.index:
            if view not in df_plot_var.columns: continue
            values = df_plot_var[view].values
            if len(values) != n_factors or len(bottom) != n_factors: continue
            bars = ax_var.bar(range(n_factors), values, bottom=bottom,
                             label=view, color=view_colors_actual.get(view, 'gray'), width=0.75)
            bottom += values
            for i, v in enumerate(values):
                if v > 5.0:
                    try: ax_var.text(i, bottom[i] - v/2, f"{v:.1f}", ha='center', va='center', color='white', fontweight='bold', fontsize=15, path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
                    except IndexError: pass
        total_variance_per_factor = df_plot_var.sum(axis=1)
        for i, total in enumerate(total_variance_per_factor):
            try: ax_var.text(i, bottom[i] + 1, f"{total:.1f}%", ha='center', va='bottom', fontsize=15, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))
            except IndexError: pass
        ax_var.set_title('a    Variance Explained by MOFA+ Factors', fontsize=17, fontweight='bold', loc='left')
        ax_var.set_xlabel('Factors', fontsize=17)
        ax_var.set_ylabel('Variance Explained (%)', fontsize=17)
        ax_var.set_xticks(range(n_factors))
        ax_var.set_xticklabels(factor_labels, rotation=45, ha='right')
        ax_var.legend(title='Data Views', loc='upper right', fontsize=16, title_fontsize=19)
        ax_var.set_ylim(0, max(100.0, np.max(bottom) * 1.1))
        ax_var.grid(axis='y', alpha=0.3, linestyle='--')
        sns.despine(ax=ax_var)

        # --- PANEL B: Factor-Condition Correlation Heatmap ---
        ax_heatmap = fig.add_subplot(gs[0, 1])

        try:
            heatmap_corr_df = corr_results_df.copy()

            if 'Metadata' in heatmap_corr_df.columns:
                print("Replacing 'Day' with 'Time Point' in heatmap metadata labels...")
                heatmap_corr_df['Metadata'] = heatmap_corr_df['Metadata'].str.replace('Day', 'Time Point', regex=False)

            pivot_df_b = heatmap_corr_df.pivot(index='Factor', columns='Metadata', values='Correlation')
            
            sig_pivot_b = heatmap_corr_df.pivot(index='Factor', columns='Metadata', values='Significant_FDR')
            
            sig_pivot_b = sig_pivot_b.reindex_like(pivot_df_b).fillna(False).astype(bool)
            mask_b = ~sig_pivot_b

            factors_in_hdf5 = data['factors_df'].columns[data['factors_df'].columns.str.startswith('Factor')].tolist()
            factors_present_in_pivot = pivot_df_b.index.tolist()
            factors_to_plot_heatmap = [f for f in factors_in_hdf5 if f in factors_present_in_pivot]

            if not factors_to_plot_heatmap:
                 raise ValueError("No common factors found between HDF5 factors and correlation pivot table.")

            pivot_df_b = pivot_df_b.loc[factors_to_plot_heatmap]
            mask_b = mask_b.loc[factors_to_plot_heatmap]

            cmap_colors = ["#fe9929", "#FFFFFF", "#238b45"] # Negative, Center, Positive
            cmap_b = LinearSegmentedColormap.from_list("custom_diverging", cmap_colors)

            sns.heatmap(
                pivot_df_b, annot=True, annot_kws={"size": 15}, fmt=".2f", cmap=cmap_b,
                center=0, vmin=-1, vmax=1, mask=mask_b, linewidths=0.5, linecolor='lightgray',
                cbar_kws={"shrink": 0.7, "label": "Spearman's ρ"}, ax=ax_heatmap
            )

        except Exception as e_heatmap:
            print(f"ERROR preparing data or plotting heatmap Panel B: {e_heatmap}")
            traceback.print_exc()
            ax_heatmap.text(0.5, 0.5, f"Correlation heatmap error:\n{e_heatmap}",
                         ha='center', va='center', transform=ax_heatmap.transAxes, color='red', fontsize=10, wrap=True)

        ax_heatmap.set_title('b    Factor-Metadata Correlations', fontsize=17, fontweight='bold', loc='left')
        ax_heatmap.set_xlabel('Experimental Condition / Variable', fontsize=17)
        ax_heatmap.set_ylabel('MOFA+ Factor', fontsize=17)
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right')

        # --- PANELS C & D: Factor Scatter Plots ---
        ax_scatter1 = fig.add_subplot(gs[1, 0]) # Genotype vs Time
        ax_scatter2 = fig.add_subplot(gs[1, 1]) # Genotype vs Treatment/Batch

        merged_df = factors_df.copy()
        if 'Time Point' in merged_df.columns: merged_df['Time Point'] = pd.to_numeric(merged_df['Time Point'], errors='coerce')
        if 'Genotype' in merged_df.columns: merged_df['Genotype'] = merged_df['Genotype'].astype(str)
        if 'Treatment' in merged_df.columns:
            merged_df['Treatment'] = merged_df['Treatment'].astype(str)
            treatment_map = {'0': 'T0', '1': 'T1'}
            merged_df['TreatmentLabel'] = merged_df['Treatment'].map(treatment_map).fillna(merged_df['Treatment'])

        has_genotype = 'Genotype' in merged_df.columns
        has_treatment = 'Treatment' in merged_df.columns
        has_timepoint = 'Time Point' in merged_df.columns

        # --- Use Identified Factors ---
        x_factor_c = geno_factor_name
        y_factor_c = time_factor_name
        x_factor_d = geno_factor_name
        y_factor_d = treat_batch_factor_name

        # --- PANEL C: Factor Scatter Plot 1 (Genotype vs Time) ---
        if has_genotype and has_timepoint and x_factor_c in merged_df.columns and y_factor_c in merged_df.columns:
            for geno in sorted(merged_df['Genotype'].unique()):
                if geno not in ['G1', 'G2', '1', '2']: continue
                for time_point in sorted(merged_df['Time Point'].unique()):
                    if pd.isna(time_point): continue
                    mask = (merged_df['Genotype'] == geno) & (merged_df['Time Point'] == time_point)
                    if sum(mask) >= 5:
                        x = merged_df.loc[mask, x_factor_c]
                        y = merged_df.loc[mask, y_factor_c]
                        ellipse, (mean_x, mean_y) = confidence_ellipse(x, y, ax_scatter1, n_std=1.96, edgecolor=TIMEPOINT_COLORS.get(int(time_point), 'gray'), alpha=0.5, linewidth=2, linestyle='-' if geno in ['G1', '1'] else '--')
                        if ellipse: ax_scatter1.add_patch(ellipse)
                        marker = 'o' if geno in ['G1', '1'] else '^'
                        ax_scatter1.scatter(mean_x, mean_y, s=150, c=TIMEPOINT_COLORS.get(int(time_point), 'gray'), marker=marker, edgecolor='black', linewidth=1.5, zorder=10)
            for geno in sorted(merged_df['Genotype'].unique()):
                 if geno not in ['G1', 'G2', '1', '2']: continue
                 for time_point in sorted(merged_df['Time Point'].unique()):
                      if pd.isna(time_point): continue
                      mask = (merged_df['Genotype'] == geno) & (merged_df['Time Point'] == time_point)
                      if sum(mask) > 0:
                           marker = 'o' if geno in ['G1', '1'] else '^'
                           ax_scatter1.scatter(merged_df.loc[mask, x_factor_c], merged_df.loc[mask, y_factor_c], s=20, c=TIMEPOINT_COLORS.get(int(time_point), 'gray'), marker=marker, alpha=0.2, edgecolor=None)

            # Legend construction
            genotype_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='G1 (Tolerant)'), Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markersize=10, label='G2 (Susceptible)')]
            timepoint_handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=TIMEPOINT_COLORS.get(int(tp), 'gray'), markersize=10, label=f'Time Point {int(tp)}') for tp in sorted(merged_df['Time Point'].dropna().unique())]
            legend1 = ax_scatter1.legend(handles=genotype_handles, title="Genotype", loc='upper left', fontsize=16, title_fontsize=19)
            ax_scatter1.add_artist(legend1)
            ax_scatter1.legend(handles=timepoint_handles, title="Time Point", loc='upper right', fontsize=16, title_fontsize=19)
        else: ax_scatter1.text(0.5, 0.5, "Data missing for plot C", ha='center', va='center', transform=ax_scatter1.transAxes)
        ax_scatter1.set_xlabel(f"{x_factor_c.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_scatter1.set_ylabel(f"{y_factor_c.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_scatter1.set_title(f"c    Genotype and temporal progression define sample distribution", fontsize=17, fontweight='bold', loc='left')
        ax_scatter1.grid(True, linestyle='--', alpha=0.7); ax_scatter1.axhline(y=0, color='gray', linestyle='-', alpha=0.3); ax_scatter1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        # --- PANEL D: Factor Scatter Plot 2 (Genotype vs Treatment/Batch) ---
        if has_genotype and has_treatment and x_factor_d in merged_df.columns and y_factor_d in merged_df.columns:
            for geno in sorted(merged_df['Genotype'].unique()):
                if geno not in ['G1', 'G2', '1', '2']: continue
                for treat_label, treat_val in [('T0', '0'), ('T1', '1')]:
                     mask = (merged_df['Genotype'] == geno) & (merged_df['Treatment'] == treat_val)
                     if sum(mask) >= 5:
                          x = merged_df.loc[mask, x_factor_d]
                          y = merged_df.loc[mask, y_factor_d]
                          treatment_color = TREATMENT_COLORS.get(treat_val, TREATMENT_COLORS.get(treat_label, 'gray'))
                          ellipse, (mean_x, mean_y) = confidence_ellipse(x, y, ax_scatter2, n_std=1.96, edgecolor=treatment_color, alpha=0.5, linewidth=2, linestyle='-' if geno in ['G1', '1'] else '--')
                          if ellipse: ax_scatter2.add_patch(ellipse)
                          marker = 'o' if geno in ['G1', '1'] else '^'
                          ax_scatter2.scatter(mean_x, mean_y, s=150, c=treatment_color, marker=marker, edgecolor='black', linewidth=1.5, zorder=10)
            for geno in sorted(merged_df['Genotype'].unique()):
                if geno not in ['G1', 'G2', '1', '2']: continue
                for treat_label, treat_val in [('T0', '0'), ('T1', '1')]:
                     mask = (merged_df['Genotype'] == geno) & (merged_df['Treatment'] == treat_val)
                     if sum(mask) > 0:
                          marker = 'o' if geno in ['G1', '1'] else '^'
                          treatment_color = TREATMENT_COLORS.get(treat_val, TREATMENT_COLORS.get(treat_label, 'gray'))
                          ax_scatter2.scatter(merged_df.loc[mask, x_factor_d], merged_df.loc[mask, y_factor_d], s=20, c=treatment_color, marker=marker, alpha=0.2, edgecolor=None)

            # Legend construction
            genotype_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='G1 (Tolerant)'), Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markersize=10, label='G2 (Susceptible)')]
            treatment_handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=TREATMENT_COLORS.get(treat, 'gray'), markersize=10, label=treat) for treat in ['T0', 'T1']]
            legend2 = ax_scatter2.legend(handles=genotype_handles, title="Genotype", loc='upper left', fontsize=16, title_fontsize=19)
            ax_scatter2.add_artist(legend2)
            ax_scatter2.legend(handles=treatment_handles, title="Treatment", loc='upper right', fontsize=16, title_fontsize=19)
        else: ax_scatter2.text(0.5, 0.5, "Data missing for plot D", ha='center', va='center', transform=ax_scatter2.transAxes)
        ax_scatter2.set_xlabel(f"{x_factor_d.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_scatter2.set_ylabel(f"{y_factor_d.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_scatter2.set_title(f"d    Genotype and treatment effects define sample distribution", fontsize=17, fontweight='bold', loc='left')
        ax_scatter2.grid(True, linestyle='--', alpha=0.7); ax_scatter2.axhline(y=0, color='gray', linestyle='-', alpha=0.3); ax_scatter2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)


        # --- PANEL E: Box Plot (Time Factor by Time Point/Genotype) ---
        ax_box1 = fig.add_subplot(gs[2, 0])
        factor_e = time_factor_name
        if has_genotype and has_timepoint and factor_e in merged_df.columns:
            plot_df = merged_df.copy()
            sns.boxplot(x='Time Point', y=factor_e, hue='Genotype', data=plot_df, palette=GENOTYPE_COLORS, ax=ax_box1)
            sns.stripplot(x='Time Point', y=factor_e, hue='Genotype', data=plot_df, palette=GENOTYPE_COLORS, dodge=True, alpha=0.3, size=4, ax=ax_box1)
            
            # --- Add Significance Markers ---
            geno_labels = sorted(plot_df['Genotype'].unique())
            if len(geno_labels) == 2:
                time_points = sorted(plot_df['Time Point'].dropna().unique())
                for i, tp in enumerate(time_points):
                    g1_data = plot_df[(plot_df['Time Point'] == tp) & (plot_df['Genotype'] == geno_labels[0])][factor_e]
                    g2_data = plot_df[(plot_df['Time Point'] == tp) & (plot_df['Genotype'] == geno_labels[1])][factor_e]

                    if g1_data.empty or g2_data.empty: continue
                    try: _, p_val = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                    except ValueError: continue

                    if p_val >= 0.05: continue
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    
                    y_max = max(g1_data.max(), g2_data.max())
                    y_range = ax_box1.get_ylim()[1] - ax_box1.get_ylim()[0]
                    y_pos = y_max + y_range * 0.05
                    bar_height = y_range * 0.02
                    text_offset = y_range * 0.01
                    x1, x2 = i - 0.2, i + 0.2

                    ax_box1.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=1.5, c='k')
                    ax_box1.text((x1 + x2) / 2, y_pos + bar_height + text_offset, sig, ha='center', va='bottom', color='k', fontsize=14)
                    
                    current_ylim_top = ax_box1.get_ylim()[1]
                    new_top = y_pos + bar_height + y_range * 0.1
                    if new_top > current_ylim_top:
                        ax_box1.set_ylim(top=new_top)

            # Legend cleanup
            handles, labels = ax_box1.get_legend_handles_labels(); unique_labels = []; unique_handles = []; seen_labels = set()
            for handle, label in zip(handles, labels):
                if label.upper() not in seen_labels: seen_labels.add(label.upper()); unique_labels.append("G1 (Tolerant)" if label in ['G1', '1'] else "G2 (Susceptible)"); unique_handles.append(handle);
                if len(unique_labels) == 2: break
            ax_box1.legend(unique_handles, unique_labels, title="Genotype", fontsize=16, title_fontsize=19, loc='lower left')
        else: ax_box1.text(0.5, 0.5, "Data missing for plot E", ha='center', va='center', transform=ax_box1.transAxes)
        ax_box1.set_xlabel("Time Point", fontsize=17)
        ax_box1.set_ylabel(f"{factor_e.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_box1.set_title(f"e    Distribution of {factor_e} by Time Point and Genotype", fontsize=17, fontweight='bold', loc='left')

        # --- PANEL F: Box Plot (Treatment/Batch Factor by Treatment/Genotype) ---
        verify_treatment_factor_plot(merged_df, treat_batch_factor_name)
        
        ax_box2 = fig.add_subplot(gs[2, 1])
        factor_f = treat_batch_factor_name
        if has_genotype and has_treatment and factor_f in merged_df.columns:
            plot_df = merged_df.copy()
            if 'TreatmentLabel' in plot_df.columns: plot_df['TreatmentPlot'] = plot_df['TreatmentLabel']
            else: plot_df['TreatmentPlot'] = plot_df['Treatment'].replace({'0': 'T0', '1': 'T1'})
            sns.boxplot(x='TreatmentPlot', y=factor_f, hue='Genotype', data=plot_df, palette=GENOTYPE_COLORS, ax=ax_box2)
            sns.stripplot(x='TreatmentPlot', y=factor_f, hue='Genotype', data=plot_df, palette=GENOTYPE_COLORS, dodge=True, alpha=0.3, size=4, ax=ax_box2)

            # --- Add Significance Markers ---
            geno_labels = sorted(plot_df['Genotype'].unique())
            if len(geno_labels) == 2:
                treatment_groups = sorted(plot_df['TreatmentPlot'].unique())
                for i, treat in enumerate(treatment_groups):
                    g1_data = plot_df[(plot_df['TreatmentPlot'] == treat) & (plot_df['Genotype'] == geno_labels[0])][factor_f]
                    g2_data = plot_df[(plot_df['TreatmentPlot'] == treat) & (plot_df['Genotype'] == geno_labels[1])][factor_f]

                    if g1_data.empty or g2_data.empty: continue
                    try: _, p_val = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                    except ValueError: continue

                    if p_val >= 0.05: continue
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    
                    y_max = max(g1_data.max(), g2_data.max())
                    y_range = ax_box2.get_ylim()[1] - ax_box2.get_ylim()[0]
                    y_pos = y_max + y_range * 0.05
                    bar_height = y_range * 0.02
                    text_offset = y_range * 0.01
                    x1, x2 = i - 0.2, i + 0.2

                    ax_box2.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=1.5, c='k')
                    ax_box2.text((x1 + x2) / 2, y_pos + bar_height + text_offset, sig, ha='center', va='bottom', color='k', fontsize=14)

                    current_ylim_top = ax_box2.get_ylim()[1]
                    new_top = y_pos + bar_height + y_range * 0.1
                    if new_top > current_ylim_top:
                        ax_box2.set_ylim(top=new_top)

            # Legend cleanup
            handles, labels = ax_box2.get_legend_handles_labels(); unique_labels = []; unique_handles = []; seen_labels = set()
            for handle, label in zip(handles, labels):
                if label.upper() not in seen_labels: seen_labels.add(label.upper()); unique_labels.append("G1 (Tolerant)" if label in ['G1', '1'] else "G2 (Susceptible)"); unique_handles.append(handle);
                if len(unique_labels) == 2: break
            ax_box2.legend(unique_handles, unique_labels, title="Genotype", fontsize=16, title_fontsize=19, loc='lower left')
        else: ax_box2.text(0.5, 0.5, "Data missing for plot F", ha='center', va='center', transform=ax_box2.transAxes)
        ax_box2.set_xlabel("Treatment", fontsize=17)
        ax_box2.set_ylabel(f"{factor_f.replace('Factor', 'Factor ')} Score", fontsize=17)
        ax_box2.set_title(f"f    Distribution of {factor_f} by Treatment and Genotype", fontsize=17, fontweight='bold', loc='left')

        # --- Save the figure ---
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"figure1_Integrated_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"figure1_Integrated_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close()

        print(f"SUCCESS: Saved Integrated Figure 1 (using factors G={geno_factor_name}, T={time_factor_name}, Tr/B={treat_batch_factor_name}) to {output_file}")
        
        # --- Consistency checks ---
        print("\nConsistency checks:")
        total_variance = r2_per_factor.sum()
        print(f"  Total variance explained across all factors: {total_variance:.2f}%")
        if total_variance > 100*len(views):
            print(f"  WARNING: Total variance exceeds expected maximum ({100*len(views)}%)")
        
        if 'Treatment' in merged_df.columns and treat_batch_factor_name in merged_df.columns:
            treat_corr = merged_df.groupby('Treatment')[treat_batch_factor_name].mean()
            print(f"  Treatment group means for {treat_batch_factor_name}:")
            print(f"    {treat_corr}")
            t0_val = treat_corr.get('0', treat_corr.get('T0'))
            t1_val = treat_corr.get('1', treat_corr.get('T1'))
            if t0_val is not None and t1_val is not None:
                if t0_val < t1_val:
                    print(f"  NOTE: {treat_batch_factor_name} has POSITIVE correlation with treatment (T1 > T0)")
                else:
                    print(f"  NOTE: {treat_batch_factor_name} has NEGATIVE correlation with treatment (T0 > T1)")

        # Export data tables if requested
        if export_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            var_export_path = os.path.join(output_dir, f"variance_explained_data_{timestamp}.csv")
            var_df.to_csv(var_export_path)
            print(f"Exported variance explained data to: {var_export_path}")
            
            if merged_df is not None:
                plot_cols = ['Genotype', 'Treatment', 'Time Point']
                if 'Day' in merged_df.columns and 'Time Point' not in merged_df.columns:
                    plot_cols[plot_cols.index('Time Point')] = 'Day'
                
                factor_cols = []
                if geno_factor_name in merged_df.columns:
                    factor_cols.append(geno_factor_name)
                if time_factor_name in merged_df.columns:
                    factor_cols.append(time_factor_name)
                if treat_batch_factor_name in merged_df.columns:
                    factor_cols.append(treat_batch_factor_name)
                
                plot_cols = [col for col in plot_cols if col in merged_df.columns]
                plot_data = merged_df[plot_cols + factor_cols]
                
                plot_data_path = os.path.join(output_dir, f"factor_plot_data_{timestamp}.csv")
                plot_data.to_csv(plot_data_path, index=False)
                print(f"Exported factor plot data to: {plot_data_path}")
                
                if os.path.exists(correlation_file):
                    try:
                        corr_df = pd.read_csv(correlation_file)
                        key_factor_corr = corr_df[corr_df['Factor'].isin(factor_cols)]
                        corr_export_path = os.path.join(output_dir, f"key_factor_correlations_{timestamp}.csv")
                        key_factor_corr.to_csv(corr_export_path, index=False)
                        print(f"Exported key factor correlation data to: {corr_export_path}")
                    except Exception as e:
                        print(f"Could not export correlation data: {e}")
        
        return output_file

    except Exception as e:
        print(f"ERROR creating integrated figure 1: {e}")
        traceback.print_exc()
        plt.close('all')
        return None

# --- Main Execution ---

def main():
    """Main function to run the visualization"""
    # --- Configuration ---
    mofa_file = r"C:/Users/ms/Desktop/hyper/output/mofa/mofa_model_for_transformer.hdf5"
    metadata_file = r"C:/Users/ms/Desktop/hyper/output/mofa/aligned_combined_metadata.csv"
    correlation_file = r"C:/Users/ms/Desktop/hyper/output/mofa/mofa_factor_metadata_associations_spearman.csv"
    output_dir = r"C:/Users/ms/Desktop/hyper/output/transformer/novility_plot"
    export_data = False

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"Integrated MOFA+ Figure 1 Generator - START")
    print(f"MOFA+ model: {mofa_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Correlations: {correlation_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # --- Load Data ---
    data = load_data(mofa_file, metadata_file)
    if data is None:
        print("\nFATAL ERROR: Data loading failed. Exiting.")
        return 1

    # --- Cross-Check with Original Data ---
    if os.path.exists(correlation_file):
        print("\nVerifying factor-metadata correlations from file:")
        corr_df = pd.read_csv(correlation_file)
        try:
            # Identify key factors based on correlations
            def find_top_factor(corr_df, meta_var, fdr_thresh=0.05, sign_filter=None):
                filtered = corr_df[
                    (corr_df['Metadata'].str.startswith(meta_var)) &
                    (corr_df['Significant_FDR'] == True) &
                    (corr_df['P_value_FDR'] < fdr_thresh)
                ].copy()

                if filtered.empty:
                    return None

                if sign_filter == 'positive':
                    filtered = filtered[filtered['Correlation'] > 0]
                elif sign_filter == 'negative':
                    filtered = filtered[filtered['Correlation'] < 0]

                if filtered.empty:
                    return None

                filtered['Abs_Correlation'] = filtered['Correlation'].abs()
                filtered = filtered.sort_values(by=['P_value_FDR', 'Abs_Correlation'], ascending=[True, False])

                return filtered['Factor'].iloc[0]

            geno_factor_name = find_top_factor(corr_df, 'Genotype')
            time_factor_name = find_top_factor(corr_df, 'Time Point')
            if time_factor_name is None:
                time_factor_name = find_top_factor(corr_df, 'Day')
            treat_factor_name = find_top_factor(corr_df, 'Treatment', sign_filter='negative')
            batch_factor_name = find_top_factor(corr_df, 'Batch', sign_filter='negative')
            treat_batch_factor_name = treat_factor_name if treat_factor_name is not None else batch_factor_name

            key_factors = []
            if geno_factor_name: key_factors.append(geno_factor_name)
            if time_factor_name: key_factors.append(time_factor_name)
            if treat_batch_factor_name: key_factors.append(treat_batch_factor_name)

            print(f"Identified key factors: {', '.join(key_factors)}")
            for factor in key_factors:
                for meta in ['Genotype', 'Treatment', 'Time Point', 'Day', 'Batch']:
                    matching_rows = corr_df[(corr_df['Factor'] == factor) &
                                          (corr_df['Metadata'] == meta)]
                    if not matching_rows.empty:
                        corr_val = matching_rows['Correlation'].iloc[0]
                        p_val = matching_rows['P_value_FDR'].iloc[0]
                        print(f"  {factor} vs {meta}: corr={corr_val:.3f}, FDR p={p_val:.3e}")
        except Exception as e:
            print(f"WARNING: Could not cross-check correlation data: {e}")

    # --- Generate Figure 1 ---
    figure1_path = create_integrated_figure1(data, correlation_file, output_dir, export_data)

    # --- Print Summary ---
    print("\n" + "=" * 80)
    if figure1_path:
        print(f"✅ Successfully generated Figure 1: {figure1_path}")
    else:
        print("❌ Failed to generate Figure 1. Please check the logs.")
    print("=" * 80)

    return 0 if figure1_path else 1

if __name__ == "__main__":
    sys.exit(main())
