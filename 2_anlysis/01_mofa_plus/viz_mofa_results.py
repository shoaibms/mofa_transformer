#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MOFA+ Visualization Script

This script creates publication-quality visualizations from MOFA+ output files.
It includes advanced visualizations designed for high-impact journals.
MOFA+ (Multi-Omics Factor Analysis plus) is a statistical method for
integrating multi-omics data, and this script provides enhanced visualization
tools for exploring MOFA+ results.

The script generates the following visualizations:
- Variance explained plots
- Factor-condition correlation heatmaps
- Temporal trajectory visualizations
- Cross-view integration networks
- Feature loadings
- Genotype response networks
- 3D factor space plots

Usage:
    python enhanced_mofa_viz.py /path/to/mofa_model.hdf5 /path/to/metadata.csv [options]
    
Options:
    --output DIR          Output directory (default: mofa_visualizations)
    --all                 Generate all visualizations
    --variance            Generate variance explained plot
    --heatmap             Generate factor-condition heatmap
    --trajectory          Generate temporal trajectory plot
    --integration         Generate cross-view integration network
    --loadings            Generate feature loadings visualizations
    --genotype_net        Generate genotype response network
    --plot3d              Generate 3D factor space visualization
    
Example:
    python enhanced_mofa_viz.py model.hdf5 metadata.csv --output results --variance --heatmap

Requirements:
    - pandas, numpy, matplotlib, seaborn, scipy, networkx, h5py
    - MOFA+ HDF5 format output file
    - Metadata CSV file with sample information matching MOFA+ samples
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
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
from scipy.stats import spearmanr
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
import traceback
from datetime import datetime
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# Set up plotting parameters for publication quality
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42  # Output as Type 42 (TrueType)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Color palettes
VIEW_COLORS = {'leaf_spectral': '#440154', 'root_spectral': '#21918c',
               'leaf_metabolite': '#fde725', 'root_metabolite': '#5ec962'}

GENOTYPE_COLORS = {'G1': '#3b528b', 'G2': '#21918c'}
TREATMENT_COLORS = {0: '#5ec962', 1: '#fde725'}
TISSUE_COLORS = {'L': '#440154', 'R': '#21918c'}

def safe_decode(byte_string):
    """Safely decodes byte strings, returns original if not bytes."""
    if isinstance(byte_string, bytes):
        try:
            return byte_string.decode('utf-8')
        except UnicodeDecodeError:
            return byte_string.decode('latin-1', errors='replace')
    return byte_string

def load_data(mofa_file, metadata_file):
    """Load data from MOFA+ HDF5 file and metadata CSV"""
    print(f"Loading data from {mofa_file} and {metadata_file}")
    results = {}
    metadata = None

    # --- Metadata Loading ---
    if not os.path.exists(metadata_file):
        print(f"FATAL ERROR: Metadata file {metadata_file} does not exist!")
        return None  # Cannot proceed without metadata
    try:
        metadata = pd.read_csv(metadata_file)
        print(f"Metadata loaded: {len(metadata)} rows, {len(metadata.columns)} columns")
        print(f"Metadata columns: {metadata.columns.tolist()}")
        
        required_meta_cols = ['Genotype', 'Treatment', 'Day', 'Tissue.type', 'Batch']
        missing_cols = [col for col in required_meta_cols if col not in metadata.columns]
        if missing_cols:
            print(f"WARNING: Metadata missing required columns: {missing_cols}")
        else:
            print("All essential metadata columns found.")
            # Print unique values for key columns
            for col in required_meta_cols:
                unique_values = metadata[col].unique()
                print(f"  Column '{col}' unique values ({len(unique_values)}): {unique_values}")
        results['metadata'] = metadata
    except Exception as e:
        print(f"FATAL ERROR reading metadata file: {e}")
        traceback.print_exc()
        return None  # Cannot proceed without metadata

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
                results['views'] = views
                print(f"Found views: {views}")
            else:
                print("ERROR: 'views/views' dataset not found in HDF5 file.")
                results['views'] = []

            # Extract samples
            if 'samples' in f and 'group0' in f['samples']:
                sample_data = f['samples']['group0'][()]
                results['samples'] = {'group0': [safe_decode(s) for s in sample_data]}
                num_samples = len(results['samples']['group0'])
                print(f"Loaded {num_samples} sample names for group 'group0'")
                # --- Metadata Alignment Check ---
                if metadata is not None and len(metadata) != num_samples:
                    print(f"FATAL ERROR: Metadata row count ({len(metadata)}) does not match MOFA sample count ({num_samples})!")
                    return None
            else:
                print("ERROR: 'samples/group0' not found in HDF5 file.")
                return None  # Cannot proceed without samples

            # Extract factors (Z)
            if 'expectations' in f and 'Z' in f['expectations'] and 'group0' in f['expectations']['Z']:
                z_data = f['expectations']['Z']['group0'][()]
                # Original shape: (factors, samples). Transpose to (samples, factors)
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
                    w_data = f['expectations']['W'][view_key][()]
                    weights[view_name] = w_data
                    # Check factor dimension consistency
                    if expected_factors is not None and w_data.shape[0] != expected_factors:
                        print(f"WARNING: Weight matrix for view '{view_name}' has {w_data.shape[0]} factors, expected {expected_factors}.")
                results['weights'] = weights
                print(f"Loaded weights (W) for {len(weights)} views.")
            else:
                print("ERROR: 'expectations/W' not found.")
                results['weights'] = {}

            # Extract feature names
            if 'features' in f:
                features = {}
                for view_key in f['features']:
                    view_name = safe_decode(view_key)
                    try:
                        feature_data = f['features'][view_key][()]
                        features[view_name] = [safe_decode(feat) for feat in feature_data]
                        # Check consistency with weights
                        if view_name in results['weights'] and len(features[view_name]) != results['weights'][view_name].shape[1]:
                            print(f"WARNING: Feature count for view '{view_name}' ({len(features[view_name])}) doesn't match weight matrix dimension ({results['weights'][view_name].shape[1]}).")
                    except Exception as e:
                        print(f"Error extracting features for {view_name}: {e}")
                results['features'] = features
                print(f"Loaded feature names for {len(features)} views.")
            else:
                print("ERROR: 'features' group not found.")
                results['features'] = {}

            # Extract variance explained
            variance_dict = {}
            if 'variance_explained' in f:
                # Extract r2_total
                if 'r2_total' in f['variance_explained'] and 'group0' in f['variance_explained']['r2_total']:
                    try:
                        r2_total_data = f['variance_explained']['r2_total']['group0'][()]
                        # Ensure it's per view, matching the number of views
                        if len(r2_total_data) == len(results.get('views', [])):
                            variance_dict['r2_total_per_view'] = r2_total_data
                            print(f"Loaded r2_total_per_view: {r2_total_data}")
                        else:
                            print(f"Warning: r2_total shape {r2_total_data.shape} mismatch with view count {len(results.get('views', []))}. Storing raw.")
                            variance_dict['r2_total_raw'] = r2_total_data
                    except Exception as e:
                        print(f"Error extracting r2_total: {e}")

                # Extract r2_per_factor
                r2_pf_path = 'variance_explained/r2_per_factor/group0'
                if r2_pf_path in f:
                    try:
                        r2_per_factor_data = f[r2_pf_path][()]
                        # Expected shape: (n_views, n_factors)
                        n_views = len(results.get('views', []))
                        n_factors = results.get('factors', np.array([])).shape[1]

                        if r2_per_factor_data.shape == (n_views, n_factors):
                            variance_dict['r2_per_factor'] = r2_per_factor_data  # Store as views x factors
                            print(f"Loaded r2_per_factor: shape {r2_per_factor_data.shape} (views x factors)")
                        else:
                            print(f"WARNING: r2_per_factor shape {r2_per_factor_data.shape} mismatch with expected ({n_views}, {n_factors}). Check model output.")
                            # Attempt transpose if dimensions match swapped
                            if r2_per_factor_data.shape == (n_factors, n_views):
                                variance_dict['r2_per_factor'] = r2_per_factor_data.T  # Store as views x factors
                                print(f"  -> Transposed to expected shape ({n_views}, {n_factors})")
                            else:
                                variance_dict['r2_per_factor_raw'] = r2_per_factor_data  # Store raw if shape is unexpected
                                print(f"  -> Stored raw r2_per_factor data.")

                    except Exception as e:
                        print(f"Error extracting r2_per_factor from {r2_pf_path}: {e}")
                        traceback.print_exc()
                else:
                    print(f"Dataset {r2_pf_path} not found.")

            results['variance'] = variance_dict

        # --- Post-Loading Processing ---

        # Create combined factor + metadata DataFrame
        if results.get('factors') is not None and metadata is not None:
            factors = results['factors']
            factor_cols = [f"Factor{i+1}" for i in range(factors.shape[1])]
            factors_df = pd.DataFrame(factors, columns=factor_cols)

            # Add all metadata columns, ensuring index alignment isn't broken
            meta_cols_to_add = [col for col in metadata.columns if col not in factors_df.columns]
            # Use .values to avoid index mismatch issues if pandas tries to align
            for col in meta_cols_to_add:
                try:
                    factors_df[col] = metadata[col].values
                except ValueError as ve:
                    print(f"ERROR adding metadata column '{col}': {ve}. Length mismatch likely.")
                    print(f"  Factor df length: {len(factors_df)}, Metadata column length: {len(metadata[col])}")
                    return None  # Cannot proceed if lengths don't match

            results['factors_df'] = factors_df
            print(f"Created combined factors + metadata DataFrame: {factors_df.shape}")
            print(f"  factors_df columns: {factors_df.columns.tolist()}")

        # Calculate feature importance if weights and features are available
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

                # Calculate importance as sum of absolute weights across factors
                # Weights shape: (factors, features)
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

    # --- Final check for essential components ---
    essential = ['views', 'samples', 'factors', 'weights', 'features', 'metadata', 'factors_df']
    missing_essential = []
    for item in essential:
        # Check if item exists and is not None
        if item not in results or results[item] is None:
            missing_essential.append(item)
            continue  # Skip to next item if missing or None

        value = results[item]
        is_empty = False
        # Check for emptiness based on type
        if isinstance(value, (dict, list)):
            if not value:  # Check if dict or list is empty
                is_empty = True
        elif isinstance(value, pd.DataFrame):
            if value.empty:  # Explicitly check if DataFrame is empty
                is_empty = True
        elif isinstance(value, np.ndarray):
            if value.size == 0:  # Check if numpy array is empty
                is_empty = True

        if is_empty:
            missing_essential.append(item)

    if missing_essential:
        print(f"FATAL ERROR: Missing or empty essential data components after loading: {missing_essential}")
        # Optionally print which ones are just empty vs missing
        print("Detailed Check:")
        for item in essential:
            status = "Missing/None"
            if item in results and results[item] is not None:
                value = results[item]
                if isinstance(value, (dict, list)) and not value: 
                    status = "Empty dict/list"
                elif isinstance(value, pd.DataFrame) and value.empty: 
                    status = "Empty DataFrame"
                elif isinstance(value, np.ndarray) and value.size == 0: 
                    status = "Empty ndarray"
                else: 
                    status = "Present"
            print(f"  - {item}: {status}")
        return None  # Exit if any essential component is missing or empty

    # If checks pass
    print("SUCCESS: All essential data components loaded successfully.")
    return results


def create_enhanced_variance_plot(data, output_dir):
    """Create an enhanced version of the variance explained plot"""
    print("\n--- Creating enhanced variance explained plot ---")

    # --- Data Validation ---
    if 'variance' not in data or not data['variance']:
        print("ERROR: Variance explained data ('variance') not available in loaded data.")
        return None
    if 'r2_per_factor' not in data['variance']:
        # Check if raw data exists due to shape mismatch warning during load
        if 'r2_per_factor_raw' in data['variance']:
            print("ERROR: 'r2_per_factor' has unexpected shape. Check MOFA model output. Using raw data if possible.")
            r2_per_factor = data['variance']['r2_per_factor_raw']
        else:
            print("ERROR: Variance per factor ('r2_per_factor') not found in variance data.")
            return None
    else:
        r2_per_factor = data['variance']['r2_per_factor']

    if 'views' not in data or not data['views']:
        print("ERROR: View names ('views') not available.")
        return None

    views = data['views']
    n_views = len(views)
    n_factors = r2_per_factor.shape[1]  # Shape is views x factors

    # Basic shape check
    if r2_per_factor.shape[0] != n_views:
        print(f"ERROR: Mismatch between r2_per_factor rows ({r2_per_factor.shape[0]}) and number of views ({n_views}).")
        return None
    if n_factors == 0:
        print("ERROR: Zero factors found in r2_per_factor data.")
        return None

    print(f"Plotting variance for {n_views} views and {n_factors} factors.")

    try:
        factor_labels = [f"Factor{i+1}" for i in range(n_factors)]

        # --- Create DataFrame for Plotting ---
        # Index=Views, Columns=Factors (r2_per_factor is already views x factors)
        df = pd.DataFrame(r2_per_factor, index=views, columns=factor_labels)
        df_plot = df.T  # Transpose for plotting (Factors as rows, Views as columns)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(max(8, n_factors * 0.6), 8))  # Adjust width based on factor count

        bottom = np.zeros(n_factors)
        view_colors = {view: VIEW_COLORS.get(view, plt.cm.viridis(i / max(1, n_views - 1))) for i, view in enumerate(views)}

        for view in views:
            if view not in df_plot.columns:  # Should not happen if df is correct
                print(f"Warning: View '{view}' not found in plotting DataFrame columns.")
                continue
            values = df_plot[view].values
            # Ensure values and bottom have same length
            if len(values) != n_factors or len(bottom) != n_factors:
                print(f"ERROR: Length mismatch for view '{view}'. Values: {len(values)}, Bottom: {len(bottom)}, Factors: {n_factors}")
                continue

            bars = ax.bar(range(n_factors), values, bottom=bottom,
                         label=view, color=view_colors.get(view, 'gray'), width=0.75)
            bottom += values  # Update bottom for next stack

            # Add value labels (adjust threshold for clarity)
            for i, v in enumerate(values):
                if v > 5.0:  # Label if explains > 5% variance
                    try:
                        ax.text(i, bottom[i] - v/2, f"{v:.1f}",  # Use 1 decimal place for percentages
                                ha='center', va='center', color='white', fontweight='bold',
                                fontsize=7, path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
                    except IndexError:
                        print(f"IndexError adding text for view {view}, factor {i+1}")

        # Calculate total variance explained per factor
        total_variance_per_factor = df_plot.sum(axis=1)

        # Add total variance text above bars
        for i, total in enumerate(total_variance_per_factor):
            try:
                ax.text(i, bottom[i] + 1, f"{total:.1f}%", ha='center', va='bottom',  # Add % sign
                       fontsize=8, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))
            except IndexError:
                print(f"IndexError adding total text for factor {i+1}")

        # Suggest factors based on elbow or threshold (e.g., >5%)
        suggested_factors_indices = np.where(total_variance_per_factor > 5.0)[0]
        if len(suggested_factors_indices) > 0:
            suggested_text = f"Factors > 5% total variance: " + ", ".join([f"F{i+1}" for i in suggested_factors_indices])
            ax.text(0.5, 1.06, suggested_text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='#f2f2f2', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

        # --- Enhance Plot Appearance ---
        ax.set_title('Variance Explained by MOFA+ Factors (per view)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Factors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')  # Clarify unit
        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_labels, rotation=45, ha='right')
        ax.legend(title='Data Views', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.set_ylim(0, max(100.0, np.max(bottom) * 1.1))  # Extend y-limit, max 100%
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        sns.despine(ax=ax)  # Remove top and right spines

        # Add total variance explained across all factors per view (from r2_total_per_view)
        if 'r2_total_per_view' in data['variance']:
            r2_total_view = data['variance']['r2_total_per_view']
            if len(r2_total_view) == n_views:
                total_text = "Total Variance Explained per View:\n"
                total_text += "\n".join([f"  {view}: {val:.1f}%" for view, val in zip(views, r2_total_view)])
                ax.text(1.03, 0.5, total_text, transform=ax.transAxes, fontsize=9, va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend/text

        # --- Save Plot ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"enhanced_variance_explained_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"enhanced_variance_explained_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved enhanced variance explained plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating enhanced variance explained plot: {e}")
        traceback.print_exc()
        plt.close('all')  # Close any potentially open figures
        return None


def create_factor_condition_heatmap(data, output_dir):
    """Create a heatmap of factor-condition correlations"""
    print("\n--- Creating factor-condition heatmap ---")

    # --- Data Validation ---
    if 'factors_df' not in data or data['factors_df'].empty:
        print("ERROR: Combined factors and metadata ('factors_df') not available or empty.")
        return None

    factors_df = data['factors_df']
    factor_cols = [col for col in factors_df.columns if col.startswith('Factor')]
    if not factor_cols:
        print("ERROR: No factor columns found in 'factors_df'.")
        return None

    # Identify potential experimental condition columns based on background info
    potential_exp_cols = ['Genotype', 'Treatment', 'Day', 'Batch', 'Tissue.type']
    available_exp_cols = [col for col in potential_exp_cols if col in factors_df.columns]

    if not available_exp_cols:
        print("ERROR: No relevant experimental condition columns found in 'factors_df'.")
        print(f"  Available columns: {factors_df.columns.tolist()}")
        return None
    print(f"Using experimental columns for correlation: {available_exp_cols}")

    try:
        correlations = []
        pvalues = []

        # --- Calculate Correlations ---
        for factor in factor_cols:
            if factor not in factors_df.columns: 
                continue  # Should not happen

            for condition in available_exp_cols:
                if condition not in factors_df.columns: continue # Should not happen

                condition_data = factors_df[condition].dropna() # Drop NaNs for correlation
                factor_data = factors_df.loc[condition_data.index, factor] # Align factor data

                if condition_data.empty or factor_data.empty:
                     print(f"Warning: Skipping {factor}-{condition} due to NaNs or empty data after alignment.")
                     continue

                # Check data types for correlation method
                if pd.api.types.is_numeric_dtype(condition_data) and condition_data.nunique() > 2:
                    # Numeric condition (like Day if treated as number, or potentially Batch if numeric ID)
                     try:
                         corr, p = spearmanr(factor_data, condition_data)
                         correlations.append({'Factor': factor, 'Condition': condition, 'Correlation': corr, 'P_value': p})
                         pvalues.append(p)
                     except ValueError as ve:
                          print(f"Warning: Spearman correlation failed for {factor}-{condition}: {ve}")

                else:
                    # Categorical condition (Genotype, Treatment, Tissue.type, Batch if string)
                    # Convert to string to be safe
                    condition_data = condition_data.astype(str)
                    categories = sorted(condition_data.unique())

                    # Handle binary case directly or create dummies
                    if len(categories) == 2:
                         # Use first category as reference (0 vs 1)
                         dummy = (condition_data == categories[1]).astype(int)
                         try:
                              corr, p = spearmanr(factor_data, dummy)
                              correlations.append({'Factor': factor, 'Condition': f"{condition}_{categories[1]}", 'Correlation': corr, 'P_value': p})
                              pvalues.append(p)
                         except ValueError as ve:
                              print(f"Warning: Spearman correlation failed for {factor}-{condition}_{categories[1]}: {ve}")
                    elif len(categories) > 2:
                        # Create dummy variables for multi-category
                        for category in categories:
                            dummy = (condition_data == category).astype(int)
                            try:
                                corr, p = spearmanr(factor_data, dummy)
                                correlations.append({'Factor': factor, 'Condition': f"{condition}_{category}", 'Correlation': corr, 'P_value': p})
                                pvalues.append(p)
                            except ValueError as ve:
                                print(f"Warning: Spearman correlation failed for {factor}-{condition}_{category}: {ve}")

        if not correlations:
            print("ERROR: No correlations could be calculated.")
            return None

        # --- Multiple Testing Correction ---
        corr_df = pd.DataFrame(correlations)
        valid_pvalues = corr_df['P_value'].dropna()
        if not valid_pvalues.empty:
             reject, pvalues_corrected, _, _ = multipletests(valid_pvalues, alpha=0.05, method='fdr_bh')
             # Map corrected p-values back, handle NaNs
             p_adjusted_map = dict(zip(valid_pvalues.index, pvalues_corrected))
             reject_map = dict(zip(valid_pvalues.index, reject))
             corr_df['P_adjusted'] = corr_df.index.map(p_adjusted_map)
             corr_df['Significant'] = corr_df.index.map(reject_map)
             # Fill NaNs where p-value was NaN initially
             corr_df['P_adjusted'].fillna(1.0, inplace=True)
             corr_df['Significant'].fillna(False, inplace=True)
        else:
             print("Warning: No valid p-values for multiple testing correction.")
             corr_df['P_adjusted'] = 1.0
             corr_df['Significant'] = False


        # --- Create Pivot Table and Mask ---
        try:
             pivot_df = corr_df.pivot(index='Factor', columns='Condition', values='Correlation')
             sig_pivot = corr_df.pivot(index='Factor', columns='Condition', values='Significant')
             # Ensure sig_pivot is boolean and aligns with pivot_df
             sig_pivot = sig_pivot.reindex_like(pivot_df).fillna(False).astype(bool)
             mask = ~sig_pivot # Mask where NOT significant
        except Exception as e:
             print(f"ERROR creating pivot tables for heatmap: {e}")
             traceback.print_exc()
             print("Correlation DF columns:", corr_df.columns)
             print("Correlation DF sample:\n", corr_df.head())
             return None

        # Check if pivot table is empty
        if pivot_df.empty:
            print("ERROR: Pivot table for heatmap is empty.")
            return None


        # --- Plotting ---
        plt.figure(figsize=(max(10, pivot_df.shape[1] * 0.6), max(8, pivot_df.shape[0] * 0.5)))
        cmap = sns.diverging_palette(240, 10, as_cmap=True) # Blue-Red diverging

        # Handle potential all-NaN columns/rows if pivoting failed partially
        pivot_df.dropna(axis=0, how='all', inplace=True)
        pivot_df.dropna(axis=1, how='all', inplace=True)
        mask = mask.reindex_like(pivot_df) # Realign mask after dropping NaNs

        if pivot_df.empty:
            print("ERROR: Pivot table became empty after dropping NaNs.")
            return None


        sns.heatmap(
            pivot_df,
            annot=True, # Annotate all cells
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            mask=mask, # Apply significance mask
            linewidths=0.5, linecolor='lightgray',
            cbar_kws={"shrink": 0.7, "label": "Spearman's ρ (Significant)"}
        )

        # Optional: Add annotations for significant values even if masked (using stars)
        # for i in range(pivot_df.shape[0]):
        #     for j in range(pivot_df.shape[1]):
        #         if sig_pivot.iloc[i, j]: # If significant
        #             plt.text(j + 0.5, i + 0.5, "*", ha='center', va='center', color='black', fontsize=10)


        # --- Enhance Plot Appearance ---
        plt.title('Factor - Condition Associations (Spearman Corr.)', fontsize=16, fontweight='bold')
        plt.xlabel('Experimental Condition / Variable', fontsize=12)
        plt.ylabel('MOFA+ Factor', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.figtext(0.5, 0.01, "Showing significant correlations only (FDR < 0.05). Masked cells are non-significant.",
                   ha='center', fontsize=9, wrap=True,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

        plt.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout


        # --- Save Plot and Data ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"factor_condition_heatmap_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"factor_condition_heatmap_{timestamp}.svg")
        corr_file = os.path.join(output_dir, f"factor_condition_correlations_{timestamp}.csv")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        corr_df.to_csv(corr_file, index=False)

        plt.close('all')
        print(f"SUCCESS: Saved factor-condition heatmap to {output_file}")
        print(f"SUCCESS: Saved correlation data to {corr_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating factor-condition heatmap: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def create_temporal_trajectory_plot(data, output_dir):
    """Create a plot showing temporal trajectories in factor space"""
    print("\n--- Creating temporal trajectory plot ---")

    # --- Data Validation ---
    if 'factors_df' not in data or data['factors_df'].empty:
        print("ERROR: Combined factors and metadata ('factors_df') not available.")
        return None

    factors_df = data['factors_df']
    factor_cols = [col for col in factors_df.columns if col.startswith('Factor')]
    if len(factor_cols) < 2:
        print(f"ERROR: Need at least 2 factors for trajectory plot. Found: {len(factor_cols)}")
        return None

    # Check for required metadata columns
    required_cols = ['Day']
    has_genotype = 'Genotype' in factors_df.columns
    has_treatment = 'Treatment' in factors_df.columns
    has_tissue = 'Tissue.type' in factors_df.columns
    if has_genotype: 
        required_cols.append('Genotype')
    if has_treatment: 
        required_cols.append('Treatment')
    if has_tissue: 
        required_cols.append('Tissue.type')

    missing_req = [col for col in required_cols if col not in factors_df.columns]
    if 'Day' not in factors_df.columns:
        print("ERROR: 'Day' column missing in factors_df. Cannot plot trajectories.")
        return None
    if missing_req:
        print(f"Warning: Missing optional columns for detailed trajectory plots: {missing_req}")

    # Select top 2 factors (consider using factors with highest variance or biological relevance later)
    factor1 = factor_cols[0]
    factor2 = factor_cols[1]
    print(f"Using factors {factor1} and {factor2} for trajectories.")
    print(f"Grouping variables available: Genotype={has_genotype}, Treatment={has_treatment}, Tissue={has_tissue}")

    # Ensure Day is numeric
    try:
        factors_df['Day'] = pd.to_numeric(factors_df['Day'])
    except ValueError:
        print("ERROR: 'Day' column cannot be converted to numeric.")
        return None

    try:
        # --- Set up Figure Layout ---
        # Determine layout based on available dimensions (especially tissue)
        if has_tissue and factors_df['Tissue.type'].nunique() > 1 and (has_genotype or has_treatment):
            n_tissues = factors_df['Tissue.type'].nunique()
            if n_tissues == 2:
                fig = plt.figure(figsize=(18, 10))
                gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
                ax_t1 = plt.subplot(gs[0, 0])  # Tissue 1 trajectories
                ax_t2 = plt.subplot(gs[0, 1])  # Tissue 2 trajectories
                ax_mag = plt.subplot(gs[1, 0])  # Response magnitude
                ax_div = plt.subplot(gs[1, 1])  # Divergence
                tissue_types = sorted(factors_df['Tissue.type'].unique())
                tissue_axes = {tissue_types[0]: ax_t1, tissue_types[1]: ax_t2}
                axes_for_analysis = {'magnitude': ax_mag, 'divergence': ax_div}
                print(f"Using 2x2 layout for tissues: {tissue_types}")
            else:  # More than 2 tissues - simplify layout for now
                print(f"Warning: Found {n_tissues} tissues. Simplifying layout to single trajectory panel.")
                fig, ax_traj = plt.subplots(figsize=(10, 8))
                tissue_axes = {'all_tissues_combined': ax_traj}
                axes_for_analysis = {}
                has_tissue = False  # Treat as combined for plotting simplicity
        else:
            # Simpler figure: one panel for trajectories
            fig, ax_traj = plt.subplots(figsize=(10, 8))
            tissue_axes = {'all': ax_traj}  # Use 'all' key if no tissue split
            axes_for_analysis = {}
            print("Using single panel layout for trajectories.")

        fig.suptitle(f'Temporal Trajectories ({factor1} vs {factor2})', fontsize=16, fontweight='bold')

        # --- Define Plotting Function ---
        def plot_single_trajectory(ax, subset_df, group_label, color, marker, linestyle='-'):
            if subset_df.empty or 'Day' not in subset_df.columns: 
                return None, None, None
            # Group by Day and calculate centroid (mean)
            day_groups = subset_df.groupby('Day')[[factor1, factor2]].mean()
            days_sorted = sorted(day_groups.index.unique())
            if len(days_sorted) < 2: 
                return None, None, None  # Need at least 2 time points

            coords = day_groups.loc[days_sorted].values  # Get coordinates in order
            x_coords, y_coords = coords[:, 0], coords[:, 1]

            # Plot connecting lines with arrows
            for i in range(len(days_sorted) - 1):
                ax.annotate(
                    '', xy=(x_coords[i+1], y_coords[i+1]), xytext=(x_coords[i], y_coords[i]),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7, shrinkA=5, shrinkB=5)
                )

            # Plot points for each day
            for i, day in enumerate(days_sorted):
                ax.scatter(
                    x_coords[i], y_coords[i], s=120, color=color, marker=marker,
                    edgecolors='black', linewidth=1, alpha=0.9,
                    label=group_label if i == 0 else ""  # Label only first point of trajectory
                )
                # Label point with Day number
                ax.text(
                    x_coords[i], y_coords[i] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,  # Offset slightly
                    f"D{day}", ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.1', lw=0.5)
                )
            return days_sorted, x_coords, y_coords

        # --- Iterate and Plot Trajectories ---
        # Define colors/markers based on available factors
        genotype_colors = GENOTYPE_COLORS if has_genotype else {'all': 'blue'}
        treatment_markers = {0: 'o', 1: 'X'} if has_treatment else {'all': 'o'}
        tissue_linestyles = {'L': '-', 'R': '--'} if has_tissue else {'all': '-'}

        trajectory_data = {}  # To store coords for analysis plots

        # Loop through each panel (tissue or 'all')
        for tissue_key, ax in tissue_axes.items():
            if tissue_key == 'all':
                tissue_df = factors_df
                current_tissue_name = "All Tissues"
                current_linestyle = '-'
            elif tissue_key == 'all_tissues_combined':
                tissue_df = factors_df
                current_tissue_name = "All Tissues Combined"
                current_linestyle = '-'
            elif has_tissue:
                tissue_df = factors_df[factors_df['Tissue.type'] == tissue_key]
                current_tissue_name = f"{'Leaf' if tissue_key == 'L' else 'Root' if tissue_key == 'R' else tissue_key} Tissue"
                current_linestyle = tissue_linestyles.get(tissue_key, '-')
            else:  # Should not happen based on layout setup
                continue

            if tissue_df.empty:
                ax.text(0.5, 0.5, f"No data for {current_tissue_name}", transform=ax.transAxes, ha='center', va='center')
                continue

            # Determine groups to iterate over
            genotype_groups = sorted(tissue_df['Genotype'].unique()) if has_genotype else ['all']
            treatment_groups = sorted(tissue_df['Treatment'].unique()) if has_treatment else ['all']

            # Nested loops for plotting each combination
            for genotype in genotype_groups:
                for treatment in treatment_groups:
                    # --- Filter Data for the Specific Trajectory ---
                    current_subset = tissue_df.copy()
                    label_parts = []
                    if has_genotype and genotype != 'all':
                        current_subset = current_subset[current_subset['Genotype'] == genotype]
                        label_parts.append(genotype)
                    if has_treatment and treatment != 'all':
                        current_subset = current_subset[current_subset['Treatment'] == treatment]
                        label_parts.append(f"T{treatment}")

                    if current_subset.empty or current_subset['Day'].nunique() < 2:
                        continue

                    # --- Get Plotting Aesthetics ---
                    group_label = ", ".join(label_parts) if label_parts else current_tissue_name
                    color = genotype_colors.get(genotype, 'gray')
                    marker = treatment_markers.get(treatment, 'o')

                    # --- Plot the Trajectory ---
                    days, x_coords, y_coords = plot_single_trajectory(
                        ax, current_subset, group_label, color, marker, linestyle=current_linestyle
                    )

                    # --- Store Data for Analysis Plots ---
                    if days is not None:
                        traj_key = (tissue_key, genotype, treatment)
                        trajectory_data[traj_key] = {
                            'days': days,
                            'coords': list(zip(x_coords, y_coords))
                        }

            # --- Enhance Trajectory Subplot Appearance ---
            ax.set_title(current_tissue_name, fontsize=14, fontweight='bold')
            ax.set_xlabel(factor1, fontsize=12)
            ax.set_ylabel(factor2, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            if ax.has_data():  # Only add legend if something was plotted
                ax.legend(title="Group (Genotype, Treatment)", loc='best', fontsize=8, ncol=max(1, len(genotype_groups)//2))
            sns.despine(ax=ax)

        # --- Add Analysis Plots (Magnitude & Divergence) ---
        if axes_for_analysis:
            analysis_results = {'magnitude': [], 'divergence': []}

            # Calculate metrics
            for tissue_key in factors_df['Tissue.type'].unique() if has_tissue else ['all']:
                genotype_groups = sorted(factors_df['Genotype'].unique()) if has_genotype else ['all']
                treatment_groups = sorted(factors_df['Treatment'].unique()) if has_treatment else ['all']
                days_present = sorted(factors_df['Day'].unique())

                for day in days_present:
                    # Magnitude (Treatment vs Control within Genotype/Tissue)
                    if has_treatment and has_genotype and len(treatment_groups) == 2:
                        for genotype in genotype_groups:
                            key0 = (tissue_key, genotype, treatment_groups[0])  # Control
                            key1 = (tissue_key, genotype, treatment_groups[1])  # Treatment
                            if key0 in trajectory_data and key1 in trajectory_data:
                                if day in trajectory_data[key0]['days'] and day in trajectory_data[key1]['days']:
                                    idx0 = trajectory_data[key0]['days'].index(day)
                                    idx1 = trajectory_data[key1]['days'].index(day)
                                    p0 = trajectory_data[key0]['coords'][idx0]
                                    p1 = trajectory_data[key1]['coords'][idx1]
                                    dist = distance.euclidean(p0, p1)
                                    analysis_results['magnitude'].append({
                                        'Tissue': tissue_key, 'Genotype': genotype, 'Day': day, 'Response': dist
                                    })

                    # Divergence (Genotype vs Genotype within Treatment/Tissue)
                    if has_genotype and len(genotype_groups) == 2:
                        for treatment in treatment_groups:
                            key_g1 = (tissue_key, genotype_groups[0], treatment)
                            key_g2 = (tissue_key, genotype_groups[1], treatment)
                            if key_g1 in trajectory_data and key_g2 in trajectory_data:
                                if day in trajectory_data[key_g1]['days'] and day in trajectory_data[key_g2]['days']:
                                    idx_g1 = trajectory_data[key_g1]['days'].index(day)
                                    idx_g2 = trajectory_data[key_g2]['days'].index(day)
                                    p_g1 = trajectory_data[key_g1]['coords'][idx_g1]
                                    p_g2 = trajectory_data[key_g2]['coords'][idx_g2]
                                    dist = distance.euclidean(p_g1, p_g2)
                                    analysis_results['divergence'].append({
                                        'Tissue': tissue_key, 'Treatment': treatment, 'Day': day, 'Divergence': dist
                                    })

            # Plot Magnitude
            if analysis_results['magnitude'] and 'magnitude' in axes_for_analysis:
                ax_mag = axes_for_analysis['magnitude']
                mag_df = pd.DataFrame(analysis_results['magnitude'])
                # Use seaborn for potentially complex grouping
                sns.lineplot(data=mag_df, x='Day', y='Response', hue='Genotype', style='Tissue' if has_tissue else None,
                            palette=genotype_colors, markers=True, dashes=has_tissue, ax=ax_mag, err_style="bars", ci=68)  # Show SEM
                ax_mag.set_title('Treatment Response Magnitude', fontsize=12, fontweight='bold')
                ax_mag.set_ylabel('Distance(Treated - Control)', fontsize=10)
                ax_mag.set_xlabel('Day', fontsize=10)
                ax_mag.grid(True, linestyle='--', alpha=0.4)
                ax_mag.legend(title="Genotype/Tissue", fontsize=8)
                sns.despine(ax=ax_mag)

            # Plot Divergence
            if analysis_results['divergence'] and 'divergence' in axes_for_analysis:
                ax_div = axes_for_analysis['divergence']
                div_df = pd.DataFrame(analysis_results['divergence'])
                hue_order = sorted(div_df['Treatment'].unique()) if has_treatment and 'Treatment' in div_df else None
                style_order = sorted(div_df['Tissue'].unique()) if has_tissue and 'Tissue' in div_df else None

                sns.lineplot(data=div_df, x='Day', y='Divergence', hue='Treatment' if has_treatment else None, 
                             style='Tissue' if has_tissue else None,
                             palette=TREATMENT_COLORS if has_treatment else None, hue_order=hue_order, 
                             style_order=style_order,
                             markers=True, dashes=has_tissue, ax=ax_div, err_style="bars", ci=68)  # Show SEM
                ax_div.set_title('Genotype Divergence', fontsize=12, fontweight='bold')
                ax_div.set_ylabel(f'Distance({genotype_groups[0]} - {genotype_groups[1]})', fontsize=10)
                ax_div.set_xlabel('Day', fontsize=10)
                ax_div.grid(True, linestyle='--', alpha=0.4)
                ax_div.legend(title="Treatment/Tissue", fontsize=8)
                sns.despine(ax=ax_div)

        # --- Final Adjustments and Saving ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"temporal_trajectories_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"temporal_trajectories_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved temporal trajectory plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating temporal trajectory plot: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def create_cross_view_integration(data, output_dir):
    """Create a visualization showing relationships between views"""
    print("\n--- Creating cross-view integration visualization ---")

    # --- Data Validation ---
    if 'weights' not in data or not data['weights']:
        print("ERROR: Weights data not available.")
        return None
    if 'features' not in data or not data['features']:
        print("ERROR: Features data not available.")
        return None
    if 'views' not in data or len(data['views']) < 2:
        print(f"ERROR: Need at least 2 views for integration. Found: {len(data.get('views', []))}")
        return None

    weights = data['weights']
    features = data['features']
    views = data['views']
    n_views = len(views)
    # Infer n_factors from first weight matrix
    first_view = next(iter(weights.keys()))
    n_factors = weights[first_view].shape[0]

    print(f"Integrating {n_views} views across {n_factors} factors.")

    try:
        # --- Set up Figure Layout ---
        fig = plt.figure(figsize=(18, 16))  # Increased size
        # GridSpec: 2 rows (variance, network), 1 col. Variance plot smaller height.
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.3)
        ax_var = plt.subplot(gs[0])  # Variance overlap
        ax_net = plt.subplot(gs[1])  # Network
        fig.suptitle("Cross-View Integration", fontsize=18, fontweight='bold')

        # --- 1. Variance Overlap Plot (Panel A - ax_var) ---
        variance_data = None
        if 'variance' in data and 'r2_per_factor' in data['variance']:
            variance_data = data['variance']['r2_per_factor']  # Should be views x factors

        if variance_data is not None and variance_data.shape == (n_views, n_factors):
            factor_totals = np.sum(variance_data, axis=0)  # Sum variance across views for each factor
            # Sort factors by total variance explained, descending
            sorted_factor_indices = np.argsort(factor_totals)[::-1]

            # Choose top N factors for display (e.g., top 5 or all if fewer)
            top_k_factors = min(n_factors, 10)  # Show top 10 factors max
            display_indices = sorted_factor_indices[:top_k_factors]
            display_labels = [f"Factor {i+1}" for i in display_indices]

            # Prepare data for stacked bar plot
            plot_data = variance_data[:, display_indices]  # Shape: (n_views, top_k_factors)

            bar_width = 0.7
            x_pos = np.arange(top_k_factors)
            bottom = np.zeros(top_k_factors)
            view_colors = {view: VIEW_COLORS.get(view, plt.cm.viridis(i / max(1, n_views - 1))) for i, view in enumerate(views)}

            for v_idx, view in enumerate(views):
                values = plot_data[v_idx, :]
                ax_var.bar(x_pos, values, bottom=bottom, width=bar_width,
                          label=view, color=view_colors.get(view, 'gray'))
                bottom += values

            ax_var.set_title('Shared Variance Across Views (Top Factors)', fontsize=14, fontweight='bold')
            ax_var.set_ylabel('Variance Explained (%)', fontsize=10)
            ax_var.set_xticks(x_pos)
            ax_var.set_xticklabels(display_labels, rotation=45, ha='right')
            ax_var.legend(title='View', loc='upper right', fontsize=8)
            ax_var.grid(axis='y', alpha=0.3, linestyle='--')
            ax_var.set_ylim(bottom=0)
            sns.despine(ax=ax_var)
        else:
            ax_var.text(0.5, 0.5, "Variance data per factor not available or incorrect shape.",
                      ha='center', va='center', transform=ax_var.transAxes)
            ax_var.set_title('Shared Variance Across Views', fontsize=14, fontweight='bold')

        # --- 2. Cross-View Network (Panel B - ax_net) ---
        G = nx.Graph()
        top_features_per_view = {}
        MAX_FEATURES_PER_VIEW = 15  # Limit features shown per view for clarity

        # Add Factor nodes first
        factor_nodes = [f"Factor {i+1}" for i in range(n_factors)]
        for factor_name in factor_nodes:
            G.add_node(factor_name, node_type='factor', size=1200)  # Larger size for factors

        # Add Top Feature nodes and Factor-Feature edges
        max_abs_weight_overall = 0  # For scaling edge widths later
        for view in views:
            if view not in weights or view not in features: 
                continue
            view_weights_matrix = weights[view]  # factors x features
            view_feature_list = features[view]

            if len(view_feature_list) != view_weights_matrix.shape[1]:
                print(f"Warning: Feature/Weight mismatch for {view}. Skipping network contributions.")
                continue

            # Calculate feature importance (sum abs weight across factors)
            importance = np.sum(np.abs(view_weights_matrix), axis=0)
            top_indices = np.argsort(importance)[-MAX_FEATURES_PER_VIEW:]  # Indices of top features for this view
            top_features_per_view[view] = [view_feature_list[i] for i in top_indices]

            # Add nodes and edges for these top features
            for feature_idx in top_indices:
                feature_name = view_feature_list[feature_idx]
                # Add feature node
                display_name = feature_name.replace(f"_{view}", "")  # Cleaner label
                if len(display_name) > 12: 
                    display_name = display_name[:10]+".."
                G.add_node(feature_name, node_type='feature', view=view,
                          importance=importance[feature_idx], display_name=display_name, size=400)  # Smaller size

                # Add edges to factors based on weights
                for factor_idx in range(n_factors):
                    weight = view_weights_matrix[factor_idx, feature_idx]
                    # Add edge if weight is substantial (e.g., top X% or abs value > threshold)
                    # For simplicity, let's use a threshold based on the distribution of weights for that factor
                    factor_weights_for_view = view_weights_matrix[factor_idx, :]
                    weight_threshold = np.percentile(np.abs(factor_weights_for_view), 95)  # Connect top 5% weights

                    if abs(weight) > weight_threshold and abs(weight) > 0.05:  # Absolute min threshold too
                        factor_name = f"Factor {factor_idx+1}"
                        G.add_edge(feature_name, factor_name, weight=abs(weight), type='factor_feature')
                        max_abs_weight_overall = max(max_abs_weight_overall, abs(weight))

        # Optional: Add edges between features (e.g., based on co-association with factors)
        # This can make the graph very dense; consider skipping or using a stringent threshold.
        # Example: Connect features from different views if they are both strongly connected to the SAME factor
        for factor_node in factor_nodes:
            connected_features = [n for n in G.neighbors(factor_node) if G.nodes[n]['node_type'] == 'feature']
            # Group features by view
            features_by_view = {}
            for feat in connected_features:
                view = G.nodes[feat]['view']
                if view not in features_by_view: 
                    features_by_view[view] = []
                features_by_view[view].append(feat)

            # Connect top features between pairs of views
            for view1, view2 in combinations(features_by_view.keys(), 2):
                for feat1 in features_by_view[view1]:
                    for feat2 in features_by_view[view2]:
                        # Check if edge doesn't already exist
                        if not G.has_edge(feat1, feat2):
                            # Add edge with weight based on combined strength? (Simplified: fixed weight)
                            G.add_edge(feat1, feat2, weight=0.5, type='cross_view')  # Use a smaller weight

        # --- Network Layout ---
        try:
            # Refined spring layout: position factors first, then let features arrange
            fixed_nodes = factor_nodes
            pos_factors = nx.circular_layout(G.subgraph(factor_nodes))  # Arrange factors in a circle
            pos = nx.spring_layout(G, pos=pos_factors, fixed=fixed_nodes, k=0.3, iterations=100, seed=42)

        except Exception as layout_err:
            print(f"Warning: Network layout failed ({layout_err}). Falling back to random layout.")
            pos = nx.random_layout(G)

        # --- Draw Network ---
        # Node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}
        view_colors = {view: VIEW_COLORS.get(view, plt.cm.viridis(i / max(1, n_views - 1))) for i, view in enumerate(views)}

        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'factor':
                node_colors.append('lightgrey')
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = node  # Use full factor name
            elif G.nodes[node]['node_type'] == 'feature':
                view = G.nodes[node]['view']
                node_colors.append(view_colors.get(view, 'blue'))
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = G.nodes[node]['display_name']  # Use shortened name
            else:  # Fallback
                node_colors.append('red')
                node_sizes.append(200)
                node_labels[node] = node[:5]+".."

        # Edge attributes
        edge_widths = []
        edge_colors = []
        factor_feature_edges = []
        cross_view_edges = []

        for u, v, d in G.edges(data=True):
            edge_type = d.get('type', 'unknown')
            weight = d.get('weight', 0.1)
            if edge_type == 'factor_feature':
                factor_feature_edges.append((u, v))
                # Scale width by weight, normalize by max observed weight
                edge_widths.append(0.5 + 4 * (weight / max(max_abs_weight_overall, 0.1)))
                edge_colors.append('gray')
            elif edge_type == 'cross_view':
                cross_view_edges.append((u, v))
                edge_widths.append(1.0)  # Fixed thin width for cross-view
                edge_colors.append('lightblue')  # Distinct color
            else:  # Fallback for unknown edges
                factor_feature_edges.append((u, v))  # Treat as factor-feature for drawing
                edge_widths.append(0.5)
                edge_colors.append('pink')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax_net)

        # Draw edges (draw cross-view first, so factor-feature are on top)
        if cross_view_edges:
            widths_cv = [G[u][v].get('weight', 0.5) * 2 for u, v in cross_view_edges]  # Example width
            nx.draw_networkx_edges(G, pos, edgelist=cross_view_edges, width=widths_cv,
                               edge_color='dodgerblue', style='dashed', alpha=0.4, ax=ax_net)
        if factor_feature_edges:
            widths_ff = [G[u][v].get('weight', 0.1) * 5 for u, v in factor_feature_edges]  # Scale factor-feature edges
            nx.draw_networkx_edges(G, pos, edgelist=factor_feature_edges, width=widths_ff,
                               edge_color='darkgrey', style='solid', alpha=0.6, ax=ax_net)

        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax_net)

        # --- Enhance Network Plot Appearance ---
        ax_net.set_title('Cross-View Feature Integration Network (Top Features)', fontsize=14, fontweight='bold')
        ax_net.axis('off')

        # Add legend manually
        legend_handles = []
        # Views
        for view, color in view_colors.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{view} Feature',
                                          markerfacecolor=color, markersize=8))
        # Factor
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Factor',
                                       markerfacecolor='lightgrey', markersize=10))
        # Edges
        legend_handles.append(plt.Line2D([0], [0], color='darkgrey', lw=2, label='Factor-Feature Link (Weight Scaled)'))
        if cross_view_edges:
            legend_handles.append(plt.Line2D([0], [0], color='dodgerblue', lw=1.5, linestyle='--', label='Inferred Cross-View Link'))

        ax_net.legend(handles=legend_handles, loc='lower right', fontsize=9, frameon=True, facecolor='white', framealpha=0.7)

        # --- Save Plot ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved cross-view integration plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating cross-view integration visualization: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


# Feature Loadings function seems okay based on user report, keeping it mostly as is.
# Add minor checks.
def create_enhanced_feature_loadings(data, output_dir):
    """Create enhanced feature loadings visualizations"""
    print("\n--- Creating enhanced feature loadings visualizations ---")

    # --- Data Validation ---
    if 'weights' not in data or not data['weights']:
        print("ERROR: Weights data not available.")
        return None
    if 'features' not in data or not data['features']:
        print("ERROR: Features data not available.")
        return None

    weights = data['weights']
    features = data['features']
    views = data.get('views', list(weights.keys()))
    if not views:
         print("ERROR: No views found to plot loadings for.")
         return None

    files_created = []

    try:
        view_colors = {view: VIEW_COLORS.get(view, plt.cm.viridis(i / max(1, len(views) - 1))) for i, view in enumerate(views)}

        # --- 1. Multi-panel Feature Loadings Plot (Top Factors per View) ---
        n_factors_to_plot = 3 # Plot top 3 factors
        for view in views:
            if view not in weights or view not in features:
                print(f"Skipping loadings for view '{view}': Missing weights or features.")
                continue

            view_weights = weights[view] # Shape: (factors, features)
            view_features = features[view]
            actual_n_factors = view_weights.shape[0]
            plot_factors = min(n_factors_to_plot, actual_n_factors)

            if len(view_features) != view_weights.shape[1]:
                 print(f"ERROR: Feature/Weight mismatch for '{view}'. Skipping loadings plot.")
                 continue
            if plot_factors == 0:
                 print(f"Warning: No factors to plot for '{view}'.")
                 continue

            fig, axes = plt.subplots(plot_factors, 1, figsize=(12, 4 * plot_factors), squeeze=False) # Ensure axes is always 2D array

            for i in range(plot_factors):
                factor_idx = i # Plot Factor 1, Factor 2, ...
                ax = axes[i, 0] # Access subplot correctly

                factor_weights_vector = view_weights[factor_idx, :]
                abs_weights = np.abs(factor_weights_vector)
                sorted_indices = np.argsort(abs_weights)

                top_n_features = min(15, len(view_features)) # Show top 15 features
                top_indices = sorted_indices[-top_n_features:]

                top_feature_names = [view_features[j] for j in top_indices]
                top_feature_weights = factor_weights_vector[top_indices]

                # Sort by weight value for plotting
                sorted_plot_indices = np.argsort(top_feature_weights)
                plot_features = [top_feature_names[j] for j in sorted_plot_indices]
                plot_weights = top_feature_weights[sorted_plot_indices]

                # Clean feature names
                display_names = []
                for name in plot_features:
                    name = name.replace(f"_{view}", "")
                    if len(name) > 25: name = name[:22] + "..."
                    display_names.append(name)

                # --- Plotting ---
                norm = Normalize(vmin=-max(abs(plot_weights)), vmax=max(abs(plot_weights)))
                colors = [plt.cm.RdBu_r(norm(w)) for w in plot_weights]

                bars = ax.barh(range(len(display_names)), plot_weights, color=colors,
                              height=0.7, edgecolor='black', linewidth=0.5)

                # Add value labels
                max_weight_val = max(abs(plot_weights)) * 0.02 # Offset for text
                for bar, weight in zip(bars, plot_weights):
                    width = bar.get_width()
                    label_pos = width + max_weight_val if width >= 0 else width - max_weight_val
                    ha = 'left' if width >= 0 else 'right'
                    ax.text(label_pos, bar.get_y() + bar.get_height()/2., f"{weight:.2f}",
                           va='center', ha=ha, fontsize=8)

                # --- Enhance Subplot ---
                ax.set_title(f'Top Features for Factor {factor_idx+1} in {view}', fontsize=12, fontweight='bold')
                ax.set_yticks(range(len(display_names)))
                ax.set_yticklabels(display_names, fontsize=9)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3, linestyle=':')
                # Add subtle background shading
                xlims = ax.get_xlim()
                ax.axvspan(0, xlims[1], alpha=0.05, color='blue', lw=0)
                ax.axvspan(xlims[0], 0, alpha=0.05, color='red', lw=0)
                ax.set_xlim(xlims) # Reset limits after axvspan
                sns.despine(ax=ax, left=True) # Remove y-axis line

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"enhanced_loadings_{view}_{timestamp}.png")
            svg_file = os.path.join(output_dir, f"enhanced_loadings_{view}_{timestamp}.svg")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.savefig(svg_file, format='svg', bbox_inches='tight')
            files_created.append(output_file)
            plt.close(fig)
            print(f"Saved enhanced loadings for {view}, Factor {factor_idx+1}")


        # --- 2. Cross-View Top Feature Comparison (for a selected factor, e.g., Factor 1) ---
        if len(views) >= 2:
            factor_idx_compare = 0 # Compare Factor 1
            fig_comp, axes_comp = plt.subplots(len(views), 1, figsize=(12, 3 * len(views)), squeeze=False) # Ensure 2D axes
            fig_comp.suptitle(f'Cross-View Feature Comparison for Factor {factor_idx_compare+1}', fontsize=16, fontweight='bold')

            max_abs_weight_across_views = 0 # For consistent x-axis limits

            # First pass: find max weight to sync axes
            for i, view in enumerate(views):
                 if view not in weights or view not in features: continue
                 view_weights = weights[view]
                 view_features = features[view]
                 if factor_idx_compare >= view_weights.shape[0]: continue
                 if len(view_features) != view_weights.shape[1]: continue

                 factor_weights = view_weights[factor_idx_compare, :]
                 abs_weights = np.abs(factor_weights)
                 sorted_indices = np.argsort(abs_weights)
                 top_n = min(10, len(view_features))
                 top_indices = sorted_indices[-top_n:]
                 top_weights = factor_weights[top_indices]
                 if len(top_weights) > 0:
                     max_abs_weight_across_views = max(max_abs_weight_across_views, np.max(np.abs(top_weights)))


            # Second pass: plot
            for i, view in enumerate(views):
                ax = axes_comp[i, 0]
                if view not in weights or view not in features:
                     ax.text(0.5, 0.5, f"Data missing for {view}", transform=ax.transAxes, ha='center', va='center')
                     continue

                view_weights = weights[view]
                view_features = features[view]

                if factor_idx_compare >= view_weights.shape[0]:
                     ax.text(0.5, 0.5, f"Factor {factor_idx_compare+1} not available for {view}", transform=ax.transAxes, ha='center', va='center')
                     continue
                if len(view_features) != view_weights.shape[1]:
                     ax.text(0.5, 0.5, f"Feature/Weight mismatch for {view}", transform=ax.transAxes, ha='center', va='center')
                     continue


                factor_weights = view_weights[factor_idx_compare, :]
                abs_weights = np.abs(factor_weights)
                sorted_indices = np.argsort(abs_weights)
                top_n = min(10, len(view_features))
                top_indices = sorted_indices[-top_n:]

                top_feature_names = [view_features[j] for j in top_indices]
                top_feature_weights = factor_weights[top_indices]

                if len(top_feature_names) == 0:
                     ax.text(0.5, 0.5, f"No features to plot for {view}", transform=ax.transAxes, ha='center', va='center')
                     continue

                # Sort by weight value for plotting
                sorted_plot_indices = np.argsort(top_feature_weights)
                plot_features = [top_feature_names[j] for j in sorted_plot_indices]
                plot_weights = top_feature_weights[sorted_plot_indices]

                # Clean feature names
                display_names = []
                for name in plot_features:
                    name = name.replace(f"_{view}", "")
                    if len(name) > 20: name = name[:17] + "..."
                    display_names.append(name)


                # --- Plotting ---
                norm = Normalize(vmin=-max_abs_weight_across_views, vmax=max_abs_weight_across_views) # Use consistent scale
                colors = [plt.cm.RdBu_r(norm(w)) for w in plot_weights]

                bars = ax.barh(range(len(display_names)), plot_weights, color=colors,
                              height=0.7, edgecolor='black', linewidth=0.5)


                # --- Enhance Subplot ---
                ax.set_title(f'{view} Features (Factor {factor_idx_compare+1})', fontsize=12, fontweight='bold', color=view_colors.get(view, 'black'))
                ax.set_yticks(range(len(display_names)))
                ax.set_yticklabels(display_names, fontsize=9)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3, linestyle=':')
                # Set consistent x-limits
                ax.set_xlim(-max_abs_weight_across_views*1.1, max_abs_weight_across_views*1.1)
                sns.despine(ax=ax, left=True)

            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"cross_view_feature_comparison_{timestamp}.png")
            svg_file = os.path.join(output_dir, f"cross_view_feature_comparison_{timestamp}.svg")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.savefig(svg_file, format='svg', bbox_inches='tight')
            files_created.append(output_file)
            plt.close(fig_comp)
            print(f"Saved cross-view feature comparison (Factor {factor_idx_compare+1})")

        return files_created # Indicate success by returning list of files

    except Exception as e:
        print(f"ERROR creating enhanced feature loadings: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def create_genotype_response_network(data, output_dir):
    """Create a network visualization showing differential response between genotypes"""
    print("\n--- Creating genotype response network visualization ---")

    # --- Data Validation ---
    if 'factors_df' not in data or data['factors_df'].empty:
        print("ERROR: factors_df not available.")
        return None
    if 'feature_importance' not in data or not data['feature_importance']:
        print("ERROR: feature_importance data not available.")
        return None
    if 'Genotype' not in data['factors_df'].columns or 'Day' not in data['factors_df'].columns:
        print("ERROR: Genotype or Day column missing in factors_df.")
        return None

    factors_df = data['factors_df']
    feature_importance = data['feature_importance']
    views = data.get('views', list(feature_importance.keys()))

    genotypes = sorted(factors_df['Genotype'].unique())
    if len(genotypes) < 2:
        print("Warning: Need at least 2 genotypes for comparison plot. Found only 1.")
        # Proceed with single genotype plot? Or skip? Skipping for now.
        return None

    days = sorted(factors_df['Day'].unique())
    if len(days) < 1:
         print("ERROR: No days found in data.")
         return None

    print(f"Plotting genotype network for {genotypes} across days {days}.")

    try:
        # --- Set up Figure Layout ---
        fig = plt.figure(figsize=(20, 14)) # Slightly larger
        # GridSpec: 2 rows (network+temporal, diff_features), 2 cols in first row
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
        ax_net = plt.subplot(gs[0, 0]) # Genotype response network
        ax_temp = plt.subplot(gs[0, 1]) # Temporal response strength
        ax_diff = plt.subplot(gs[1, :]) # Top differential features (across row 1)
        fig.suptitle('Genotype-Specific Response Network & Features', fontsize=18, fontweight='bold')


        # --- 1. Prepare Network Data ---
        G = nx.Graph()
        MAX_FEATURES_PER_VIEW_NET = 7 # Limit features in network plot

        # Add Genotype nodes
        for i, genotype in enumerate(genotypes):
            G.add_node(genotype, node_type='genotype', size=2500, color=GENOTYPE_COLORS.get(genotype, 'grey'))

        # Add Day nodes
        for i, day in enumerate(days):
            G.add_node(f"Day {day}", node_type='day', size=1800, color='lightcoral')

        # Add Top Feature nodes (based on overall importance)
        network_features = {} # Store features added to network {feature_name: {view, display_name}}
        for view in views:
            if view not in feature_importance: continue
            view_imp = feature_importance[view].head(MAX_FEATURES_PER_VIEW_NET)
            for idx, row in view_imp.iterrows():
                feature = row['Feature']
                if feature not in G: # Add only once
                     display_name = feature.replace(f"_{view}","")
                     if len(display_name) > 12: display_name = display_name[:10]+".."
                     G.add_node(feature, node_type='feature', view=view,
                                 display_name=display_name, size=800, color=VIEW_COLORS.get(view, 'blue'))
                     network_features[feature] = {'view': view, 'display_name': display_name}


        # Calculate genotype-specific temporal factor patterns (using top 3 factors)
        factor_cols = [col for col in factors_df.columns if col.startswith('Factor')]
        top_factors_for_edges = factor_cols[:min(3, len(factor_cols))] # Use top 3 factors for edge weight
        temporal_patterns = {} # {genotype: {day: {factor: mean_score}}}

        if top_factors_for_edges:
             for genotype in genotypes:
                 temporal_patterns[genotype] = {}
                 for day in days:
                     subset = factors_df[(factors_df['Genotype'] == genotype) & (factors_df['Day'] == day)]
                     if not subset.empty:
                          temporal_patterns[genotype][day] = subset[top_factors_for_edges].mean().to_dict()

        # Add Genotype-Day edges (Weight = norm of factor means)
        max_gen_day_weight = 0
        genotype_day_edges = []
        for genotype in genotypes:
            for day in days:
                if genotype in temporal_patterns and day in temporal_patterns[genotype]:
                    factor_means = temporal_patterns[genotype][day]
                    # Calculate Euclidean norm of the factor vector for this day/genotype
                    edge_weight = np.sqrt(sum(v**2 for v in factor_means.values()))
                    if np.isnan(edge_weight): edge_weight = 0
                    max_gen_day_weight = max(max_gen_day_weight, edge_weight)
                    day_node = f"Day {day}"
                    if G.has_node(genotype) and G.has_node(day_node):
                         G.add_edge(genotype, day_node, weight=edge_weight, type='genotype_day')
                         genotype_day_edges.append((genotype, day_node)) # Keep track

        # Add Day-Feature edges (Placeholder: connect feature if it's important on that day?)
        # For now, connect based on simple logic: feature connected to day if its importance changes significantly by that day?
        # Simplified: Connect each feature node to Day 1 and Day 3 for illustration.
        day_feature_edges = []
        day1_node = "Day 1"
        day3_node = f"Day {days[-1]}" if days else "Day 3" # Use last day
        for feature_node in network_features:
             if G.has_node(feature_node):
                  if G.has_node(day1_node):
                       G.add_edge(feature_node, day1_node, weight=0.5, type='day_feature')
                       day_feature_edges.append((feature_node, day1_node))
                  if G.has_node(day3_node):
                       G.add_edge(feature_node, day3_node, weight=0.8, type='day_feature') # Stronger later?
                       day_feature_edges.append((feature_node, day3_node))


        # --- Network Layout (Fixed positions) ---
        pos = {}
        genotype_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'genotype']
        day_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'day']
        feature_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'feature']

        # Genotypes on left
        for i, node in enumerate(genotype_nodes): pos[node] = (-2, i * 1.5 - (len(genotype_nodes)-1)*0.75)
        # Days in middle
        for i, node in enumerate(day_nodes): pos[node] = (0, i * 1.5 - (len(day_nodes)-1)*0.75)
        # Features on right, grouped by view
        views_present = sorted(list(set(d['view'] for n, d in G.nodes(data=True) if d['node_type'] == 'feature')))
        current_y = (len(feature_nodes) - 1) * -0.5 # Center features vertically
        x_offset = 2
        for view_idx, view in enumerate(views_present):
             view_feature_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'feature' and d['view'] == view]
             for i, node in enumerate(view_feature_nodes):
                  pos[node] = (x_offset + view_idx*0.5, current_y + i * 1.0) # Stagger x by view, spread y
             current_y += (len(view_feature_nodes) -1) * 1.0 + 1.5 # Add spacing between view groups

        # --- Draw Network (Panel A - ax_net) ---
        # Node attributes
        node_colors = [d.get('color', 'grey') for n, d in G.nodes(data=True)]
        node_sizes = [d.get('size', 500) for n, d in G.nodes(data=True)]
        # Edge attributes & drawing
        edge_colors = []
        edge_widths = []
        # Genotype-Day Edges
        if genotype_day_edges:
             g_d_widths = [G[u][v]['weight'] / max(max_gen_day_weight, 0.1) * 4 + 0.5 for u,v in genotype_day_edges] # Normalize and scale
             g_d_colors = [G.nodes[u]['color'] if G.nodes[u]['node_type']=='genotype' else G.nodes[v]['color'] for u,v in genotype_day_edges]
             nx.draw_networkx_edges(G, pos, edgelist=genotype_day_edges, width=g_d_widths, edge_color=g_d_colors, alpha=0.6, ax=ax_net)
        # Day-Feature Edges
        if day_feature_edges:
             d_f_widths = [G[u][v]['weight'] * 1.5 for u,v in day_feature_edges] # Simple weight scaling
             d_f_colors = [G.nodes[u]['color'] if G.nodes[u]['node_type']=='feature' else G.nodes[v]['color'] for u,v in day_feature_edges]
             nx.draw_networkx_edges(G, pos, edgelist=day_feature_edges, width=d_f_widths, edge_color=d_f_colors, alpha=0.3, style='dashed', ax=ax_net)
        # Draw nodes on top
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='grey', linewidths=0.5, ax=ax_net)
        # Labels
        labels = {n: (d.get('display_name', n) if d.get('node_type') == 'feature' else n) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax_net)

        ax_net.set_title('Genotype Response Network (Simplified)', fontsize=14, fontweight='bold')
        
        # --- Add Legend to Network ---
        legend_handles = []
        # Genotype nodes
        for genotype in genotypes:
             color = GENOTYPE_COLORS.get(genotype, 'grey')
             legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Genotype {genotype}',
                                             markersize=10, markerfacecolor=color))
        # Day nodes
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Day',
                                         markersize=10, markerfacecolor='lightcoral')) # Use Day node color
        # Feature nodes by view
        network_views = sorted(list(set(d['view'] for n, d in G.nodes(data=True) if d['node_type'] == 'feature')))
        for view in network_views:
             color = VIEW_COLORS.get(view, 'blue')
             legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{view} Feature',
                                             markersize=8, markerfacecolor=color)) # Smaller marker for features

        # Add the legend to the axis
        ax_net.legend(handles=legend_handles, title="Node Types", loc='upper right', fontsize=9,
                     bbox_to_anchor=(1.0, 1.0), frameon=False) # Position top right

        ax_net.axis('off') # Turn off axis AFTER legend is created based on its elements

        # --- 2. Temporal Response Strength (Panel B - ax_temp) ---
        response_data = []
        if temporal_patterns:
             for genotype in genotypes:
                 for day in days:
                     if genotype in temporal_patterns and day in temporal_patterns[genotype]:
                          factor_means = temporal_patterns[genotype][day]
                          response = np.sqrt(sum(v**2 for v in factor_means.values()))
                          response_data.append({'Genotype': genotype, 'Day': day, 'Response': response})

        if response_data:
             response_df = pd.DataFrame(response_data)
             sns.lineplot(data=response_df, x='Day', y='Response', hue='Genotype',
                         palette=GENOTYPE_COLORS, marker='o', ax=ax_temp)
             ax_temp.set_title('Temporal Response Strength', fontsize=12, fontweight='bold')
             ax_temp.set_xlabel('Day', fontsize=10)
             ax_temp.set_ylabel('Response Strength (Factor Norm)', fontsize=10)
             ax_temp.grid(True, linestyle='--', alpha=0.4)
             ax_temp.legend(title="Genotype", fontsize=8)
             sns.despine(ax=ax_temp)


        # --- 3. Top Differential Features (Panel C - ax_diff) ---
        diff_features_list = [] # Renamed to avoid confusion with the dict from previous scope
        N_DIFF_FEATURES = 20
        # Collect top features from each view's importance data
        for view, importance_df in feature_importance.items():
             if not importance_df.empty:
                  # Add view information to each feature record
                  view_top_features = importance_df.head(N_DIFF_FEATURES).copy()
                  view_top_features['View'] = view
                  diff_features_list.extend(view_top_features.to_dict('records'))

        if diff_features_list:
             # Create DataFrame from the list of dictionaries
             diff_df = pd.DataFrame(diff_features_list)
             # Ensure Importance is numeric
             diff_df['Importance'] = pd.to_numeric(diff_df['Importance'], errors='coerce')
             diff_df.dropna(subset=['Importance'], inplace=True)

             # Get unique features, keeping the one with highest importance if duplicated across views
             diff_df = diff_df.sort_values('Importance', ascending=False)
             diff_df = diff_df.drop_duplicates(subset=['Feature'], keep='first')

             # Select top N overall
             diff_df = diff_df.head(N_DIFF_FEATURES)
             # Sort again for plotting (most important at top)
             diff_df = diff_df.sort_values('Importance', ascending=True) # Ascending for barh bottom-to-top

             # Clean names and add colors for plotting
             diff_df['Display'] = diff_df['Feature'].apply(lambda x: x.split('_')[0]) # Basic cleaning
             # Further split if it contains 'Cluster'
             diff_df['Display'] = diff_df['Display'].apply(lambda x: x.replace('Cluster', 'Clst') if 'Cluster' in x else x)
             diff_df['Display'] = diff_df['Display'].apply(lambda x: x[:15] + '..' if len(x) > 18 else x)
             diff_df['Color'] = diff_df['View'].map(VIEW_COLORS)

             # Check if we have data to plot
             if not diff_df.empty:
                  # Plot horizontal bars
                  bars = ax_diff.barh(diff_df['Display'], diff_df['Importance'], color=diff_df['Color'].fillna('grey'), height=0.7)
                  # ax_diff.invert_yaxis() # Already sorted ascending, so highest importance is at the top

                  # Add view labels next to bars
                  max_importance_val = diff_df['Importance'].max()
                  for i, (bar, view) in enumerate(zip(bars, diff_df['View'])):
                       # Place label consistently outside the bar
                       ax_diff.text(max_importance_val * 1.02, # Position slightly right of max bar end
                                    bar.get_y() + bar.get_height()/2,
                                    f"({view})",
                                    va='center', ha='left', fontsize=8, color='dimgrey')

                  ax_diff.set_title(f'Top {min(N_DIFF_FEATURES, len(diff_df))} Features by Overall Importance', fontsize=12, fontweight='bold')
                  ax_diff.set_xlabel('Overall Importance (Sum Abs Weights)', fontsize=10)
                  # Ensure y-ticks match the number of bars plotted
                  ax_diff.set_yticks(range(len(diff_df)))
                  ax_diff.set_yticklabels(diff_df['Display']) # Use the display names
                  ax_diff.tick_params(axis='y', labelsize=9)
                  ax_diff.grid(axis='x', alpha=0.3, linestyle=':')
                  # Adjust xlim based on data range
                  ax_diff.set_xlim(left=0, right=max_importance_val * 1.15) # Start at 0, add padding
                  sns.despine(ax=ax_diff, left=True, bottom=False) # Keep bottom axis
             else:
                  ax_diff.text(0.5, 0.5, "No differential features to display.", transform=ax_diff.transAxes, ha='center', va='center')
                  ax_diff.set_title('Top Features by Overall Importance', fontsize=12, fontweight='bold')
                  sns.despine(ax=ax_diff, left=True, bottom=True)

        else:
             ax_diff.text(0.5, 0.5, "Feature importance data missing.", transform=ax_diff.transAxes, ha='center', va='center')
             ax_diff.set_title('Top Features by Overall Importance', fontsize=12, fontweight='bold')
             sns.despine(ax=ax_diff, left=True, bottom=True)

        # --- Save Plot ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"genotype_response_network_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"genotype_response_network_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved genotype response network plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating genotype response network: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


# 3D Plot function seems okay based on user report. Add minor checks.
def create_3d_factor_space(data, output_dir):
    """Create 3D visualization of samples in factor space"""
    print("\n--- Creating 3D factor space visualization ---")

    # --- Data Validation ---
    if 'factors_df' not in data or data['factors_df'].empty:
        print("ERROR: factors_df not available.")
        return None

    factors_df = data['factors_df']
    factor_cols = [col for col in factors_df.columns if col.startswith('Factor')]
    if len(factor_cols) < 3:
        print(f"ERROR: Need at least 3 factors for 3D plot. Found: {len(factor_cols)}")
        return None

    top_factors = factor_cols[:3] # Use Factor1, Factor2, Factor3
    print(f"Using factors: {top_factors}")

    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Determine coloring/markers based on available metadata
        color_by = None
        marker_by = None
        color_dict = None
        marker_dict = None
        legend_title = "Group"

        if 'Genotype' in factors_df.columns:
            color_by = 'Genotype'
            color_dict = GENOTYPE_COLORS
            legend_title = 'Genotype'
            if 'Treatment' in factors_df.columns:
                marker_by = 'Treatment'
                marker_dict = {0: 'o', 1: 'X'} # Circle=Control, X=Treated
                legend_title += ' / Treatment (marker)'
            elif 'Tissue.type' in factors_df.columns:
                 marker_by = 'Tissue.type'
                 marker_dict = {'L':'^', 'R':'s'} # Triangle=Leaf, Square=Root
                 legend_title += ' / Tissue (marker)'
        elif 'Treatment' in factors_df.columns:
            color_by = 'Treatment'
            color_dict = TREATMENT_COLORS
            legend_title = 'Treatment'
            if 'Tissue.type' in factors_df.columns:
                 marker_by = 'Tissue.type'
                 marker_dict = {'L':'^', 'R':'s'}
                 legend_title += ' / Tissue (marker)'
        elif 'Tissue.type' in factors_df.columns:
            color_by = 'Tissue.type'
            color_dict = TISSUE_COLORS
            legend_title = 'Tissue Type'

        print(f"Coloring by: {color_by}, Marking by: {marker_by}")

        # --- Plotting Points ---
        if color_by:
            handles = [] # For custom legend
            labels = []
            # Iterate through unique color categories
            for color_cat in sorted(factors_df[color_by].unique()):
                 color_subset = factors_df[factors_df[color_by] == color_cat]
                 color = color_dict.get(color_cat, 'grey')

                 if marker_by:
                      # Iterate through unique marker categories
                      for marker_cat in sorted(color_subset[marker_by].unique()):
                           subset = color_subset[color_subset[marker_by] == marker_cat]
                           marker = marker_dict.get(marker_cat, 'd') # Default diamond
                           label = f"{color_cat} / {marker_cat}"
                           if label not in labels: # Add unique labels to legend
                                labels.append(label)
                                handles.append(plt.Line2D([0], [0], marker=marker, color='w', label=label,
                                                          markerfacecolor=color, markersize=8))
                           ax.scatter(subset[top_factors[0]], subset[top_factors[1]], subset[top_factors[2]],
                                      c=[color], marker=marker, s=60, alpha=0.6, edgecolors='w', linewidth=0.5) # Added white edge
                 else:
                      # Only color category
                      label = f"{color_cat}"
                      if label not in labels:
                           labels.append(label)
                           handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                                     markerfacecolor=color, markersize=8))
                      ax.scatter(color_subset[top_factors[0]], color_subset[top_factors[1]], color_subset[top_factors[2]],
                                 c=[color], marker='o', s=60, alpha=0.7, edgecolors='w', linewidth=0.5)
        else:
            # No grouping, plot all points in default color
            ax.scatter(factors_df[top_factors[0]], factors_df[top_factors[1]], factors_df[top_factors[2]],
                       c='dodgerblue', s=50, alpha=0.6)


        # --- Enhance Plot Appearance ---
        ax.set_xlabel(top_factors[0], fontsize=11, labelpad=10)
        ax.set_ylabel(top_factors[1], fontsize=11, labelpad=10)
        ax.set_zlabel(top_factors[2], fontsize=11, labelpad=10)
        ax.set_title(f'Sample Distribution in 3D Factor Space ({", ".join(top_factors)})', fontsize=14, fontweight='bold')

        # Add legend if handles were created
        if handles:
            ax.legend(handles=handles, title=legend_title, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)

        # Improve view angle and grid
        ax.view_init(elev=25, azim=-50)
        ax.grid(True, linestyle=':', alpha=0.3)
        # Set background pane colors for better depth perception
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.2))


        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend

        # --- Save Plot ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"3d_factor_space_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"3d_factor_space_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved 3D factor space visualization to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating 3D factor space visualization: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def main():
    """Main function to run the visualization"""
    parser = argparse.ArgumentParser(description="Enhanced MOFA+ Visualization Script")
    parser.add_argument("mofa_file", help="Path to MOFA+ model HDF5 file")
    parser.add_argument("metadata_file", help="Path to metadata CSV file")
    parser.add_argument("--output", default="mofa_visualizations", help="Output directory (default: mofa_visualizations)")
    # Add flags for specific plots
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--variance", action="store_true", help="Generate variance explained plot")
    parser.add_argument("--heatmap", action="store_true", help="Generate factor-condition heatmap")
    parser.add_argument("--trajectory", action="store_true", help="Generate temporal trajectory plot")
    parser.add_argument("--integration", action="store_true", help="Generate cross-view integration network")
    parser.add_argument("--loadings", action="store_true", help="Generate feature loadings visualizations")
    parser.add_argument("--genotype_net", action="store_true", help="Generate genotype response network")
    parser.add_argument("--plot3d", action="store_true", help="Generate 3D factor space visualization")
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output, exist_ok=True)
    print("=" * 80)
    print(f"Enhanced MOFA+ Visualization Script - START")
    print(f"MOFA+ model: {args.mofa_file}")
    print(f"Metadata: {args.metadata_file}")
    print(f"Output directory: {args.output}")
    print("=" * 80)

    # --- Load Data ---
    data = load_data(args.mofa_file, args.metadata_file)
    if data is None:
        print("\nFATAL ERROR: Data loading failed. Exiting.")
        return 1  # Exit with error code

    # --- Generate Visualizations ---
    visualizations_status = {}  # Track success/failure

    # Determine which plots to run
    run_all = args.all or not any([args.variance, args.heatmap, args.trajectory,
                                 args.integration, args.loadings, args.genotype_net, args.plot3d])

    if run_all or args.variance:
        visualizations_status["Enhanced Variance Plot"] = create_enhanced_variance_plot(data, args.output)

    if run_all or args.heatmap:
        visualizations_status["Factor-Condition Heatmap"] = create_factor_condition_heatmap(data, args.output)

    if run_all or args.trajectory:
        visualizations_status["Temporal Trajectory Plot"] = create_temporal_trajectory_plot(data, args.output)

    if run_all or args.integration:
        visualizations_status["Cross-View Integration"] = create_cross_view_integration(data, args.output)

    if run_all or args.loadings:
        # This function returns a list of files or None
        loadings_result = create_enhanced_feature_loadings(data, args.output)
        visualizations_status["Feature Loadings"] = loadings_result is not None  # Mark success if it didn't return None

    if run_all or args.genotype_net:
        visualizations_status["Genotype Response Network"] = create_genotype_response_network(data, args.output)

    if run_all or args.plot3d:
        visualizations_status["3D Factor Space"] = create_3d_factor_space(data, args.output)

    # --- Print Summary ---
    print("\n" + "=" * 80)
    print("Visualization Generation Summary:")
    all_successful = True
    for name, result in visualizations_status.items():
        # For loadings, result is True/False. For others, it's filename or None.
        success = result if isinstance(result, bool) else result is not None
        status_icon = "✅" if success else "❌"
        print(f"  {status_icon} {name}")
        if not success:
            all_successful = False

    print("-" * 80)
    if all_successful:
        print("All requested visualizations generated successfully.")
    else:
        print("Some visualizations failed. Please check the logs above for errors.")
    print(f"Output directory: {args.output}")
    print("=" * 80)

    return 0 if all_successful else 1  # Return 0 on full success, 1 otherwise


if __name__ == "__main__":
    sys.exit(main())