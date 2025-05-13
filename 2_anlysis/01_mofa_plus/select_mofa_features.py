#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOFA+ Feature Selection Script

This script performs Multi-Omics Factor Analysis (MOFA+) feature selection,
targeting the top 50 features from each data modality. It loads a pre-trained
MOFA+ model, extracts weights and factors, and selects features based on their
importance across biologically relevant factors. The script generates input files
for downstream analysis with the selected features.

Features:
- Loads pre-trained MOFA+ model
- Performs factor-metadata association analysis
- Selects features using a hybrid stratified approach
- Generates transformer-ready input files with selected features
- Optional cross-validation to verify biological signal preservation

Dependencies: pandas, numpy, h5py, mofapy2, matplotlib, seaborn, scikit-learn, scipy
"""

import os
import json
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu

# Machine learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Ensure mofapy2 is installed
try:
    from mofapy2.run.entry_point import entry_point
except ImportError:
    print("ERROR: mofapy2 not found. Please install it: pip install mofapy2")
    exit()

# Ensure statsmodels is installed
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("ERROR: statsmodels not found. Please install it: pip install statsmodels")
    exit()

print("MOFA+ Analysis Script - MODE: Feature Selection Only (50 Features)")
print("="*80)
start_time = time.time()

# --- Configuration Section ---
print("1. Configuring paths and parameters...")
config = {
    # Paths
    "original_output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa",
    "output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa/mofa50",
    "figure_dir": r"C:/Users/ms/Desktop/hyper/output/mofa/mofa50/figures",
    "data_paths": {
        "leaf_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv",
        "root_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv",
        "leaf_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv",
        "root_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv",
    },
    "metadata_file": r"C:/Users/ms/Desktop/hyper/output/mofa/all_metadata.json",
    "mapping_file": r"C:/Users/ms/Desktop/hyper/output/mofa/row_name_mapping.tsv",
    "existing_model_path": r"C:/Users/ms/Desktop/hyper/output/mofa/mofa_model_for_transformer.hdf5",

    # Execution Control
    "load_existing_model": True,
    # Optional Skipping
    "skip_visualizations": False,
    "skip_hypothetical_outputs": False,
    "skip_cv_check": False,

    # MOFA+ Model Parameters
    "view_names": ["leaf_spectral", "root_spectral", "leaf_metabolite", "root_metabolite"],
    "groups_names": ["group0"],
    "num_factors": 20,
    "maxiter": 1000,
    "convergence_mode": "medium",
    "drop_factor_threshold": 0.01,

    # Feature Selection & Output Parameters
    "fdr_alpha": 0.05,
    "transformer_feature_cap": 75,
    "feature_suffix": "_50feat",
    "transformer_context_percentages": {"Genotype": 0.35, "Time": 0.35, "Other": 0.30},
    "min_variance_explained_proxy_report": 50.0,
    "relevant_factor_contexts": [
        'Genotype_Difference',
        'Day_Correlation',
        'Treatment_Correlation',
        'Batch_Correlation'
    ],
    "mapping_column_names": {
        "leaf_spectral": "hyper_l_w_augment",
        "root_spectral": "hyper_r_w_augment",
        "leaf_metabolite": "n_p_l2_augmt",
        "root_metabolite": "n_p_r2_augmt"
    },
    "reference_view": "leaf_spectral"
}

# Validate configuration
if not os.path.exists(config["existing_model_path"]) and config["load_existing_model"]:
    print(f"ERROR: load_existing_model is True, but specified file not found: "
          f"{config['existing_model_path']}")
    exit()
if not config["load_existing_model"]:
    print("WARNING: load_existing_model is False. Script will attempt to TRAIN a new model.")
    # Reset outfile name if training a new model
    config["outfile"] = f"mofa_model_new_train_{int(time.time())}.hdf5"
else:
    config["outfile"] = os.path.basename(config["existing_model_path"])

# Sanity check context percentages
total_perc = sum(config["transformer_context_percentages"].values())
if not np.isclose(total_perc, 1.0):
    print(f"ERROR: transformer_context_percentages must sum to 1.0 "
          f"(sums to {total_perc}). Adjusting 'Other'.")
    current_other = config["transformer_context_percentages"].get('Other', 0)
    config["transformer_context_percentages"]['Other'] = max(0, current_other + (1.0 - total_perc))
    print(f"       Adjusted percentages: {config['transformer_context_percentages']}")
    new_total = sum(config["transformer_context_percentages"].values())
    if not np.isclose(new_total, 1.0):
        print("       ERROR: Could not adjust percentages.")
        exit()

# Create output directories
os.makedirs(config["output_dir"], exist_ok=True)
os.makedirs(config["figure_dir"], exist_ok=True)
print(f"Using Output directory: {config['output_dir']}")
print(f"Using Figure directory: {config['figure_dir']}")
print(f"Targeting up to {config['transformer_feature_cap']} features per view "
      f"using suffix '{config['feature_suffix']}'.")

# Load Mapping File
print("\n1.5 Loading Row Name Mapping File...")
mapping_file_path = config["mapping_file"]
if not os.path.exists(mapping_file_path):
    print(f"   ERROR: Mapping file not found at: {mapping_file_path}")
    exit()
try:
    df_mapping = pd.read_csv(mapping_file_path, sep='\t')
    expected_map_cols = list(config["mapping_column_names"].values())
    missing_map_cols = [col for col in expected_map_cols if col not in df_mapping.columns]
    if missing_map_cols:
        print(f"   ERROR: Mapping file is missing expected columns: {missing_map_cols}")
        exit()
    print(f"   Successfully loaded mapping file with {df_mapping.shape[0]} rows "
          f"and columns: {list(df_mapping.columns)}")
    reference_map_col = config["mapping_column_names"][config["reference_view"]]
    print(f"   Using mapping column '{reference_map_col}' as the reference for master row names.")
except Exception as e:
    print(f"   ERROR loading or parsing mapping file '{mapping_file_path}': {e}")
    traceback.print_exc()
    exit()

# --- Section 2: Load Metadata Definitions --- (Unchanged from v5d) ---
print("\n2. Loading metadata definitions...")
try:
    with open(config["metadata_file"], 'r') as f:
        all_metadata = json.load(f)
    metadata_columns = all_metadata['datasets']['leaf_spectral']['metadata_columns']
    print(f"Identified {len(metadata_columns)} metadata columns: {metadata_columns}")
    print(f"Aligning data using mapping file.")
except Exception as e:
    print(f"ERROR loading/parsing metadata: {e}")
    traceback.print_exc()
    exit()
# --- End Section 2 ---

# --- Section 3: Load and Prepare Data --- (Unchanged from v5d - needed for Section 10.4) ---
# This section loads the *original* data to select features from for the transformer inputs
print("\n3. Loading and preparing original data...")
data_views_input_format = []
metadata_dfs_list = []
feature_names_dict = {}
master_reference_row_names = None
expected_num_rows = None
reference_view_name = config["reference_view"]
reference_map_col = config["mapping_column_names"][reference_view_name]
master_index_col_name = 'MasterIndex'

# Loop through views to establish master order and feature names
for i, view_name in enumerate(config["view_names"]):
    print(f"   - Pre-processing view (for feature names/master list): {view_name}")
    file_path = config["data_paths"].get(view_name)
    current_map_col = config["mapping_column_names"][view_name]
    if not file_path or not os.path.exists(file_path):
        print(f"   ERROR: Data file not found: {file_path}")
        exit()

    try:
        df = pd.read_csv(file_path)
        if 'Row_names' not in df.columns:
            print(f"   ERROR: 'Row_names' column not found in '{view_name}'.")
            exit()

        if expected_num_rows is None:
            expected_num_rows = df.shape[0]
            if len(df_mapping) != expected_num_rows:
                print(f"   ERROR: Row count mismatch between first data file ({expected_num_rows}) "
                      f"and mapping file ({len(df_mapping)}).")
                exit()
        elif df.shape[0] != expected_num_rows:
            print(f"   ERROR: Row count mismatch ({df.shape[0]} vs {expected_num_rows}) "
                  f"in '{view_name}'.")
            exit()

        if view_name == reference_view_name:
            df['Row_names'] = df['Row_names'].astype(str)
            if df['Row_names'].duplicated().any():
                print(f"   ERROR: Duplicate 'Row_names' in reference file '{view_name}'.")
                exit()
            df.rename(columns={'Row_names': master_index_col_name}, inplace=True)
            df.set_index(master_index_col_name, inplace=True)
            master_reference_row_names = df.index.tolist()  # Get master order
            print(f"     Established master reference row names "
                  f"({len(master_reference_row_names)} unique) from '{view_name}'.")
        else:
            # Logic is handled below more robustly
            pass

        # Extract features (ensure index is MasterIndex if ref view, or Row_names otherwise)
        meta_cols_to_drop = [c for c in metadata_columns if c != 'Row_names' and c in df.columns]
        feature_df = df.drop(columns=meta_cols_to_drop, errors='ignore')
        # Exclude index/key cols
        original_feature_names = [col for col in feature_df.columns 
                                 if col not in [master_index_col_name, 'Row_names']]
        # Store suffixed names
        feature_names_dict[view_name] = [f"{feat}_{view_name}" for feat in original_feature_names]

    except Exception as e:
        print(f"   ERROR pre-processing view '{view_name}': {e}")
        traceback.print_exc()
        exit()

if master_reference_row_names is None:
    print("ERROR: Master reference row names not established from reference view.")
    exit()
print(f"   Finished pre-processing all views. "
      f"Master Reference ID count: {len(master_reference_row_names)}")
# --- End Section 3 ---


# --- Section 4: Combine Metadata --- (Unchanged from v5d - uses master ref list now) ---
print("\n4. Combining metadata from all views...")
try:
    # Reload original data files AGAIN, but this time align properly using map and master list
    metadata_dfs_list = []  # Reset list
    for i, view_name in enumerate(config["view_names"]):
        file_path = config["data_paths"].get(view_name)
        current_map_col = config["mapping_column_names"][view_name]
        df = pd.read_csv(file_path)
        df['Row_names'] = df['Row_names'].astype(str)

        if view_name == reference_view_name:
            df.rename(columns={'Row_names': master_index_col_name}, inplace=True)
            df.set_index(master_index_col_name, inplace=True)
            df_processed = df
        else:
            df_mapping[current_map_col] = df_mapping[current_map_col].astype(str)
            df_mapping[reference_map_col] = df_mapping[reference_map_col].astype(str)
            map_subset = df_mapping[[current_map_col, reference_map_col]].drop_duplicates()
            df_merged = pd.merge(df, map_subset, left_on='Row_names', right_on=current_map_col, how='left')
            if df_merged[reference_map_col].isnull().any():
                print(f"   ERROR: Merge failed for metadata alignment in '{view_name}'.")
                exit()
            df_merged.rename(columns={reference_map_col: master_index_col_name}, inplace=True)
            df_merged.set_index(master_index_col_name, inplace=True)
            df_processed = df_merged

        # Extract Metadata and reindex to master order
        meta_df = df_processed[[col for col in metadata_columns if col in df_processed.columns]].copy()
        meta_df = meta_df.reindex(master_reference_row_names)  # Align using master list
        # Add suffix (index is already consistent MasterIndex)
        meta_df_renamed = meta_df.add_suffix(f'_{view_name}' if i > 0 else '')
        metadata_dfs_list.append(meta_df_renamed)

    # Concatenate along columns - index should align now (MasterIndex)
    combined_metadata_df_raw = pd.concat(metadata_dfs_list, axis=1)

    # Resolve suffixed columns
    final_metadata_cols_data = {}
    processed_cols = set()
    for col in metadata_columns:
        if col in processed_cols:
            continue
        if col in combined_metadata_df_raw.columns and not combined_metadata_df_raw[col].isnull().all():
            final_metadata_cols_data[col] = combined_metadata_df_raw[col]
            processed_cols.add(col)
            for i_suffix, vn_suffix in enumerate(config["view_names"]):
                suffixed_col_to_remove = f"{col}_{vn_suffix}" if i_suffix > 0 else col
                if suffixed_col_to_remove != col and suffixed_col_to_remove in combined_metadata_df_raw.columns:
                    processed_cols.add(suffixed_col_to_remove)
        else:
            found = False
            for i, vn in enumerate(config["view_names"]):
                suffixed_col = f"{col}_{vn}" if i > 0 else col
                if (suffixed_col in combined_metadata_df_raw.columns and 
                        not combined_metadata_df_raw[suffixed_col].isnull().all()):
                    final_metadata_cols_data[col] = combined_metadata_df_raw[suffixed_col]
                    processed_cols.add(col)
                    processed_cols.add(suffixed_col)
                    found = True
                    for i_suffix_rem, vn_suffix_rem in enumerate(config["view_names"]):
                        other_suffixed_col = f"{col}_{vn_suffix_rem}" if i_suffix_rem > 0 else col
                        if other_suffixed_col != suffixed_col and other_suffixed_col in combined_metadata_df_raw.columns:
                            processed_cols.add(other_suffixed_col)
                    break
            if not found:
                print(f"   WARNING: Could not find non-null metadata for '{col}'. "
                      f"Using first column found (may be NaN).")
                if col in combined_metadata_df_raw.columns:
                    final_metadata_cols_data[col] = combined_metadata_df_raw[col]
                    processed_cols.add(col)
                else:
                    for i, vn in enumerate(config["view_names"]):
                        suffixed_col = f"{col}_{vn}" if i > 0 else col
                        if suffixed_col in combined_metadata_df_raw.columns: final_metadata_cols_data[col] = combined_metadata_df_raw[suffixed_col]; processed_cols.add(col); processed_cols.add(suffixed_col); break

    combined_metadata_df = pd.DataFrame(final_metadata_cols_data)
    # Index should already be master reference order from reindexing above
    if not combined_metadata_df.index.equals(pd.Index(master_reference_row_names)): print("   ERROR: Metadata index does not match master reference order after final combination."); exit()
    combined_metadata_df.index.name = "MasterReferenceID" # Set index name

    # Ensure correct data types (same as before)
    if 'Day' in combined_metadata_df.columns: combined_metadata_df['Day'] = pd.to_numeric(combined_metadata_df['Day'], errors='coerce')
    if 'Genotype' in combined_metadata_df.columns: combined_metadata_df['Genotype'] = combined_metadata_df['Genotype'].astype(str)
    if 'Treatment' in combined_metadata_df.columns: combined_metadata_df['Treatment'] = combined_metadata_df['Treatment'].astype(str)
    if 'Batch' in combined_metadata_df.columns: combined_metadata_df['Batch'] = combined_metadata_df['Batch'].astype(str)

    print(f"Successfully combined metadata for {len(combined_metadata_df)} samples.")
    # --- Save to NEW output dir ---
    combined_metadata_outfile = os.path.join(config["output_dir"], "aligned_combined_metadata.csv")
    combined_metadata_df.to_csv(combined_metadata_outfile, index=True)
    print(f"Combined aligned metadata saved to: {combined_metadata_outfile}")

except Exception as e: print(f"ERROR combining metadata: {e}"); traceback.print_exc(); exit()
# --- End Section 4 ---

# --- Section 5: Initialize MOFA+ and Set Data/Options --- *** SKIPPED if loading model *** ---
print("\n5. Initializing MOFA+ model and setting data/options...")
if not config["load_existing_model"]:
    try:
        # --- This block only runs if NOT loading an existing model ---
        print("   (SKIPPED: load_existing_model is True)")
        # --- Re-load data in the format MOFA expects (using the scaled data) ---
        # --- This part would need to be re-implemented if training from scratch ---
        # --- For now, we assume loading, so this remains skipped ---
        # ent = entry_point()
        # ent.set_data_options(scale_views=False) # Data was scaled in Section 3
        # ent.set_data_matrix(data_views_input_format_LOADED_FROM_ABOVE, views_names=config["view_names"], ...) # Need to load scaled data
        # ent.set_model_options(...)
        # ent.set_train_options(...)
        # print("   MOFA+ initialized for NEW training.")
    except Exception as e: print(f"ERROR during MOFA+ initialization: {e}"); traceback.print_exc(); exit()
else:
    print("   (SKIPPED: load_existing_model is True)")
# --- End Section 5 ---

# --- Section 6: Build and Run MOFA+ Model --- *** SKIPPED if loading model *** ---
print("\n6. Building and running MOFA+ model...")
if not config["load_existing_model"]:
    # --- This block only runs if NOT loading an existing model ---
    print("   (SKIPPED: load_existing_model is True)")
    # print(f"(Running up to {config['maxiter']} iterations... This can take time.)")
    # mofa_start_time = time.time()
    # try:
    #     ent.build(); ent.run(); print("   Inference completed.")
    #     model_outfile = os.path.join(config["output_dir"], config["outfile"]) # Save to NEW dir if training
    #     ent.save(model_outfile, save_data=False); print(f"   Model saved to {model_outfile}")
    #     mofa_end_time = time.time(); print(f"   Time for MOFA+ Training: {mofa_end_time - mofa_start_time:.2f} seconds")
    # except Exception as e: print(f"ERROR: MOFA+ model training failed: {e}"); traceback.print_exc(); exit()
else:
    print("   (SKIPPED: load_existing_model is True)")
# --- End Section 6 ---

# --- Section 7: Load Results from EXISTING Model --- *** MODIFIED *** ---
print(f"\n7. Loading results from EXISTING model file: {config['existing_model_path']}")
extraction_start_time = time.time()
factors_df, weights_dict, variance_explained_df = None, {}, None
active_factors_indices, num_active_factors, factor_column_names = None, 0, []

if not os.path.exists(config["existing_model_path"]):
    print(f"   ERROR: Specified existing model file not found: {config['existing_model_path']}")
    exit()

try:
    with h5py.File(config["existing_model_path"], 'r') as hf:
        group_name_in_hdf5 = config["groups_names"][0]  # Assume 'group0'

        # Determine Active Factors from Loaded Variance Data
        var_exp_factors_path = f'variance_explained/r2_per_factor/{group_name_in_hdf5}'
        total_r2_per_factor = None
        if var_exp_factors_path in hf:
            var_exp_data = hf[var_exp_factors_path][()]
            # Ensure shape is (views, factors)
            n_views_expected = len(config["view_names"])
            if var_exp_data.shape[0] != n_views_expected:
                if var_exp_data.shape[1] == n_views_expected:
                    print("     Transposing loaded r2_per_factor data to (views, factors).")
                    var_exp_data = var_exp_data.T
                else:
                    print(f"     ERROR: Loaded r2_per_factor shape {var_exp_data.shape} "
                          f"inconsistent with view count {n_views_expected}. "
                          f"Cannot determine active factors reliably.")
                    exit()
            
            total_r2_per_factor = var_exp_data.sum(axis=0)  # Sum across views for each factor
            activity_threshold = config["drop_factor_threshold"]
            active_factors_mask = total_r2_per_factor > activity_threshold
            active_factors_indices = np.where(active_factors_mask)[0]
            num_active_factors = len(active_factors_indices)
            num_converged_factors = len(total_r2_per_factor)
            print(f"   Loaded variance explained. Identified {num_active_factors} "
                  f"active factors (R2>{activity_threshold:.3f}) "
                  f"out of {num_converged_factors}.")
        else:
            print(f"   WARNING: VE path '{var_exp_factors_path}' not found in loaded model. "
                  f"Assuming all factors active based on Z matrix.")
            factors_path_temp = f'expectations/Z/{group_name_in_hdf5}'
            if factors_path_temp in hf:
                # Factors shape is (factors, samples) in HDF5
                num_converged_factors = hf[factors_path_temp].shape[0]
                active_factors_indices = np.arange(num_converged_factors)
                num_active_factors = num_converged_factors
                print(f"   Assuming {num_active_factors} active factors based on Z matrix.")
            else:
                print(f"   FATAL ERROR: Cannot determine factor count from loaded model (missing Z).")
                exit()

        # Load Factors (Z)
        if num_active_factors > 0:
            factor_column_names = [f"Factor{i+1}" for i in active_factors_indices]
            factors_path = f'expectations/Z/{group_name_in_hdf5}'
            if factors_path in hf:
                factors_raw_all = hf[factors_path][()]  # Shape: (factors, samples)
                factors_raw_active = factors_raw_all[active_factors_indices, :]  # Select active factors
                factors_active = factors_raw_active.T  # Transpose to (samples, factors)

                # Align with master_reference_row_names
                # Get sample names from HDF5 to check order
                samples_path = f'samples/{group_name_in_hdf5}'
                if samples_path in hf:
                    hdf5_sample_names = [s.decode('utf-8') if isinstance(s, bytes) else str(s)
                                         for s in hf[samples_path][()]]
                    # Create temp df with HDF5 order
                    temp_factors_df = pd.DataFrame(factors_active, 
                                                  index=hdf5_sample_names,
                                                  columns=factor_column_names)
                    # Reindex to master order
                    factors_df = temp_factors_df.reindex(master_reference_row_names)
                    if factors_df.isnull().values.any():
                        print("     WARNING: NaNs introduced when reindexing factors to master order. "
                              "Check sample name consistency.")
                        factors_df = factors_df.fillna(0)  # Fill NaNs
                    factors_df.index.name = "MasterReferenceID"
                    # Save to output dir
                    factors_outfile = os.path.join(config["output_dir"], 
                                                  "mofa_latent_factors_active.csv")
                    factors_df.to_csv(factors_outfile)
                    print(f"   - Active factors loaded, aligned, and saved ({factors_df.shape}): "
                          f"{factors_outfile}")
                else:
                    print(f"   ERROR: Sample names path '{samples_path}' not found in HDF5. "
                          f"Cannot reliably align factors.")
                    factors_df = None
            else:
                print(f"   ERROR: Factors path '{factors_path}' not found.")
                factors_df = None

            # Load Weights (W)
            weights_dict = {}
            weights_group_path = 'expectations/W'
            if weights_group_path in hf:
                for view_name in config["view_names"]:
                    weights_path = f'{weights_group_path}/{view_name}'
                    if weights_path in hf:
                        weights_raw_all = hf[weights_path][()]  # Shape: (factors, features)
                        weights_raw_active = weights_raw_all[active_factors_indices, :]  # Select active
                        weights_active = weights_raw_active.T  # Transpose to (features, factors)

                        view_unique_feature_names = feature_names_dict.get(view_name)
                        if view_unique_feature_names is None:
                            print(f"   ERROR: Feature names for view '{view_name}' not found.")
                            continue
                        if weights_active.shape[0] != len(view_unique_feature_names):
                            print(f"   ERROR: Weight dimensions ({weights_active.shape[0]}) mismatch "
                                  f"feature count ({len(view_unique_feature_names)}) for '{view_name}'.")
                            continue

                        weights_df = pd.DataFrame(weights_active, 
                                                 index=view_unique_feature_names,
                                                 columns=factor_column_names)
                        weights_dict[view_name] = weights_df
                        # Save to output dir
                        weights_outfile = os.path.join(config["output_dir"], 
                                                     f"mofa_feature_weights_{view_name}_active.csv")
                        weights_df.to_csv(weights_outfile)
                    else:
                        print(f"   WARNING: Weights path '{weights_path}' not found for '{view_name}'.")
                print(f"   - Active weights loaded and saved for {len(weights_dict)} views.")
            else:
                print("   ERROR: Weights group 'expectations/W' not found in HDF5.")

            # Load and Save Variance Explained
            if var_exp_factors_path in hf:  # Use var_exp_data loaded earlier
                variance_explained_active = var_exp_data[:, active_factors_indices]  # Select active factors
                if variance_explained_active.shape[1] == num_active_factors:
                    # Transpose to (factors, views)
                    variance_explained_active_transposed = variance_explained_active.T
                    variance_explained_df = pd.DataFrame(variance_explained_active_transposed,
                                                        index=factor_column_names,
                                                        columns=config["view_names"])
                    total_r2_per_view_active = variance_explained_df.sum(axis=0)
                    variance_explained_df.loc['Total R2 (Active Factors)'] = total_r2_per_view_active
                    # Save to output dir
                    var_exp_outfile = os.path.join(config["output_dir"], 
                                                 "mofa_variance_explained_active.csv")
                    variance_explained_df.to_csv(var_exp_outfile)
                    print(f"   - Active Variance explained loaded and saved ({variance_explained_df.shape}).")
                    print("     Total Variance Explained per View (by Active Factors):")
                    [print(f"       - {view}: {r2_val:.3f}") 
                     for view, r2_val in total_r2_per_view_active.items()]
                else:
                    print(f"   ERROR: Active VE dimensions mismatch after loading.")
                    variance_explained_df = None
            else:
                print("   WARNING: VE data not loaded.")
                variance_explained_df = None
        else:
            print("   Skipping results extraction: no active factors identified from loaded model.")
            factors_df, weights_dict, variance_explained_df = None, {}, None

except Exception as e:
    print(f"ERROR loading results from existing model: {e}")
    traceback.print_exc()
    factors_df, weights_dict, variance_explained_df = None, {}, None

extraction_end_time = time.time()
print(f"   Results Loading Time: {extraction_end_time - extraction_start_time:.2f} seconds")
# --- End Section 7 ---


# --- Section 8: Basic Visualizations --- (Optional) ---
if not config["skip_visualizations"]:
    print("\n8. Generating basic visualizations...")
    vis_start_time = time.time()
    try:
        if variance_explained_df is not None and not variance_explained_df.empty:
            print("   - Plotting Variance Explained...")
            ve_totals = variance_explained_df.loc['Total R2 (Active Factors)']
            plt.figure(figsize=(8, 5))
            sns.barplot(x=ve_totals.index, y=ve_totals.values)
            plt.title('Total Variance Explained per View (Active Factors)')
            plt.ylabel('Total R2')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(config["figure_dir"], "variance_explained_total_per_view.png"))
            plt.close()
            
            ve_heatmap_data = variance_explained_df.drop('Total R2 (Active Factors)', errors='ignore')
            if not ve_heatmap_data.empty:
                plt.figure(figsize=(10, max(6, len(ve_heatmap_data) * 0.4)))
                sns.heatmap(ve_heatmap_data, cmap="viridis", annot=True, fmt=".2f", linewidths=.5)
                plt.title('Variance Explained per Factor and View (Active Factors)')
                plt.xlabel('View')
                plt.ylabel('Factor')
                plt.tight_layout()
                plt.savefig(os.path.join(config["figure_dir"], "variance_explained_heatmap.png"))
                plt.close()
        else:
            print("   Skipping variance plots (data unavailable or empty).")

        if factors_df is not None and not factors_df.empty and combined_metadata_df is not None:
            print("   - Plotting Factor Values vs Metadata...")
            common_index = factors_df.index.intersection(combined_metadata_df.index)
            plot_factors_df = factors_df.loc[common_index]
            plot_metadata_df = combined_metadata_df.loc[common_index]
            num_factors_to_plot = min(plot_factors_df.shape[1], 6)
            metadata_to_plot = [c for c in ['Genotype', 'Day', 'Treatment'] 
                              if c in plot_metadata_df.columns]
            
            if num_factors_to_plot > 0 and metadata_to_plot:
                for factor in plot_factors_df.columns[:num_factors_to_plot]:
                    for meta_col in metadata_to_plot:
                        plt.figure(figsize=(10, 6)); plot_data = pd.concat([plot_factors_df[factor], plot_metadata_df[meta_col]], axis=1).dropna()
                        if pd.api.types.is_numeric_dtype(plot_data[meta_col]): sns.scatterplot(data=plot_data, x=meta_col, y=factor, alpha=0.7)
                        else: sns.boxplot(data=plot_data, x=meta_col, y=factor, showfliers=False, color="lightblue"); sns.stripplot(data=plot_data, x=meta_col, y=factor, alpha=0.5, color="black", jitter=True)
                        plt.title(f'{factor} vs {meta_col}'); plt.xlabel(meta_col); plt.ylabel(factor); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(os.path.join(config["figure_dir"], f"factor_plot_{factor}_vs_{meta_col}.png")); plt.close()
            else: print("     Skipping factor vs metadata plots (no factors/metadata to plot or alignment failed).")
        else: print("   Skipping factor vs metadata plots (data unavailable or empty).")
    except Exception as e: print(f"   ERROR during visualization: {e}"); traceback.print_exc()
    vis_end_time = time.time(); print(f"   Visualizations generation attempt finished (Time: {vis_end_time - vis_start_time:.2f} seconds)")
else:
    print("\n8. SKIPPING basic visualizations.")
# --- End Section 8 ---

# --- Section 8.5: Advanced Visualizations --- (Optional) ---
if not config["skip_visualizations"]:
    print("\n8.5 Generating Advanced Visualizations...")
    adv_vis_start_time = time.time()
    # (Advanced visualization code unchanged - but ensure variables exist)
    required_data_available = True; plot_error_msg = "   Skipping advanced visualizations because: "
    if 'factors_df' not in locals() or factors_df is None or factors_df.empty: required_data_available = False; plot_error_msg += "factors_df missing. "
    if 'combined_metadata_df' not in locals() or combined_metadata_df is None or combined_metadata_df.empty: required_data_available = False; plot_error_msg += "combined_metadata_df missing. "
    if 'weights_dict' not in locals() or not weights_dict: required_data_available = False; plot_error_msg += "weights_dict missing. "
    # Check variables needed for context factors calculation (run in Section 9)
    if 'factor_metadata_corr' not in locals(): factor_metadata_corr = None
    if 'genotype_diff_results' not in locals(): genotype_diff_results = None
    # Check variable needed for heatmap (run in Section 10.3)
    if 'selected_features_for_transformer' not in locals(): selected_features_for_transformer = None

    if required_data_available:
        print("   - Plotting Factor Trajectories over Time...") # Needs factor_metadata_corr
        try:
            if factor_metadata_corr is not None and not factor_metadata_corr.empty:
                day_corr_factors = factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Day') & factor_metadata_corr['Significant_FDR']]['Factor'].unique().tolist()
                if day_corr_factors:
                    plot_data_time = pd.merge(factors_df[day_corr_factors], combined_metadata_df[['Day', 'Genotype', 'Treatment']], left_index=True, right_index=True)
                    plot_data_time['Day'] = pd.to_numeric(plot_data_time['Day'], errors='coerce'); plot_data_time.dropna(subset=['Day'], inplace=True)
                    for factor in day_corr_factors: plt.figure(figsize=(10, 6)); sns.lineplot(data=plot_data_time, x='Day', y=factor, hue='Genotype', style='Treatment', marker='o', errorbar=('ci', 95)); plt.title(f'Mean Trajectory of {factor} over Time'); plt.xlabel('Day'); plt.ylabel(f'{factor} Score (Mean +/- 95% CI)'); plt.xticks(sorted(plot_data_time['Day'].unique())); plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(config["figure_dir"], f"factor_trajectory_{factor}_vs_Day.png")); plt.close()
                else: print("     Skipping factor trajectories (no factors significantly correlated with Day).")
            else: print("     Skipping factor trajectories (correlation data unavailable).")
        except Exception as e: print(f"   ERROR plotting factor trajectories: {e}"); traceback.print_exc()

        print("\n   - Plotting Top Feature Loadings for Key Factors...") # Needs weights_dict, variance_explained_df, genotype_diff_results
        try:
            factors_to_plot_weights = []; n_top_features = 15
            if 'variance_explained_df' in locals() and variance_explained_df is not None and not variance_explained_df.empty: factors_to_plot_weights.extend(variance_explained_df.drop('Total R2 (Active Factors)', errors='ignore').sum(axis=1).nlargest(2).index.tolist())
            if genotype_diff_results is not None and not genotype_diff_results.empty: factors_to_plot_weights.extend(genotype_diff_results[genotype_diff_results['Significant_FDR']].nsmallest(2, 'P_value_FDR')['Factor'].tolist())
            if factor_metadata_corr is not None and not factor_metadata_corr.empty:
                day_corr_df_sig = factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Day') & factor_metadata_corr['Significant_FDR']]
                if not day_corr_df_sig.empty: factors_to_plot_weights.extend(day_corr_df_sig.nsmallest(2, 'P_value_FDR')['Factor'].tolist())
            factors_to_plot_weights = sorted(list(set(factors_to_plot_weights)))
            if not factors_to_plot_weights: print("     Skipping top feature loadings plot (no key factors identified).")
            else:
                print(f"     Plotting top {n_top_features} feature loadings for factors: {factors_to_plot_weights}")
                for factor in factors_to_plot_weights:
                    for view_name, weights_df in weights_dict.items():
                        if factor not in weights_df.columns: continue
                        factor_weights = weights_df[factor].dropna().sort_values(ascending=False);
                        if len(factor_weights) < n_top_features * 2 : continue
                        top_positive = factor_weights.head(n_top_features); top_negative = factor_weights.tail(n_top_features).sort_values(ascending=True); top_features_factor = pd.concat([top_positive, top_negative])
                        plt.figure(figsize=(10, max(8, len(top_features_factor) * 0.3))); sns.barplot(x=top_features_factor.values, y=top_features_factor.index, palette="vlag"); cleaned_labels = [label.replace(f"_{view_name}", "") for label in top_features_factor.index]; plt.yticks(ticks=range(len(cleaned_labels)), labels=cleaned_labels)
                        plt.title(f'Top {n_top_features} +/- Feature Loadings for {factor} in {view_name}'); plt.xlabel('Weight (Loading)'); plt.ylabel('Feature'); plt.tight_layout(); plt.savefig(os.path.join(config["figure_dir"], f"feature_loadings_{factor}_{view_name}.png")); plt.close()
        except Exception as e: print(f"   ERROR plotting feature loadings: {e}"); traceback.print_exc()

        print("\n   - Plotting Heatmap of Selected Features...") # Needs selected_features_for_transformer
        try:
            if selected_features_for_transformer:
                views_for_heatmap = [vn for vn in ["leaf_spectral", "leaf_metabolite"] if vn in selected_features_for_transformer and selected_features_for_transformer[vn]]
                if not views_for_heatmap: print("     Skipping selected features heatmap (no features for target views).")
                else:
                    metadata_heatmap = combined_metadata_df[['Genotype', 'Treatment', 'Day', 'Batch']].copy()
                    for view_name in views_for_heatmap:
                        print(f"     Generating heatmap for selected features in '{view_name}'...")
                        # --- Load from NEW transformer input file ---
                        infile = os.path.join(config["output_dir"], f"transformer_input_{view_name}{config['feature_suffix']}.csv") # Use suffix
                        if not os.path.exists(infile): print(f"       Skipping heatmap for {view_name}: File not found: {infile}"); continue
                        data_df = pd.read_csv(infile);
                        if 'Row_names' not in data_df.columns: print(f"       Skipping heatmap: 'Row_names' missing in {infile}"); continue
                        data_df.set_index('Row_names', inplace=True) # Index is MasterReferenceID
                        # Get selected features for THIS view (already suffixed)
                        selected_suffixed_features = selected_features_for_transformer.get(view_name, [])
                        # Get original names to match columns in loaded file
                        selected_original_names = [f.replace(f"_{view_name}", "") for f in selected_suffixed_features]
                        selected_cols_in_data = [col for col in selected_original_names if col in data_df.columns]
                        if not selected_cols_in_data: print(f"       Skipping heatmap: No selected features found in columns of {infile}"); continue
                        heatmap_data = data_df[selected_cols_in_data]; common_index_heatmap = heatmap_data.index.intersection(metadata_heatmap.index); heatmap_data = heatmap_data.loc[common_index_heatmap]; metadata_groups = metadata_heatmap.loc[common_index_heatmap]
                        scaler_heatmap = StandardScaler(); heatmap_data_scaled = scaler_heatmap.fit_transform(heatmap_data); heatmap_data_scaled_df = pd.DataFrame(heatmap_data_scaled, index=heatmap_data.index, columns=heatmap_data.columns)
                        col_colors_map = {}; palettes = {"Genotype": "Set1", "Treatment": "Set2", "Day": "viridis", "Batch": "Set3"}
                        for col in ['Genotype', 'Treatment', 'Day', 'Batch']:
                            if col in metadata_groups.columns: unique_vals = sorted(metadata_groups[col].unique()); lut = dict(zip(unique_vals, sns.color_palette(palettes.get(col, "coolwarm"), len(unique_vals)))); col_colors_map[col] = metadata_groups[col].map(lut)
                        col_colors = pd.DataFrame(col_colors_map) if col_colors_map else None
                        max_features_heatmap = 75
                        if heatmap_data_scaled_df.shape[1] > max_features_heatmap: print(f"       Reducing features shown in heatmap to {max_features_heatmap} (highest variance)."); top_var_features = heatmap_data_scaled_df.var(axis=0).nlargest(max_features_heatmap).index; heatmap_data_to_plot = heatmap_data_scaled_df[top_var_features]
                        else: heatmap_data_to_plot = heatmap_data_scaled_df
                        print(f"       Plotting clustermap for {heatmap_data_to_plot.shape[1]} features...")
                        g = sns.clustermap(heatmap_data_to_plot.T, cmap="vlag", col_colors=col_colors, col_cluster=True, row_cluster=True, figsize=(12, max(10, heatmap_data_to_plot.shape[1] * 0.15)), linewidths=0.0, xticklabels=False, z_score=0); g.ax_heatmap.set_ylabel("Selected Features"); g.ax_heatmap.set_xlabel("Samples (Clustered)")
                        plt.suptitle(f"Heatmap of Selected {view_name.replace('_',' ').title()} Features (Scaled, N={config['transformer_feature_cap']})", y=1.02) # Added N
                        plt.savefig(os.path.join(config["figure_dir"], f"heatmap_selected_features_{view_name}{config['feature_suffix']}.png"), bbox_inches='tight'); plt.close() # Add suffix
            else: print("     Skipping selected features heatmap (feature selection dictionary missing or empty).")
        except Exception as e: print(f"   ERROR plotting selected feature heatmap: {e}"); traceback.print_exc()

        print("\n   - Plotting Factor Correlation Heatmap...")
        try:
            if factors_df is not None and not factors_df.empty and factors_df.shape[1] > 1:
                factor_corr = factors_df.corr(); plt.figure(figsize=(max(8, factor_corr.shape[0]*0.6), max(6, factor_corr.shape[0]*0.6))); sns.heatmap(factor_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5, center=0); plt.title('Correlation Between MOFA+ Factors'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(config["figure_dir"], "factor_correlation_heatmap.png")); plt.close()
            else: print("     Skipping factor correlation heatmap (<= 1 factor or factors_df is empty/None).")
        except Exception as e: print(f"   ERROR plotting factor correlation heatmap: {e}"); traceback.print_exc()
    else: print(plot_error_msg)
    adv_vis_end_time = time.time(); print(f"   Advanced visualizations attempt finished (Time: {adv_vis_end_time - adv_vis_start_time:.2f} seconds)")
else:
    print("\n8.5 SKIPPING advanced visualizations.")
# --- End Section 8.5 ---


# --- Section 9: Enhanced Validation and Biological Analysis --- (Run Functions) ---
# --- SAVES TO NEW output_dir ---
print("\n9. Enhanced Validation and Biological Analysis...")
enhanced_start_time = time.time()

# Define analysis functions
def calculate_factor_metadata_associations(factors_df, metadata_df, fdr_alpha=0.05, output_dir="."):
    """Calculate associations between MOFA+ factors and metadata variables"""
    print("   - Calculating Factor-Metadata associations (Spearman)...")
    if factors_df is None or metadata_df is None or factors_df.empty or metadata_df.empty:
        print("     Skipping Factor-Metadata associations (Input data unavailable).")
        return None

    # Ensure indices match before calculation
    common_index = factors_df.index.intersection(metadata_df.index)
    if len(common_index) == 0:
        print("     ERROR: No common index between factors and metadata. Cannot calculate associations.")
        return None
    if len(common_index) < len(factors_df) or len(common_index) < len(metadata_df):
         print(f"     WARN: Index mismatch. Using {len(common_index)} common samples for associations.")

    factors_aligned = factors_df.loc[common_index]
    metadata_aligned = metadata_df.loc[common_index]


    results_list = []
    # Use the aligned metadata dataframe here
    metadata_cols_to_test = [
        c for c in metadata_aligned.columns
        if c not in ["Row_names", "Vac_id", "Entry", "Tissue.type"] # BaseSampleID is index, Row_names might be a column
           and c != metadata_aligned.index.name # Exclude index name itself
           and metadata_aligned[c].nunique() > 1
    ]
    print(f"     Testing associations for metadata columns: {metadata_cols_to_test}")

    for factor in factors_aligned.columns:
        for meta_col in metadata_cols_to_test:
            # Data is already aligned, just drop NAs for the specific pair
            temp_df = pd.concat([factors_aligned[factor], metadata_aligned[meta_col]], axis=1).dropna()

            if temp_df.shape[0] < 5: continue # Skip if too few samples after NA drop
            if temp_df[meta_col].nunique() < 2: continue # Skip if only one category left

            factor_values = temp_df[factor].values
            meta_values = temp_df[meta_col].values

            try:
                # Attempt numeric correlation first
                meta_values_numeric = pd.to_numeric(meta_values)
                corr, p_value = spearmanr(factor_values, meta_values_numeric)
                note = None
            except ValueError:
                # If numeric fails, it's likely categorical - TRY factorizing
                # *** CORRECTED BLOCK START ***
                try:
                    meta_values_codes, _ = pd.factorize(meta_values)
                    # Check if factorization resulted in only one code (can happen after NA drop)
                    if len(np.unique(meta_values_codes)) < 2:
                        # print(f"     Skipping {factor} vs {meta_col} (Categorical): Only one category after NA drop.") # Optional Debug
                        continue # Skip this specific correlation
                    corr, p_value = spearmanr(factor_values, meta_values_codes)
                    note = 'Used factorized codes'
                except Exception as e_cat:
                    print(f"     WARN: Could not correlate {factor} vs {meta_col} (Categorical): {e_cat}")
                    corr, p_value, note = np.nan, np.nan, 'Correlation failed'
                # *** CORRECTED BLOCK END ***

            # Append result if correlation was successful
            if not np.isnan(corr) and not np.isnan(p_value):
                results_list.append({'Factor': factor, 'Metadata': meta_col, 'Correlation': corr, 'P_value': p_value, 'Note': note})

    if not results_list:
        print("     No valid correlations calculated.")
        return None

    results_df = pd.DataFrame(results_list)
    # --- FDR Correction ---
    if 'P_value' in results_df.columns and not results_df['P_value'].isnull().all():
        p_vals_clean = results_df['P_value'].dropna()
        if not p_vals_clean.empty:
            reject, pvals_corrected, _, _ = multipletests(p_vals_clean, alpha=fdr_alpha, method='fdr_bh')
            results_df.loc[p_vals_clean.index, 'P_value_FDR'] = pvals_corrected
            results_df.loc[p_vals_clean.index, 'Significant_FDR'] = reject
        else:
            results_df['P_value_FDR'] = np.nan
            results_df['Significant_FDR'] = False # Ensure column exists even if no p-values
    else:
        results_df['P_value_FDR'] = np.nan
        results_df['Significant_FDR'] = False # Ensure column exists

    results_df['Significant_FDR'] = results_df['Significant_FDR'].fillna(False).astype(bool)
    results_df = results_df.sort_values(by=['Significant_FDR', 'P_value_FDR'], ascending=[False, True])

    # --- Save results ---
    outfile = os.path.join(output_dir, "mofa_factor_metadata_associations_spearman.csv")
    results_df.to_csv(outfile, index=False)
    print(f"     Factor-Metadata associations saved to {outfile}")
    return results_df

def outline_bootstrap_analysis(weights_dict, factors_df, output_dir, num_bootstrap_runs=100):
    if config["skip_hypothetical_outputs"]: print("   - SKIPPING Bootstrap Validation (Hypothetical Output)"); return None
    print("   - Bootstrap Validation (Outline & Hypothetical Output):") # (Rest of function unchanged)
    if not weights_dict or factors_df is None or factors_df.empty: print("     Skipping hypothetical output generation (weights/factors unavailable)."); return None
    hypothetical_stability_data = []
    for view_name, weights_df in weights_dict.items():
        if weights_df is None or weights_df.empty: continue
        for factor_name in factors_df.columns:
             if factor_name not in weights_df.columns: continue
             abs_weights = weights_df[factor_name].abs().sort_values(ascending=False); n_feat = len(abs_weights); rank = 1; stability_scores = np.random.uniform(0.0, 1.0, n_feat)
             if n_feat == 0: continue
             for feature_name, abs_weight in abs_weights.items(): hypothetical_stability_data.append({'Factor': factor_name, 'View': view_name, 'Feature': feature_name, 'StabilityScore': stability_scores[rank-1], 'Rank_AbsWeight_OriginalRun': rank}); rank += 1
    if not hypothetical_stability_data: print("     No data to create hypothetical stability file."); return None
    stability_df = pd.DataFrame(hypothetical_stability_data).sort_values(by=['Factor', 'View', 'StabilityScore'], ascending=[True, True, False])
    outfile = os.path.join(output_dir, "mofa_bootstrap_stability_HYPOTHETICAL.csv"); stability_df.to_csv(outfile, index=False); print(f"     Hypothetical bootstrap stability structure saved to: {outfile}"); return stability_df

def outline_permutation_testing(factors_df, factor_metadata_corr, output_dir, num_permutations=1000):
    if config["skip_hypothetical_outputs"]: print("   - SKIPPING Permutation Testing (Hypothetical Output)"); return None
    print("   - Permutation Testing (Outline & Hypothetical Output):") # (Rest of function unchanged)
    if factors_df is None or factors_df.empty or factor_metadata_corr is None or factor_metadata_corr.empty: print("     Skipping hypothetical output generation (factors/correlations unavailable)."); return None
    hypothetical_perm_data = []
    tested_metadata_vars = factor_metadata_corr['Metadata'].unique()
    for factor_name in factors_df.columns:
        for meta_var in tested_metadata_vars:
            real_corr_row = factor_metadata_corr[(factor_metadata_corr['Factor'] == factor_name) & (factor_metadata_corr['Metadata'] == meta_var)]
            if real_corr_row.empty: continue
            real_metric_value = real_corr_row['Correlation'].iloc[0]; real_p_value_fdr = real_corr_row['P_value_FDR'].iloc[0]; is_significant_fdr = real_corr_row['Significant_FDR'].iloc[0]; hypothetical_p_val = np.random.uniform(0.0001, 0.049) if is_significant_fdr else (np.random.uniform(0.1, 1.0) if not np.isnan(real_p_value_fdr) else np.nan)
            hypothetical_perm_data.append({'Factor': factor_name, 'Tested_Metadata': meta_var, 'Real_Association_Metric': real_metric_value, 'Real_P_value_FDR': real_p_value_fdr, 'Real_Significant_FDR': is_significant_fdr, 'Hypothetical_Permutation_P_Value': hypothetical_p_val})
    if not hypothetical_perm_data: print("     No data to create hypothetical permutation results file."); return None
    perm_df = pd.DataFrame(hypothetical_perm_data).sort_values(by=['Tested_Metadata', 'Hypothetical_Permutation_P_Value']); outfile = os.path.join(output_dir, "mofa_permutation_test_results_HYPOTHETICAL.csv"); perm_df.to_csv(outfile, index=False); print(f"     Hypothetical permutation test structure saved to: {outfile}"); return perm_df

def analyze_genotype_differences(factors_df, metadata_df, fdr_alpha=0.05, output_dir="."):
    print("   - Analyzing Genotype Differences in Factors (Mann-Whitney U)...") # (Rest of function unchanged)
    if factors_df is None or metadata_df is None or factors_df.empty or metadata_df.empty: print("     Skipping Genotype analysis (Input data unavailable)."); return None
    if 'Genotype' not in metadata_df.columns: print("     Skipping Genotype analysis ('Genotype' column not in metadata)."); return None
    common_index = factors_df.index.intersection(metadata_df.index); factors_aligned = factors_df.loc[common_index]; metadata_aligned = metadata_df.loc[common_index]
    genotypes = metadata_aligned['Genotype'].unique(); genotypes = [g for g in genotypes if pd.notna(g)]
    if len(genotypes) != 2: print(f"     Skipping Genotype analysis (Expected 2 non-NA genotypes, found {len(genotypes)})."); return None
    g1_label, g2_label = genotypes[0], genotypes[1]; print(f"     Comparing genotypes: '{g1_label}' vs '{g2_label}'"); results = []
    for factor in factors_aligned.columns:
        g1_values = factors_aligned.loc[metadata_aligned['Genotype'] == g1_label, factor].dropna().values; g2_values = factors_aligned.loc[metadata_aligned['Genotype'] == g2_label, factor].dropna().values
        if len(g1_values) < 3 or len(g2_values) < 3: continue
        try: stat, p_value = mannwhitneyu(g1_values, g2_values, alternative='two-sided', use_continuity=True); results.append({'Factor': factor, 'Comparison': f"{g1_label}_vs_{g2_label}", 'Statistic_U': stat, 'P_value': p_value, f'N_{g1_label}': len(g1_values), f'N_{g2_label}': len(g2_values), f'Mean_{g1_label}': np.mean(g1_values), f'Mean_{g2_label}': np.mean(g2_values)})
        except Exception as e: print(f"     Error during Mann-Whitney U test for {factor}: {e}")
    if not results: print("     No genotype comparisons performed."); return None
    results_df = pd.DataFrame(results); p_vals_clean = results_df['P_value'].dropna()
    if not p_vals_clean.empty: reject, pvals_corrected, _, _ = multipletests(p_vals_clean, alpha=fdr_alpha, method='fdr_bh'); results_df.loc[p_vals_clean.index, 'P_value_FDR'] = pvals_corrected; results_df.loc[p_vals_clean.index, 'Significant_FDR'] = reject
    else: results_df['P_value_FDR'], results_df['Significant_FDR'] = np.nan, False
    results_df['Significant_FDR'] = results_df['Significant_FDR'].fillna(False).astype(bool); results_df = results_df.sort_values(by=['Significant_FDR', 'P_value_FDR'], ascending=[False, True])
    outfile = os.path.join(output_dir, "mofa_genotype_diff_factors.csv"); results_df.to_csv(outfile, index=False); print(f"     Genotype difference analysis saved to {outfile}"); return results_df

def analyze_tissue_coordination(variance_explained_df, r2_threshold=0.01, output_dir="."):
    print("   - Analyzing Tissue/Modality Coordination...") # (Rest of function unchanged)
    if variance_explained_df is None or variance_explained_df.empty: print("     Skipping Tissue Coordination (Variance data unavailable)."); return None
    ve_factors = variance_explained_df.drop('Total R2 (Active Factors)', errors='ignore')
    if ve_factors.empty: print("     Skipping Tissue Coordination (No factor variance data available)."); return None
    tissue_pairs = [('leaf_spectral', 'root_spectral'), ('leaf_metabolite', 'root_metabolite')]; modality_pairs = [('leaf_spectral', 'leaf_metabolite'), ('root_spectral', 'root_metabolite')]; cross_pairs = [('leaf_spectral', 'root_metabolite'), ('root_spectral', 'leaf_metabolite')]; all_pairs = tissue_pairs + modality_pairs + cross_pairs
    print(f"     Identifying factors explaining > {r2_threshold*100:.1f}% variance simultaneously in paired views:"); coord_dfs = []
    for v1, v2 in all_pairs:
         if v1 in ve_factors.columns and v2 in ve_factors.columns:
             coord_factors_mask = (ve_factors[v1] > r2_threshold) & (ve_factors[v2] > r2_threshold); coord_factors = ve_factors[coord_factors_mask]
             if not coord_factors.empty: num_found = len(coord_factors); print(f"       - {v1} <--> {v2}: Found {num_found} coordinating factors."); coord_data = coord_factors[[v1, v2]].copy(); coord_data['Coordination_Type'] = f'{v1}_vs_{v2}'; coord_data['Factor'] = coord_data.index; coord_dfs.append(coord_data.reset_index(drop=True))
    if not coord_dfs: print("     No coordinating factors found."); return None
    all_coord_factors_df = pd.concat(coord_dfs, ignore_index=True); cols_order = ['Factor', 'Coordination_Type'] + [col for col in all_coord_factors_df.columns if col not in ['Factor', 'Coordination_Type']]; all_coord_factors_df = all_coord_factors_df[cols_order].sort_values(by=['Coordination_Type', 'Factor'])
    outfile = os.path.join(output_dir, "mofa_tissue_coordination_factors.csv"); all_coord_factors_df.to_csv(outfile, index=False); print(f"     View coordination analysis saved to {outfile}"); return all_coord_factors_df

# --- Run Enhanced Validation ---
print("   Executing validation functions...")
factor_metadata_corr = calculate_factor_metadata_associations(factors_df, combined_metadata_df, config["fdr_alpha"], config["output_dir"])
hypothetical_stability_df = outline_bootstrap_analysis(weights_dict, factors_df, config["output_dir"])
hypothetical_perm_results_df = outline_permutation_testing(factors_df, factor_metadata_corr, config["output_dir"])
genotype_diff_results = analyze_genotype_differences(factors_df, combined_metadata_df, config["fdr_alpha"], config["output_dir"])
tissue_coordination_results = analyze_tissue_coordination(variance_explained_df, config["drop_factor_threshold"], config["output_dir"])

enhanced_val_end_time = time.time(); print(f"   Enhanced Validation Functions Executed (Time: {enhanced_val_end_time - enhanced_start_time:.2f} seconds)")
# --- End Section 9 ---


# --- Section 10: Feature Selection & Transformer Preparation --- *** MODIFIED *** ---
print(f"\n10. Feature Selection & Transformer Data Preparation "
      f"(Target N={config['transformer_feature_cap']}{config['feature_suffix']})...")
transformer_prep_start_time = time.time()


def calculate_overall_feature_importance(weights_dict, all_relevant_factor_names, output_dir="."):
    """Calculate overall feature importance across all relevant MOFA+ factors"""
    print("   - Calculating Overall Feature Importance (across specified relevant factors)...")
    if not weights_dict:
        print("     Skipping overall importance calculation (weights_dict unavailable).")
        return None
    if not all_relevant_factor_names:
        print("     Skipping overall importance calculation (no relevant factor names provided).")
        return None
    
    overall_importance_dict = {}
    print(f"     Aggregating importance across {len(all_relevant_factor_names)} relevant factors: "
          f"{all_relevant_factor_names}")
    
    for view_name, weights_df in weights_dict.items():
        if weights_df is None or weights_df.empty:
            print(f"     Skipping view '{view_name}': No weights data.")
            continue
        
        factors_in_view = [f for f in all_relevant_factor_names if f in weights_df.columns]
        if not factors_in_view:
            print(f"     Skipping view '{view_name}': None relevant factors present.")
            continue
        
        view_importance_series = weights_df[factors_in_view].abs().sum(axis=1)
        view_importance_df = pd.DataFrame(view_importance_series, columns=['OverallImportance'])
        min_imp, max_imp = view_importance_df['OverallImportance'].min(), view_importance_df['OverallImportance'].max()
        view_importance_df['OverallImportance_scaled'] = (
            (view_importance_df['OverallImportance'] - min_imp) / (max_imp - min_imp) 
            if max_imp > min_imp else 0.0
        )
        view_importance_df = view_importance_df.sort_values(by='OverallImportance', ascending=False)
        overall_importance_dict[view_name] = view_importance_df
        
        # Save to output dir
        outfile = os.path.join(output_dir, f"mofa_feature_importance_{view_name}.csv")
        view_importance_df.to_csv(outfile)
    
    print(f"     Overall importance calculated and saved for {len(overall_importance_dict)} views.")
    return overall_importance_dict


# Identify relevant factors based on significance
print("   - Identifying ALL relevant factors based on significance for Overall Importance...")
all_relevant_factors_list = []
unique_relevant_factor_names = set()

significant_dfs_map = {
    'Genotype_Difference': genotype_diff_results,
    'Day_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Day')] 
                       if factor_metadata_corr is not None else None,
    'Treatment_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Treatment')] 
                           if factor_metadata_corr is not None else None,
    'Batch_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Batch')] 
                        if factor_metadata_corr is not None else None
}

for context_name, sig_df in significant_dfs_map.items():
    if context_name in config['relevant_factor_contexts']:
        if sig_df is not None and not sig_df.empty and 'Significant_FDR' in sig_df.columns:
            context_sig_factors = sig_df[sig_df['Significant_FDR']].copy()
            if not context_sig_factors.empty:
                factor_names = context_sig_factors['Factor'].unique().tolist()
                print(f"     Found {len(factor_names)} significant factors for context "
                      f"'{context_name}': {factor_names}")
                unique_relevant_factor_names.update(factor_names)
                context_sig_factors['Context'] = context_name
                all_relevant_factors_list.append(context_sig_factors[['Factor', 'Context']])

all_relevant_factors_names_list = sorted(list(unique_relevant_factor_names))
all_relevant_factors_combined_df = None

if all_relevant_factors_list:
    all_relevant_factors_combined_df = pd.concat(all_relevant_factors_list).drop_duplicates().reset_index(drop=True)
    # Save to output dir
    outfile_rel = os.path.join(config["output_dir"], "mofa_relevant_factors_list.csv")
    all_relevant_factors_combined_df.to_csv(outfile_rel, index=False)
    print(f"     Identified {len(all_relevant_factors_names_list)} unique relevant factors overall: "
          f"{all_relevant_factors_names_list}")
    print(f"     Relevant factors list saved to: {outfile_rel}")
else:
    print("     WARNING: No relevant factors identified based on loaded results.")
    all_relevant_factors_names_list = []

# Calculate overall importance
overall_feature_importance_dict = calculate_overall_feature_importance(
    weights_dict, all_relevant_factors_names_list, config["output_dir"])

# Bootstrap stability integration (optional)
overall_feature_importance_dict_with_stability = overall_feature_importance_dict  # Default if skipping

if not config["skip_hypothetical_outputs"]:
    def placeholder_integrate_bootstrap_stability(overall_importance_dict, hypothetical_stability_df):
        """Placeholder function to integrate bootstrap stability scores with feature importance"""
        print("   - Placeholder: Integrating Bootstrap Stability...")
        if overall_importance_dict is None:
            print("     Skipping stability integration: Overall importance unavailable.")
            return None
        if hypothetical_stability_df is None:
            print("     Skipping stability integration: Stability data unavailable.")
            return overall_importance_dict
        
        print("     Integrating HYPOTHETICAL stability scores into importance dataframes...")
        updated_importance = {}
        
        for view_name, importance_df in overall_importance_dict.items():
            if importance_df is None or importance_df.empty:
                updated_importance[view_name] = importance_df
                continue
            
            current_importance_df = importance_df.copy()
            view_stability = hypothetical_stability_df[hypothetical_stability_df['View'] == view_name]
            
            if not view_stability.empty:
                feature_stability = view_stability.groupby('Feature')['StabilityScore'].max().reset_index()
                current_importance_df.reset_index(inplace=True)
                merged_df = pd.merge(current_importance_df, feature_stability, 
                                    left_on='index', right_on='Feature', how='left')
                merged_df.set_index('index', inplace=True)
                merged_df.index.name = None
                merged_df.drop(columns=['Feature'], inplace=True, errors='ignore')
                merged_df['StabilityScore'] = merged_df['StabilityScore'].fillna(0)
                updated_importance[view_name] = merged_df
            else:
                current_importance_df['StabilityScore'] = 0.0
                updated_importance[view_name] = current_importance_df
        
        print("     (NOTE: Stability scores used here are RANDOM placeholders).")
        return updated_importance
    
    overall_feature_importance_dict_with_stability = placeholder_integrate_bootstrap_stability(
        overall_feature_importance_dict, hypothetical_stability_df)
else:
    print("   - SKIPPING Bootstrap Stability Integration (Hypothetical Output)")


def select_features_hybrid_stratified(weights_dict, factor_metadata_corr, genotype_diff_results, 
                                     variance_explained_df, config, output_dir="."):
    """
    Select top features using a hybrid stratified approach that balances different 
    biological contexts (Genotype, Time, Other)
    """
    print("   - Selecting Top Features using Hybrid Stratified Method...")
    if not weights_dict:
        print("     ERROR: Weights dictionary is empty.")
        return None, None
    
    fdr_alpha = config["fdr_alpha"]
    max_cap_per_view = config["transformer_feature_cap"]
    context_percentages = config["transformer_context_percentages"]
    min_var_proxy_report_threshold = config["min_variance_explained_proxy_report"]
    feature_suffix = config["feature_suffix"]
    
    selected_features_final = {}
    variance_proxy_report = {}
    stratification_summary = []
    context_factors = {'Genotype': [], 'Time': [], 'Other': []}
    all_relevant_fnames_in_selection = set()
    
    # Get genotype-related factors
    if (genotype_diff_results is not None and not genotype_diff_results.empty and 
            'Significant_FDR' in genotype_diff_results.columns):
        geno_factors = genotype_diff_results[genotype_diff_results['Significant_FDR']]['Factor'].unique().tolist()
        context_factors['Genotype'] = geno_factors
        all_relevant_fnames_in_selection.update(geno_factors)
        print(f"     Context 'Genotype': Identified {len(geno_factors)} factors.")
    
    # Get time-related factors
    if factor_metadata_corr is not None and not factor_metadata_corr.empty:
        time_df = factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Day') & 
                                       factor_metadata_corr['Significant_FDR']]
        if not time_df.empty:
            time_factors = time_df['Factor'].unique().tolist()
            context_factors['Time'] = time_factors
            all_relevant_fnames_in_selection.update(time_factors)
            print(f"     Context 'Time' (Day): Identified {len(time_factors)} factors.")
        
        # Get other factors (treatment, batch)
        other_factors_set = set()
        treat_df = factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Treatment') & 
                                        factor_metadata_corr['Significant_FDR']]
        if not treat_df.empty:
            other_factors_set.update(treat_df['Factor'].unique())
            print(f"     Context 'Other': Identified {len(treat_df['Factor'].unique())} Treatment factors.")
        
        batch_df = factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Batch') & 
                                       factor_metadata_corr['Significant_FDR']]
        if not batch_df.empty:
            other_factors_set.update(batch_df['Factor'].unique())
            print(f"     Context 'Other': Identified {len(batch_df['Factor'].unique())} Batch factors.")
        
        if other_factors_set:
            context_factors['Other'] = list(other_factors_set)
            all_relevant_fnames_in_selection.update(list(other_factors_set))
            print(f"     Context 'Other' (Treat+Batch): Total {len(context_factors['Other'])} unique factors.")
    
    all_relevant_fnames_in_selection_list = sorted(list(all_relevant_fnames_in_selection))
    if not all_relevant_fnames_in_selection_list:
        print("     WARNING: No significant factors found for ANY context.")
        return None, None
    
    print(f"     Factors used for selection: {all_relevant_fnames_in_selection_list}")
    overall_importance_for_capping = calculate_overall_feature_importance(
        weights_dict, all_relevant_fnames_in_selection_list, output_dir)
    
    if overall_importance_for_capping is None:
        print("     ERROR: Could not calculate overall importance for capping.")
        return None, None
    
    for view_name, weights_df in weights_dict.items():
        print(f"     --- Processing view: {view_name} ---")
        if weights_df is None or weights_df.empty:
            print("       Skipping view: No weights data.")
            selected_features_final[view_name] = []
            variance_proxy_report[view_name] = 0.0
            continue
        
        overall_importance_view = overall_importance_for_capping.get(view_name)
        context_importance_scores = {}
        top_features_by_context = {}
        selected_counts = {}
        
        # Get top features for each context
        for context, factors in context_factors.items():
            target_n = int(round(max_cap_per_view * context_percentages[context]))
            print(f"       Context '{context}': Target N = {target_n}")
            
            factors_in_view = [f for f in factors if f in weights_df.columns]
            if not factors_in_view:
                print(f"         - No significant factors for context in view.")
                context_importance_scores[context] = pd.Series(dtype=float)
                top_features_by_context[context] = []
                selected_counts[context] = 0
                continue
            
            context_imp = weights_df[factors_in_view].abs().sum(axis=1)
            context_importance_scores[context] = context_imp.sort_values(ascending=False)
            top_features = context_importance_scores[context].head(target_n).index.tolist()
            top_features_by_context[context] = top_features
            selected_counts[context] = len(top_features)
            print(f"         - Selected {len(top_features)} top features based on {len(factors_in_view)} factors.")
        
        # Combine features from all contexts
        combined_features = [feature for sublist in top_features_by_context.values() for feature in sublist]
        unique_features = sorted(list(set(combined_features)))
        n_unique = len(unique_features)
        print(f"       Combined unique features from contexts: {n_unique}")
        
        # Apply cap if needed
        final_selected_features_view = []
        if n_unique <= max_cap_per_view:
            print(f"       Total unique features ({n_unique}) within cap ({max_cap_per_view}). Using all.")
            final_selected_features_view = unique_features
        else:
            print(f"       Total unique features ({n_unique}) exceeds cap ({max_cap_per_view}). Pruning...")
            if overall_importance_view is not None and not overall_importance_view.empty:
                overall_imp_filtered = overall_importance_view.loc[unique_features]['OverallImportance']
                overall_imp_sorted = overall_imp_filtered.sort_values(ascending=False)
                final_selected_features_view = overall_imp_sorted.head(max_cap_per_view).index.tolist()
                print(f"       Pruned to {len(final_selected_features_view)} features.")
            else:
                print(f"       WARNING: Cannot prune (missing importance). Keeping all {n_unique} features.")
                final_selected_features_view = unique_features
        
        selected_features_final[view_name] = final_selected_features_view
        final_n = len(final_selected_features_view)
        
        # Calculate variance proxy
        proxy_pct = 0.0
        relevant_factors_in_view = [f for f in all_relevant_fnames_in_selection_list 
                                   if f in weights_df.columns]
        
        if not relevant_factors_in_view:
            print("       No relevant factors in view for variance proxy.")
        elif not final_selected_features_view:
            print("       No features selected, variance proxy is 0.")
        elif variance_explained_df is None:
            print("       Variance explained unavailable for proxy calc.")
        else:
            try:
                # Select rows first
                weights_relevant = weights_df.loc[final_selected_features_view, relevant_factors_in_view]
                sum_sq_weights_selected = (weights_relevant**2).sum().sum()
                # Original relevant weights
                weights_all_relevant = weights_df[relevant_factors_in_view]
                sum_sq_weights_total = (weights_all_relevant**2).sum().sum()
                
                if sum_sq_weights_total > 1e-9:
                    proxy_pct = (sum_sq_weights_selected / sum_sq_weights_total) * 100
                    print(f"       Variance Explained Proxy (SumSqWeights): {proxy_pct:.2f}%")
                else:
                    print("       Total sum sq weights near zero. Cannot calc proxy.")
                    proxy_pct = 0.0
            except KeyError as e_key:
                print(f"       WARNING: Key error accessing weights for variance proxy "
                      f"(likely feature missing): {e_key}")
                proxy_pct = np.nan
            except Exception as e_vp:
                print(f"       ERROR calculating variance proxy: {e_vp}")
                proxy_pct = np.nan
        
        variance_proxy_report[view_name] = proxy_pct
        
        # Save selected features to file
        outfile_selected = os.path.join(output_dir, 
                                      f"mofa_selected_hybrid_{final_n}_features_{view_name}{feature_suffix}.txt")
        with open(outfile_selected, 'w') as f:
            f.write('\n'.join(final_selected_features_view))
        print(f"       Selected features list saved to: {outfile_selected} (N={final_n})")
        
        stratification_summary.append({
            'View': view_name,
            'N_Selected_Genotype': selected_counts.get('Genotype', 0),
            'N_Selected_Time': selected_counts.get('Time', 0),
            'N_Selected_Other': selected_counts.get('Other', 0),
            'N_Unique_Combined': n_unique,
            'N_Final_Selected': final_n,
            'Variance_Proxy_Pct': round(proxy_pct, 2) if pd.notna(proxy_pct) else 'Error'
        })
    
    summary_df = pd.DataFrame(stratification_summary)
    summary_outfile = os.path.join(output_dir, f"mofa_feature_selection_hybrid_summary{feature_suffix}.csv")
    summary_df.to_csv(summary_outfile, index=False)
    print(f"\n   Hybrid feature selection summary saved to: {summary_outfile}")
    
    return selected_features_final, variance_proxy_report


# Run the hybrid feature selection
selected_features_for_transformer, variance_report = select_features_hybrid_stratified(
    weights_dict, factor_metadata_corr, genotype_diff_results, variance_explained_df, 
    config, config["output_dir"])

print("\n+++ DEBUG: Value of selected_features_for_transformer after function call:")
print(selected_features_for_transformer)
print(f"+++ DEBUG: Type is {type(selected_features_for_transformer)}")
if selected_features_for_transformer is None:
    print("\nERROR: Hybrid feature selection failed (returned None).")
    exit()
elif not selected_features_for_transformer:
    print("\nERROR: Hybrid feature selection returned an empty dictionary.")
    exit()

# --- 10.4 Create Transformer Input Files --- *** MODIFIED TO USE SUFFIX *** ---
def create_transformer_input_files_using_mapping(
    selected_features_dict,
    original_data_paths,
    mapping_df,
    mapping_column_names,
    reference_view_name,
    master_reference_ids,
    metadata_columns,
    output_dir,
    feature_suffix=""
):
    """
    Create transformer input files by selecting the specific features from original data files
    and aligning them with the master reference order.
    """
    print(f"\n   - Creating Transformer Input Files (Suffix: '{feature_suffix}', "
          f"Output Dir: '{output_dir}')...")
    if selected_features_dict is None:
        print("     Skipping (selected features unavailable).")
        return
    if mapping_df is None or mapping_df.empty:
        print(f"     ERROR: Mapping DataFrame is empty.")
        return
    if not master_reference_ids:
        print(f"     ERROR: Master Reference IDs list is empty.")
        return

    reference_map_col = mapping_column_names[reference_view_name]
    master_index_col_name = 'MasterIndex'
    if not isinstance(master_reference_ids, list):
        master_reference_ids = list(master_reference_ids)
    master_reference_ids = [str(item) for item in master_reference_ids]
    output_metadata_cols_final = metadata_columns[:]

    for view_name, feature_list in selected_features_dict.items():
        print(f"\n     ===== START Processing view: {view_name} =====")
        original_file_path = original_data_paths.get(view_name)
        current_map_col = mapping_column_names[view_name]
        if not original_file_path or not os.path.exists(original_file_path):
            print(f"     ERROR: Original data file not found: '{original_file_path}'. Skipping.")
            continue
        if not feature_list:
            print(f"     INFO: No features selected for '{view_name}'. "
                  f"File will contain metadata only.")

        try:
            df_orig = pd.read_csv(original_file_path)
            df_orig['Row_names'] = df_orig['Row_names'].astype(str)
            
            if view_name == reference_view_name:
                if df_orig['Row_names'].duplicated().any():
                    print(f"       ERROR: Duplicate 'Row_names' in reference file '{view_name}'.")
                    continue
                df_orig.rename(columns={'Row_names': master_index_col_name}, inplace=True)
                if master_index_col_name not in df_orig.columns:
                    print(f"      ERROR: Failed to rename 'Row_names'")
                    continue
                df_with_master_index = df_orig
            else:
                mapping_df[current_map_col] = mapping_df[current_map_col].astype(str)
                mapping_df[reference_map_col] = mapping_df[reference_map_col].astype(str)
                map_subset = mapping_df[[current_map_col, reference_map_col]].drop_duplicates()
                df_merged = pd.merge(df_orig, map_subset, left_on='Row_names', 
                                    right_on=current_map_col, how='left')
                if df_merged[reference_map_col].isnull().any():
                    print(f"       ERROR: Merge failed for {df_merged[reference_map_col].isnull().sum()} rows.")
                    continue
                if len(df_merged) != len(df_orig):
                    print(f"       ERROR: Merge changed row count.")
                    continue
                df_merged.rename(columns={reference_map_col: master_index_col_name}, inplace=True)
                df_merged.drop(columns=['Row_names', current_map_col], inplace=True, errors='ignore')
                df_with_master_index = df_merged

            if df_with_master_index[master_index_col_name].duplicated().any():
                print(f"       ERROR: Duplicate '{master_index_col_name}' values found before setting index.")
                continue
                
            df_with_master_index.set_index(master_index_col_name, inplace=True)
            df_with_master_index.index = df_with_master_index.index.astype(str)
            df_aligned = df_with_master_index.reindex(master_reference_ids)
            
            num_all_nan_rows = df_aligned.isnull().all(axis=1).sum()
            if num_all_nan_rows > 0:
                print(f"     WARNING: {num_all_nan_rows} rows are entirely NaN after reindexing.")
            df_aligned.reset_index(inplace=True)

            original_feature_names_to_select = []
            if feature_list:
                original_feature_names_attempt = [f.replace(f"_{view_name}", "") for f in feature_list]
                original_feature_names_to_select = [col for col in original_feature_names_attempt 
                                                  if col in df_aligned.columns]
                if len(original_feature_names_to_select) == 0 and feature_list:
                    print(f"       >>> CRITICAL WARNING: ZERO original features found.")

            cols_to_keep = [master_index_col_name] + [
                col for col in output_metadata_cols_final if col in df_aligned.columns
            ] + original_feature_names_to_select
            
            final_cols = list(dict.fromkeys(col for col in cols_to_keep if col in df_aligned.columns))
            if not final_cols or master_index_col_name not in final_cols:
                print(f"       ERROR: No columns left or '{master_index_col_name}' lost. Skipping.")
                continue
                
            filtered_df = df_aligned[final_cols].copy()
            filtered_df.rename(columns={master_index_col_name: 'Row_names'}, inplace=True)

            # Save to output file with suffix
            outfile = os.path.join(output_dir, f"transformer_input_{view_name}{feature_suffix}.csv")
            filtered_df.to_csv(outfile, index=False, na_rep='NA')
            print(f"     ✅ Standardized transformer input file saved: {outfile} ({filtered_df.shape})")
            
            meta_in_final = set(filtered_df.columns) & set(output_metadata_cols_final + ['Row_names'])
            feature_count_in_final = len(set(filtered_df.columns) - meta_in_final)
            print(f"       File contains {feature_count_in_final} selected feature cols + "
                  f"{len(meta_in_final)} metadata cols.")
            
            if filtered_df.drop(columns=['Row_names'], errors='ignore').isnull().all().all():
                print(f"       >>> CRITICAL WARNING: Data values in saved file for '{view_name}' "
                      f"appear to be all NaN/missing!")

        except Exception as e:
            print(f"     ❌ ERROR creating transformer input file for '{view_name}': {e}")
            traceback.print_exc()
        print(f"     ===== END Processing view: {view_name} =====")
    print("\n   Transformer input file generation attempt finished.")

# --- CALL the Transformer File Creation Function --- *** PASSING NEW ARGS *** ---
create_transformer_input_files_using_mapping(
    selected_features_for_transformer,
    config["data_paths"],
    df_mapping,
    config["mapping_column_names"],
    config["reference_view"],
    master_reference_row_names,
    metadata_columns,
    config["output_dir"],
    config["feature_suffix"]
)
# --- End Section 10.4 ---

transformer_prep_end_time = time.time()
print(f"   Feature Selection & Transformer Prep Completed "
      f"(Time: {transformer_prep_end_time - transformer_prep_start_time:.2f} seconds)")

# --- Section 10.5: Biological Signal Preservation Check (Cross-Validation) --- (Optional) ---
if not config["skip_cv_check"]:
    print("\n10.5 Performing Cross-Validation Check on Selected Features...")
    cv_check_start_time = time.time()
    
    def perform_biological_signal_cv_v2(selected_features_dict, transformer_input_dir, 
                                      combined_metadata_file_path, feature_suffix="", 
                                      n_splits=5, random_state=42):
        """Perform cross-validation to check if selected features preserve biological signal"""
        print(f"   - Performing {n_splits}-fold stratified CV (using suffix '{feature_suffix}')...")
        if not selected_features_dict:
            print("     ERROR: selected_features_dict is empty.")
            return
            
        feature_dfs = []
        sample_order = None
        processed_view_names = list(selected_features_dict.keys())
        print("     Loading feature data from transformer input files...")
        
        try:
            combined_metadata_aligned = pd.read_csv(combined_metadata_file_path, 
                                                  index_col="MasterReferenceID")
            print(f"       Loaded aligned metadata ({combined_metadata_aligned.shape}) "
                  f"using 'MasterReferenceID' index.")
        except Exception as e_load_meta_cv:
            print(f"    ERROR loading aligned metadata for CV: {e_load_meta_cv}")
            return
            
        for view_name in processed_view_names:
            if not selected_features_dict.get(view_name):
                print(f"     Skipping view '{view_name}': No features selected.")
                continue
                
            infile = os.path.join(transformer_input_dir, 
                                f"transformer_input_{view_name}{feature_suffix}.csv")
            if os.path.exists(infile):
                try:
                    df = pd.read_csv(infile)
                    if 'Row_names' in df.columns:
                        df = df.set_index('Row_names')
                    else:
                        raise ValueError(f"'Row_names' column missing in {infile}")
                        
                    if sample_order is None:
                        intersecting_index = df.index.intersection(combined_metadata_aligned.index)
                        sample_order = intersecting_index
                        print(f"       Using sample order from '{view_name}' ({len(sample_order)} matching).")
                        
                    valid_feature_cols = [f for f in selected_features_dict[view_name] 
                                         if f.replace(f"_{view_name}", "") in df.columns]
                    original_valid_feature_names = [f.replace(f"_{view_name}", "") 
                                                  for f in valid_feature_cols]
                    
                    if not original_valid_feature_names:
                        print(f"     WARN: No selected features found in columns of {infile}. Skipping view.")
                        continue
                        
                    df_reindexed = df.reindex(sample_order)
                    df_aligned_features = df_reindexed[original_valid_feature_names].copy()
                    df_aligned_features.columns = valid_feature_cols
                    feature_dfs.append(df_aligned_features)
                    print(f"       Loaded and aligned {df_aligned_features.shape[1]} features for {view_name}.")
                    
                except Exception as e:
                    print(f"     WARN: Could not load/process {infile}: {e}")
                    traceback.print_exc()
            else:
                print(f"     WARN: File not found: {infile}")
                
        if not feature_dfs:
            print("     ERROR: No feature data loaded for CV check.")
            return
            
        X = pd.concat(feature_dfs, axis=1)
        print(f"     Combined feature matrix shape: {X.shape}")
        
        if X.isnull().values.any():
            print("     WARNING: Combined features contain NaNs. Filling with mean...")
            X = X.fillna(X.mean())
            
        if X.isnull().values.any():
            print("     ERROR: Still contains NaNs after imputation.")
            return
            
        try:
            metadata_aligned_cv = combined_metadata_aligned.reindex(X.index)
        except KeyError as e:
            print(f"ERROR: Samples in X index not found in metadata index: {e}")
            return
            
        y_genotype, target_genotype, y_genotype_encoded = None, None, None
        if 'Genotype' in metadata_aligned_cv.columns:
            y_genotype = metadata_aligned_cv['Genotype'].astype(str)
            
        if y_genotype is not None and y_genotype.nunique() > 1:
            target_genotype = 'Genotype'
            le_genotype = LabelEncoder()
            y_genotype_encoded = le_genotype.fit_transform(y_genotype)
            print(f"     Encoded Genotype labels: "
                  f"{dict(zip(le_genotype.classes_, le_genotype.transform(le_genotype.classes_)))}")
        else:
            print("     Skipping Genotype CV: Not available or only one class.")
            
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(solver='liblinear', random_state=random_state, 
                                        class_weight='balanced', max_iter=300))
        ])
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_macro_scorer = make_scorer(f1_score, average='macro')
        cv_results = {}
        
        if target_genotype and y_genotype_encoded is not None:
            print(f"   --- CV for {target_genotype} ---")
            try:
                scores = cross_val_score(clf, X, y_genotype_encoded, cv=cv_strategy, 
                                       scoring=f1_macro_scorer, n_jobs=-1)
                mean_f1, std_f1 = np.mean(scores), np.std(scores)
                print(f"     F1 Scores: {[f'{s:.3f}' for s in scores]}")
                print(f"     Avg F1: {mean_f1:.3f} +/- {std_f1:.3f}")
                cv_results['Genotype_F1_Macro_Mean'] = mean_f1
                cv_results['Genotype_F1_Macro_Std'] = std_f1
            except Exception as e:
                print(f"     ERROR during Genotype CV: {e}")
                traceback.print_exc()
                
        cv_end_time = time.time()
        print(f"   - CV Check completed (Time: {cv_end_time - cv_check_start_time:.2f} seconds).")
        return cv_results
        
    # Run cross-validation
    cv_results = perform_biological_signal_cv_v2(
        selected_features_for_transformer,
        config["output_dir"],
        os.path.join(config["output_dir"], "aligned_combined_metadata.csv"),
        feature_suffix=config["feature_suffix"]
    )
else:
    print("\n10.5 SKIPPING Cross-Validation Check.")


# Placeholders for Downstream Analysis Frameworks (Optional)
if not config["skip_hypothetical_outputs"]:
    print("\n11. Placeholders for Downstream Analysis Frameworks...")
    
    def placeholder_consensus_scoring(mofa_importance_file, transformer_importance_file):
        """Placeholder for consensus feature importance scoring framework"""
        print("\n   - Placeholder: Consensus Feature Importance Framework")
        return None
        
    def placeholder_cross_modal_analysis(transformer_attention_file):
        """Placeholder for cross-modal relationship extraction"""
        print("\n   - Placeholder: Cross-Modal Relationship Extraction")
        return None
        
    def placeholder_temporal_analysis(factors_df, factor_metadata_corr, selected_features_dict):
        """Placeholder for temporal patterns extraction"""
        print("\n   - Placeholder: Temporal Patterns Extraction")
        return None
        
    # Call placeholder functions
    placeholder_consensus_scoring("mofa_feature_importance_VIEW.csv", 
                                "hypothetical_transformer_importance_VIEW.csv")
    placeholder_cross_modal_analysis("hypothetical_transformer_attention.csv")
    placeholder_temporal_analysis(factors_df, factor_metadata_corr, selected_features_for_transformer)
else:
    print("\n11. SKIPPING Downstream Analysis Placeholders.")


# Final Summary & Next Steps
print("\n" + "="*80)
print("12. Final Summary & Next Steps:")
print("="*80)
print(f"   - MODE: Feature Selection Only (Target N={config['transformer_feature_cap']}"
      f"{config['feature_suffix']})")
if config["load_existing_model"]:
    print(f"   - Loaded existing MOFA+ model: {config['existing_model_path']}")
else:
    print(f"   - TRAINED new MOFA+ model and saved to: {config['output_dir']}")
print(f"   - Saved all outputs to directory: {config['output_dir']}")
print("   - Loaded/Recalculated Factors, Weights, Variance Explained.")
print("   - Performed enhanced validation: Factor-Metadata correlations, Genotype diffs, "
     "View coordination.")
if not config["skip_hypothetical_outputs"]:
    print("   - Generated hypothetical output files for Bootstrap Stability & Permutation Tests.")
print(f"   - Implemented hybrid stratified feature selection targeting "
     f"{config['transformer_feature_cap']} features per view.")
print(f"   - Generated Transformer input files (`transformer_input_{{view}}{config['feature_suffix']}.csv`).")
if not config["skip_cv_check"]:
    print("   - Performed simple CV check on selected features for signal preservation.")
if not config["skip_hypothetical_outputs"]:
    print("   - Added placeholders for Consensus Scoring, Cross-Modal, and Temporal analysis.")
print("\n   Next Steps:")
print("   - Review feature selection summary & variance proxy values in the output directory.")
if not config["skip_cv_check"]:
    print("   - Review the CV check results printed above.")
print(f"   - Use the generated `transformer_input_{{view}}{config['feature_suffix']}.csv` files for "
     f"downstream transformer-based analysis.")
print("   - Implement actual downstream analysis (Consensus, Cross-Modal, Temporal) using results.")
print("   - Consider running Bootstrap and Permutation tests for the original MOFA+ model if needed.")
print("   - Perform detailed biological interpretation.")

total_end_time = time.time()
print(f"\nTotal Script Runtime: {(total_end_time - start_time)/60:.2f} minutes")
print("\n"+"="*80)
print("MOFA+ Script Finished")
print("="*80)