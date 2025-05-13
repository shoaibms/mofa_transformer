# -*- coding: utf-8 -*-
"""
MOFA+ (Multi-Omics Factor Analysis) Implementation with Enhanced Validation

This script implements MOFA+ analysis on multi-view omics data (spectral and metabolite data).
It performs data integration, factor analysis, and various validation steps including:
- Alignment of data views using mapping file
- MOFA+ model training and optimization
- Factor analysis and visualization
- Feature selection for downstream analysis
- Preparation of transformer input files
- Cross-validation for biological signal preservation

MOFA+ reference: Argelaguet et al. (2020) 
"""
import pandas as pd
import numpy as np
import os
import json
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# Ensure mofapy2 is installed: pip install mofapy2
try:
    from mofapy2.run.entry_point import entry_point
except ImportError:
    print("ERROR: mofapy2 not found. Please install it: pip install mofapy2")
    exit()
import traceback
from scipy.stats import spearmanr, mannwhitneyu
# Ensure statsmodels is installed: pip install statsmodels
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("ERROR: statsmodels not found. Please install it: pip install statsmodels")
    exit()
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

print("MOFA+ Analysis Script Started - Transformer Prep & Enhanced Validation v3 (Hybrid Selection)")
print("="*80)
start_time = time.time()

# --- Configuration Section (Section 1) ---
print("1. Configuring paths and parameters...")
config = {
    "output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa",
    "figure_dir": r"C:/Users/ms/Desktop/hyper/output/mofa/figures",
    "data_paths": {
        "leaf_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv",
        "root_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv",
        "leaf_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv",
        "root_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv",
    },
    "metadata_file": r"C:/Users/ms/Desktop/hyper/output/mofa/all_metadata.json",
    "mapping_file": r"C:/Users/ms/Desktop/hyper/output/mofa/row_name_mapping.tsv",
    "view_names": ["leaf_spectral", "root_spectral", "leaf_metabolite", "root_metabolite"],
    "groups_names": ["group0"],
    "num_factors": 20,
    "maxiter": 1000,
    "convergence_mode": "medium",
    "drop_factor_threshold": 0.01,
    "outfile": "mofa_model_for_transformer.hdf5",
    "fdr_alpha": 0.05,
    "transformer_feature_cap": 200,
    "transformer_context_percentages": {"Genotype": 0.35, "Time": 0.35, "Other": 0.30},
    "min_variance_explained_proxy_report": 50.0,
    "relevant_factor_contexts": ['Genotype_Difference', 'Day_Correlation', 
                                 'Treatment_Correlation', 'Batch_Correlation'],
    "mapping_column_names": {
        "leaf_spectral": "hyper_l_w_augment",
        "root_spectral": "hyper_r_w_augment",
        "leaf_metabolite": "n_p_l2_augmt",
        "root_metabolite": "n_p_r2_augmt"
    },
    "reference_view": "leaf_spectral"  # Define which view provides the master index
}

# --- Sanity check context percentages ---
total_perc = sum(config["transformer_context_percentages"].values())
if not np.isclose(total_perc, 1.0):
    print(f"ERROR: transformer_context_percentages must sum to 1.0 (sums to {total_perc}). "
          f"Adjusting 'Other'.")
    current_other = config["transformer_context_percentages"].get('Other', 0)
    config["transformer_context_percentages"]['Other'] = max(0, current_other + (1.0 - total_perc))
    print(f"       Adjusted percentages: {config['transformer_context_percentages']}")
    new_total = sum(config["transformer_context_percentages"].values())
    if not np.isclose(new_total, 1.0):
        print("       ERROR: Could not adjust percentages.")
        exit()

os.makedirs(config["output_dir"], exist_ok=True)
os.makedirs(config["figure_dir"], exist_ok=True)
print(f"Output directory: {config['output_dir']}")
print(f"Figure directory: {config['figure_dir']}")
print(f"Targeting up to {config['transformer_feature_cap']} features per view for Transformer "
      f"(stratified selection).")

# --- Load Mapping File ---
print("\n1.5 Loading Row Name Mapping File...")
mapping_file_path = config["mapping_file"]
if not os.path.exists(mapping_file_path):
    print(f"   ERROR: Mapping file not found at: {mapping_file_path}")
    exit()
try:
    # Assuming tab-separated file based on row_name.txt content
    df_mapping = pd.read_csv(mapping_file_path, sep='\t')
    # --- Validate mapping columns exist ---
    expected_map_cols = list(config["mapping_column_names"].values())
    missing_map_cols = [col for col in expected_map_cols if col not in df_mapping.columns]
    if missing_map_cols:
        print(f"   ERROR: Mapping file is missing expected columns: {missing_map_cols}")
        print(f"          Expected columns based on config: {expected_map_cols}")
        print(f"          Columns found in file: {list(df_mapping.columns)}")
        exit()
    print(f"   Successfully loaded mapping file with {df_mapping.shape[0]} rows and columns: "
          f"{list(df_mapping.columns)}")
    # --- Define reference column name from mapping ---
    reference_map_col = config["mapping_column_names"][config["reference_view"]]
    print(f"   Using mapping column '{reference_map_col}' as the reference for master row names.")
except Exception as e:
    print(f"   ERROR loading or parsing mapping file '{mapping_file_path}': {e}")
    traceback.print_exc()
    exit()

# --- Section 2: Load Metadata Definitions ---
print("\n2. Loading metadata definitions...")
try:
    with open(config["metadata_file"], 'r') as f:
        all_metadata = json.load(f)
    metadata_columns = all_metadata['datasets']['leaf_spectral']['metadata_columns']
    print(f"Identified {len(metadata_columns)} metadata columns: {metadata_columns}")
    print(f"Aligning data using mapping file; original row order assumption less critical.")
except Exception as e:
    print(f"ERROR loading/parsing metadata: {e}")
    traceback.print_exc()
    exit()

# --- Section 3: Load and Prepare Data ---
print("\n3. Loading, preparing data, and aligning using mapping file...")
data_views_input_format = []
metadata_dfs_list = []
feature_names_dict = {}
master_reference_row_names = None  # Will hold the names from the reference view
expected_num_rows = None
reference_view_name = config["reference_view"]
reference_map_col = config["mapping_column_names"][reference_view_name]
master_index_col_name = 'MasterIndex'  # Internal consistent name

for i, view_name in enumerate(config["view_names"]):
    print(f"   - Processing view: {view_name}")
    file_path = config["data_paths"].get(view_name)
    current_map_col = config["mapping_column_names"][view_name]
    if not file_path or not os.path.exists(file_path):
        print(f"   ERROR: Data file not found: {file_path}")
        exit()

    try:
        df = pd.read_csv(file_path)
        print(f"     Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

        if 'Row_names' not in df.columns:
            print(f"   ERROR: 'Row_names' column not found in '{view_name}'.")
            exit()

        # --- Define expected rows based on first file ---
        if expected_num_rows is None:
            expected_num_rows = df.shape[0]
            print(f"     Expected number of rows set to: {expected_num_rows}")
            if len(df_mapping) != expected_num_rows:
                print(f"   ERROR: Row count mismatch between first data file ({expected_num_rows}) "
                      f"and mapping file ({len(df_mapping)}).")
                exit()
        elif df.shape[0] != expected_num_rows:
            print(f"   ERROR: Row count mismatch ({df.shape[0]} vs {expected_num_rows}) "
                  f"in '{view_name}'.")
            exit()
        else:
            print(f"     Row count ({df.shape[0]}) matches expected.")

        # --- Handle Reference View vs Other Views ---
        if view_name == reference_view_name:
            print(f"     Processing reference view '{view_name}'. Using 'Row_names' as Master Index.")
            # Ensure Row_names is string and unique
            df['Row_names'] = df['Row_names'].astype(str)
            if df['Row_names'].duplicated().any():
                print(f"   ERROR: Duplicate 'Row_names' in reference file '{view_name}'.")
                exit()

            # Rename Row_names -> MasterIndex and set as index
            df.rename(columns={'Row_names': master_index_col_name}, inplace=True)
            df.set_index(master_index_col_name, inplace=True)
            df_processed = df  # This is the dataframe to use going forward

            # Establish master order
            master_reference_row_names = df_processed.index.tolist()
            print(f"     Established master reference row names ({len(master_reference_row_names)} "
                  f"unique) from '{view_name}'.")
            print(f"     Sample Master IDs: {master_reference_row_names[:5]}")

        else:  # For non-reference views, perform the merge
            print(f"     Merging with mapping file on '{current_map_col}' to get reference names...")
            df['Row_names'] = df['Row_names'].astype(str)
            df_mapping[current_map_col] = df_mapping[current_map_col].astype(str)
            df_mapping[reference_map_col] = df_mapping[reference_map_col].astype(str)

            # Select only the two needed columns from mapping, drop duplicates just in case
            map_subset = df_mapping[[current_map_col, reference_map_col]].drop_duplicates()

            df_merged = pd.merge(
                df,
                map_subset,
                left_on='Row_names',
                right_on=current_map_col,
                how='left'
            )
            # Validate Merge
            if df_merged[reference_map_col].isnull().any():
                num_missing = df_merged[reference_map_col].isnull().sum()
                print(f"   ERROR: Merge failed for {num_missing} rows in '{view_name}'. "
                      f"Check 'Row_names' and mapping column '{current_map_col}'.")
                print(f"     Example 'Row_names' with failed merge: "
                      f"{df_merged[df_merged[reference_map_col].isnull()]['Row_names'].tolist()[:5]}")
                exit()
            if len(df_merged) != len(df):
                print(f"   ERROR: Merge changed row count for '{view_name}' "
                      f"({len(df)} vs {len(df_merged)}). Check for duplicates?")
                exit()
            print(f"     Merge successful. Added reference column '{reference_map_col}'.")

            # Rename the added reference column consistently
            df_merged.rename(columns={reference_map_col: master_index_col_name}, inplace=True)

            # Set the index to the MasterIndex
            if df_merged[master_index_col_name].duplicated().any():
                print(f"   ERROR: Duplicate '{master_index_col_name}' values found after merge "
                      f"for '{view_name}'. Check mapping.")
                exit()
            df_merged.set_index(master_index_col_name, inplace=True)
            df_processed = df_merged  # Use this df going forward
            print(f"     Set index to '{master_index_col_name}'.")

        # --- COMMON STEPS after index is set to MasterIndex ---
        if master_reference_row_names is None:  # Should only be None before processing reference view
            if view_name != reference_view_name:  # Error if master list not set after ref view processed
                print(f"   ERROR: Master reference row names were not set by reference view "
                      f"'{reference_view_name}'.")
                exit()
            # If it's the ref view itself and still None, the logic above failed
            elif view_name == reference_view_name:
                print(f"   ERROR: Master reference row names were not set correctly even for "
                      f"reference view.")
                exit()

        # --- Extract Metadata (index is MasterIndex) ---
        meta_df = df_processed[[col for col in metadata_columns 
                               if col in df_processed.columns]].copy()
        # Add suffix (index is already consistent MasterIndex)
        meta_df_renamed = meta_df.add_suffix(f'_{view_name}' if i > 0 else '')
        metadata_dfs_list.append(meta_df_renamed)

        # --- Extract Features (index is MasterIndex) ---
        # Define columns to drop based on whether it was the reference view or not
        if view_name == reference_view_name:
            # MasterIndex is the index, 'Row_names' was renamed. Drop original meta cols that still exist.
            cols_to_drop_for_features = [c for c in metadata_columns 
                                        if c != 'Row_names' and c in df_processed.columns]
        else:
            # MasterIndex is index. Drop original metadata cols + original Row_names + the column used for mapping
            cols_to_drop_for_features = metadata_columns + [current_map_col]
            # Only drop existing cols
            cols_to_drop_for_features = [c for c in cols_to_drop_for_features 
                                        if c in df_processed.columns]

        feature_df = df_processed.drop(columns=cols_to_drop_for_features)

        original_feature_names = feature_df.columns.tolist()
        unique_feature_names = [f"{feat}_{view_name}" for feat in original_feature_names]
        feature_names_dict[view_name] = unique_feature_names

        # --- Scale features ---
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        # Ensure scaled features maintain the MasterIndex order
        scaled_features_df = pd.DataFrame(scaled_features, 
                                          index=feature_df.index, 
                                          columns=feature_df.columns)
        # Reindex using the MASTER list to ensure perfect order and add missing rows if any
        scaled_features_df = scaled_features_df.reindex(master_reference_row_names)
        # Check for NaNs introduced by reindexing
        if scaled_features_df.isnull().values.any():
            print(f"     ERROR: NaNs found in '{view_name}' feature data after final reindexing. "
                  f"Check merge/indexing logic.")
            exit()
        # Convert back to numpy array for MOFA
        scaled_features_np = scaled_features_df.values
        data_views_input_format.append([scaled_features_np])

    except Exception as e:
        print(f"   ERROR processing view '{view_name}': {e}")
        traceback.print_exc()
        exit()

if len(data_views_input_format) != len(config["view_names"]):
    print("ERROR: Not all views processed.")
    exit()
if master_reference_row_names is None:
    print("ERROR: Master reference row names not established.")
    exit()

# --- Use reference names for MOFA model samples_names ---
samples_names_list = [master_reference_row_names]
features_names_list = [feature_names_dict[view_name] for view_name in config["view_names"]]

# --- Section 4: Combine Metadata ---
print("\n4. Combining metadata from all views...")
try:
    # Concatenate along columns - index should align now (MasterIndex)
    combined_metadata_df_raw = pd.concat(metadata_dfs_list, axis=1)

    # Now, resolve the suffixed columns back to original names
    final_metadata_cols_data = {}
    processed_cols = set()

    # Handle original metadata columns (prioritize non-suffixed, then suffixed)
    for col in metadata_columns:  # Iterate through the original desired columns
        if col in processed_cols:
            continue  # Skip if already handled

        # Check if the non-suffixed version exists and is good
        if col in combined_metadata_df_raw.columns and not combined_metadata_df_raw[col].isnull().all():
            final_metadata_cols_data[col] = combined_metadata_df_raw[col]
            processed_cols.add(col)
            # Remove suffixed versions if the primary one is used
            for i_suffix, vn_suffix in enumerate(config["view_names"]):
                suffixed_col_to_remove = f"{col}_{vn_suffix}" if i_suffix > 0 else col
                if suffixed_col_to_remove != col and suffixed_col_to_remove in combined_metadata_df_raw.columns:
                    processed_cols.add(suffixed_col_to_remove)

        else:  # Look for the first valid suffixed version
            found = False
            for i, vn in enumerate(config["view_names"]):
                suffixed_col = f"{col}_{vn}" if i > 0 else col
                if suffixed_col in combined_metadata_df_raw.columns and not combined_metadata_df_raw[suffixed_col].isnull().all():
                    final_metadata_cols_data[col] = combined_metadata_df_raw[suffixed_col]  # Assign to original name
                    processed_cols.add(col)
                    processed_cols.add(suffixed_col)
                    found = True
                    # Remove other suffixed versions
                    for i_suffix_rem, vn_suffix_rem in enumerate(config["view_names"]):
                        other_suffixed_col = f"{col}_{vn_suffix_rem}" if i_suffix_rem > 0 else col
                        if other_suffixed_col != suffixed_col and other_suffixed_col in combined_metadata_df_raw.columns:
                            processed_cols.add(other_suffixed_col)
                    break  # Stop after finding the first valid suffixed column
            if not found:
                print(f"   WARNING: Could not find non-null metadata for '{col}'. "
                      f"Using first column found (may be NaN).")
                # Fallback: take the first column found matching the pattern
                if col in combined_metadata_df_raw.columns:
                    final_metadata_cols_data[col] = combined_metadata_df_raw[col]
                    processed_cols.add(col)
                else:
                    for i, vn in enumerate(config["view_names"]):
                        suffixed_col = f"{col}_{vn}" if i > 0 else col
                        if suffixed_col in combined_metadata_df_raw.columns:
                            final_metadata_cols_data[col] = combined_metadata_df_raw[suffixed_col]
                            processed_cols.add(col)
                            processed_cols.add(suffixed_col)
                            break  # Take the first one available

    # Create final DataFrame from the resolved columns
    combined_metadata_df = pd.DataFrame(final_metadata_cols_data)

    # Set index using the master reference list
    if combined_metadata_df.index.equals(pd.Index(master_reference_row_names)):
        print("   Index of combined metadata matches master reference order.")
    else:
        print("   WARN: Index order may have changed during metadata combination. "
              "Reindexing to master order.")
        # Ensure index name is correct before reindexing if needed
        combined_metadata_df = combined_metadata_df.reindex(master_reference_row_names)

    combined_metadata_df.index.name = "MasterReferenceID"  # Set index name

    # Ensure correct data types
    if 'Day' in combined_metadata_df.columns:
        combined_metadata_df['Day'] = pd.to_numeric(combined_metadata_df['Day'], errors='coerce')
    if 'Genotype' in combined_metadata_df.columns:
        combined_metadata_df['Genotype'] = combined_metadata_df['Genotype'].astype(str)
    if 'Treatment' in combined_metadata_df.columns:
        combined_metadata_df['Treatment'] = combined_metadata_df['Treatment'].astype(str)
    if 'Batch' in combined_metadata_df.columns:
        combined_metadata_df['Batch'] = combined_metadata_df['Batch'].astype(str)

    print(f"Successfully combined metadata for {len(combined_metadata_df)} samples.")
    combined_metadata_outfile = os.path.join(config["output_dir"], "aligned_combined_metadata.csv")
    combined_metadata_df.to_csv(combined_metadata_outfile, index=True)  # Save WITH the MasterReferenceID index
    print(f"Combined aligned metadata saved to: {combined_metadata_outfile}")

except Exception as e:
    print(f"ERROR combining metadata: {e}")
    traceback.print_exc()
    exit()

# --- Section 5: Initialize MOFA+ and Set Data/Options ---
print("\n5. Initializing MOFA+ model and setting data/options...")
try:
    ent = entry_point()
    ent.set_data_options(scale_views=False)
    ent.set_data_matrix(data_views_input_format, views_names=config["view_names"],
                      groups_names=config["groups_names"], samples_names=samples_names_list,
                      features_names=features_names_list)
    ent.set_model_options(factors=config["num_factors"], spikeslab_weights=False, 
                        ard_factors=True, ard_weights=True)
    ent.set_train_options(iter=config["maxiter"], convergence_mode=config["convergence_mode"], 
                        dropR2=config["drop_factor_threshold"], gpu_mode=False, seed=42, 
                        verbose=False, startELBO=50)
    print("MOFA+ initialized and options set.")
except Exception as e:
    print(f"ERROR during MOFA+ initialization: {e}")
    traceback.print_exc()
    exit()

# --- Section 6: Build and Run MOFA+ Model ---
print("\n6. Building and running MOFA+ model...")
print(f"(Running up to {config['maxiter']} iterations... This can take time.)")
mofa_start_time = time.time()
try:
    ent.build()
    ent.run()
    print("   Inference completed.")
    model_outfile = os.path.join(config["output_dir"], config["outfile"])
    ent.save(model_outfile, save_data=False)
    print(f"   Model saved to {model_outfile}")
    mofa_end_time = time.time()
    print(f"   Time for MOFA+ Training: {mofa_end_time - mofa_start_time:.2f} seconds")
except Exception as e:
    print(f"ERROR: MOFA+ model training failed: {e}")
    traceback.print_exc()
    exit()

# --- Section 7: Basic Results Extraction ---
print("\n7. Extracting basic results (Active Factors)...")
extraction_start_time = time.time()
model_outfile_path = os.path.join(config["output_dir"], config["outfile"])
if not os.path.exists(model_outfile_path):
    print(f"ERROR: Trained model file not found: {model_outfile_path}")
    exit()
factors_df, weights_dict, variance_explained_df = None, {}, None
active_factors_indices, num_active_factors, factor_column_names = None, 0, []
try:
    with h5py.File(model_outfile_path, 'r') as hf:
        group_name_in_hdf5 = config["groups_names"][0]
        var_exp_factors_path = f'variance_explained/r2_per_factor/{group_name_in_hdf5}'
        if var_exp_factors_path in hf:
            var_exp_data = hf[var_exp_factors_path][()]
            total_r2_per_factor = var_exp_data.sum(axis=0)
            activity_threshold = config["drop_factor_threshold"]
            active_factors_mask = total_r2_per_factor > activity_threshold
            active_factors_indices = np.where(active_factors_mask)[0]
            num_active_factors = len(active_factors_indices)
            num_converged_factors = len(total_r2_per_factor)
            print(f"   Identified {num_active_factors} active factors (R2>{activity_threshold:.3f}) "
                  f"out of {num_converged_factors}.")
        else:
            print(f"   WARNING: VE path '{var_exp_factors_path}' not found. "
                  f"Assuming all factors active.")
            factors_path_temp = f'expectations/Z/{group_name_in_hdf5}'
            if factors_path_temp in hf:
                num_converged_factors = hf[factors_path_temp].shape[0]
                active_factors_indices = np.arange(num_converged_factors)
                num_active_factors = num_converged_factors
            else:
                print(f"   FATAL ERROR: Cannot determine factor count.")
                exit()
        if num_active_factors > 0:
            factor_column_names = [f"Factor{i+1}" for i in active_factors_indices]
            factors_path = f'expectations/Z/{group_name_in_hdf5}'
            if factors_path in hf:
                factors_raw_all = hf[factors_path][()]
                factors_raw_active = factors_raw_all[active_factors_indices, :]
                factors_active = factors_raw_active.T
                # --- Use master_reference_row_names for index ---
                if factors_active.shape[0] == len(master_reference_row_names):
                    factors_df = pd.DataFrame(factors_active, 
                                             index=master_reference_row_names, 
                                             columns=factor_column_names)
                    factors_df.index.name = "MasterReferenceID"
                    factors_outfile = os.path.join(config["output_dir"], 
                                                 "mofa_latent_factors_active.csv")
                    factors_df.to_csv(factors_outfile)
                    print(f"   - Active factors saved ({factors_df.shape}): {factors_outfile}")
                else:
                    factors_df = None
                    print(f"   ERROR: Factor dimensions mismatch samples "
                          f"({factors_active.shape[0]} vs {len(master_reference_row_names)}).")
            else:
                print(f"   ERROR: Factors path '{factors_path}' not found.")
                factors_df = None
            # --- Weights Extraction ---
            for view_name in config["view_names"]:
                weights_path = f'expectations/W/{view_name}'
                if weights_path in hf:
                    weights_raw_all = hf[weights_path][()]
                    weights_raw_active = weights_raw_all[active_factors_indices, :]
                    weights_active = weights_raw_active.T
                    view_unique_feature_names = feature_names_dict[view_name]
                    if weights_active.shape[0] == len(view_unique_feature_names):
                        weights_df = pd.DataFrame(weights_active, 
                                                 index=view_unique_feature_names, 
                                                 columns=factor_column_names)
                        weights_dict[view_name] = weights_df
                        weights_outfile = os.path.join(config["output_dir"], 
                                                     f"mofa_feature_weights_{view_name}_active.csv")
                        weights_df.to_csv(weights_outfile)
                    else:
                        print(f"   ERROR: Weight dimensions mismatch for '{view_name}'.")
                else:
                    print(f"   WARNING: Weights path '{weights_path}' not found for '{view_name}'.")
            print(f"   - Active weights extracted for {len(weights_dict)} views.")
            # --- Variance Explained ---
            if var_exp_factors_path in hf:
                variance_explained_active = var_exp_data[:, active_factors_indices]
                if variance_explained_active.shape[1] == num_active_factors:
                    variance_explained_active_transposed = variance_explained_active.T
                    variance_explained_df = pd.DataFrame(variance_explained_active_transposed, 
                                                       index=factor_column_names, 
                                                       columns=config["view_names"])
                    total_r2_per_view_active = variance_explained_df.sum(axis=0)
                    variance_explained_df.loc['Total R2 (Active Factors)'] = total_r2_per_view_active
                    var_exp_outfile = os.path.join(config["output_dir"], 
                                                 "mofa_variance_explained_active.csv")
                    variance_explained_df.to_csv(var_exp_outfile)
                    print(f"   - Active Variance explained saved ({variance_explained_df.shape}).")
                    print("     Total Variance Explained per View (by Active Factors):")
                    [print(f"       - {view}: {r2_val:.3f}") 
                     for view, r2_val in total_r2_per_view_active.items()]
                else:
                    print(f"   ERROR: Active VE dimensions mismatch.")
                    variance_explained_df = None
            elif 'var_exp_data' not in locals():
                print("   WARNING: VE data not loaded.")
                variance_explained_df = None
        else:
            print("   Skipping extraction: no active factors.")
            factors_df, weights_dict, variance_explained_df = None, {}, None
except Exception as e:
    print(f"ERROR during basic results extraction: {e}")
    traceback.print_exc()
    factors_df, weights_dict, variance_explained_df = None, {}, None
extraction_end_time = time.time()
print(f"   Basic Results Extraction Time: {extraction_end_time - extraction_start_time:.2f} seconds")

# --- Section 8: Basic Visualizations ---
print("\n8. Generating basic visualizations...")
vis_start_time = time.time()
try:
    if variance_explained_df is not None and not variance_explained_df.empty:
        print("   - Plotting Variance Explained...")
        # Plot total variance explained per view by active factors
        ve_totals = variance_explained_df.loc['Total R2 (Active Factors)']
        plt.figure(figsize=(8, 5))
        sns.barplot(x=ve_totals.index, y=ve_totals.values)
        plt.title('Total Variance Explained per View (Active Factors)')
        plt.ylabel('Total R2')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(config["figure_dir"], "variance_explained_total_per_view.png"))
        plt.close()

        # Plot variance explained per factor (heatmap) - Drop the total row for heatmap
        ve_heatmap_data = variance_explained_df.drop('Total R2 (Active Factors)', errors='ignore')
        if not ve_heatmap_data.empty:
            plt.figure(figsize=(10, max(6, len(ve_heatmap_data) * 0.4)))  # Adjust height dynamically
            sns.heatmap(ve_heatmap_data, cmap="viridis", annot=True, fmt=".2f", linewidths=.5)
            plt.title('Variance Explained per Factor and View (Active Factors)')
            plt.xlabel('View')
            plt.ylabel('Factor')
            plt.tight_layout()
            plt.savefig(os.path.join(config["figure_dir"], "variance_explained_heatmap.png"))
            plt.close()
        else:
            print("     Skipping VE heatmap (no individual factor data after dropping total).")

    else:
        print("   Skipping variance plots (data unavailable or empty).")

    # --- Plot Factors vs Metadata ---
    if factors_df is not None and not factors_df.empty and combined_metadata_df is not None:
        print("   - Plotting Factor Values vs Metadata...")
        # Ensure indices match before plotting
        common_index = factors_df.index.intersection(combined_metadata_df.index)
        if len(common_index) < len(factors_df):
            print(f"     WARN: Index mismatch between factors ({len(factors_df)}) and metadata "
                  f"({len(combined_metadata_df)}). Plotting only {len(common_index)} common samples.")
        plot_factors_df = factors_df.loc[common_index]
        plot_metadata_df = combined_metadata_df.loc[common_index]

        num_factors_to_plot = min(plot_factors_df.shape[1], 6)  # Plot first few factors
        metadata_to_plot = [c for c in ['Genotype', 'Day', 'Treatment'] 
                           if c in plot_metadata_df.columns]

        if num_factors_to_plot > 0 and metadata_to_plot:
            for factor in plot_factors_df.columns[:num_factors_to_plot]:
                for meta_col in metadata_to_plot:
                    plt.figure(figsize=(10, 6))
                    # Use the aligned dataframes
                    plot_data = pd.concat([plot_factors_df[factor], 
                                          plot_metadata_df[meta_col]], axis=1).dropna()

                    if pd.api.types.is_numeric_dtype(plot_data[meta_col]):
                        sns.scatterplot(data=plot_data, x=meta_col, y=factor, alpha=0.7)
                    else:
                        sns.boxplot(data=plot_data, x=meta_col, y=factor, 
                                   showfliers=False, color="lightblue")
                        sns.stripplot(data=plot_data, x=meta_col, y=factor, 
                                     alpha=0.5, color="black", jitter=True)

                    plt.title(f'{factor} vs {meta_col}')
                    plt.xlabel(meta_col)
                    plt.ylabel(factor)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config["figure_dir"], 
                                          f"factor_plot_{factor}_vs_{meta_col}.png"))
                    plt.close()
        else:
            print("     Skipping factor vs metadata plots (no factors/metadata to plot or "
                  "alignment failed).")
    else:
        print("   Skipping factor vs metadata plots (data unavailable or empty).")

except Exception as e:
    print(f"   ERROR during visualization: {e}")
    traceback.print_exc()

vis_end_time = time.time()
print(f"   Visualizations generation attempt finished (Time: {vis_end_time - vis_start_time:.2f} seconds)")

# --- Section 8.5: Advanced Visualizations ---
print("\n8.5 Generating Advanced Visualizations...")
adv_vis_start_time = time.time()

# Check if necessary dataframes/dicts exist AND are not empty
required_data_available = True
plot_error_msg = "   Skipping advanced visualizations because: "

# Check factors_df
if 'factors_df' not in locals() or factors_df is None or factors_df.empty:
    required_data_available = False
    plot_error_msg += "factors_df is missing or empty. "
# Check combined_metadata_df
if 'combined_metadata_df' not in locals() or combined_metadata_df is None or combined_metadata_df.empty:
    required_data_available = False
    plot_error_msg += "combined_metadata_df is missing or empty. "
# Check weights_dict
if 'weights_dict' not in locals() or not weights_dict:  # Checks if dict exists and is not empty
    required_data_available = False
    plot_error_msg += "weights_dict is missing or empty. "
# Check factor_metadata_corr (needed for trajectories)
if 'factor_metadata_corr' not in locals() or factor_metadata_corr is None or factor_metadata_corr.empty:
    # Don't stop all plots, just print warning for trajectory plot later
    print("     WARN: factor_metadata_corr is missing or empty, factor trajectory plot will be skipped.")
# Check selected_features_for_transformer (needed for heatmap)
if 'selected_features_for_transformer' not in locals() or not selected_features_for_transformer:
    # Don't stop all plots, just print warning for heatmap later
    print("     WARN: selected_features_for_transformer is missing or empty, feature heatmap will be skipped.")
# Check variance_explained_df and genotype_diff_results
if 'variance_explained_df' not in locals() or variance_explained_df is None or variance_explained_df.empty:
    print("     WARN: variance_explained_df is missing or empty, factor selection for weights plot might be limited.")
if 'genotype_diff_results' not in locals() or genotype_diff_results is None or genotype_diff_results.empty:
    print("     WARN: genotype_diff_results is missing or empty, factor selection for weights plot might be limited.")


if required_data_available:
    print("   - Plotting Factor Trajectories over Time...")
    try:
        # Check again specifically for factor_metadata_corr before this plot
        if 'factor_metadata_corr' in locals() and factor_metadata_corr is not None and not factor_metadata_corr.empty:
            day_corr_factors = factor_metadata_corr[
                (factor_metadata_corr['Metadata'] == 'Day') & factor_metadata_corr['Significant_FDR']
            ]['Factor'].unique().tolist()

            if day_corr_factors:
                print(f"     Plotting trajectories for Day-correlated factors: {day_corr_factors}")
                plot_data_time = pd.merge(
                    factors_df[day_corr_factors],
                    combined_metadata_df[['Day', 'Genotype', 'Treatment']],
                    left_index=True, right_index=True
                )
                plot_data_time['Day'] = pd.to_numeric(plot_data_time['Day'], errors='coerce')
                plot_data_time.dropna(subset=['Day'], inplace=True)

                for factor in day_corr_factors:
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=plot_data_time, x='Day', y=factor, 
                               hue='Genotype', style='Treatment', 
                               marker='o', errorbar=('ci', 95))
                    plt.title(f'Mean Trajectory of {factor} over Time')
                    plt.xlabel('Day') 
                    plt.ylabel(f'{factor} Score (Mean +/- 95% CI)')
                    plt.xticks(sorted(plot_data_time['Day'].unique()))
                    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig(os.path.join(config["figure_dir"], 
                                          f"factor_trajectory_{factor}_vs_Day.png"))
                    plt.close()
            else:
                print("     Skipping factor trajectories (no factors significantly correlated with Day).")
        else:
            print("     Skipping factor trajectories (correlation data unavailable).")
    except Exception as e:
        print(f"   ERROR plotting factor trajectories: {e}")
        traceback.print_exc()


    print("\n   - Plotting Top Feature Loadings for Key Factors...")
    try:
        factors_to_plot_weights = []
        if 'variance_explained_df' in locals() and variance_explained_df is not None and not variance_explained_df.empty:
            top_var_factors = variance_explained_df.drop('Total R2 (Active Factors)', 
                                                      errors='ignore').sum(axis=1).nlargest(2).index.tolist()
            factors_to_plot_weights.extend(top_var_factors)
        if 'genotype_diff_results' in locals() and genotype_diff_results is not None and not genotype_diff_results.empty:
            top_geno_factors = genotype_diff_results[
                genotype_diff_results['Significant_FDR']
            ].nsmallest(2, 'P_value_FDR')['Factor'].tolist()
            factors_to_plot_weights.extend(top_geno_factors)
        # Re-check factor_metadata_corr for time factors
        if 'factor_metadata_corr' in locals() and factor_metadata_corr is not None and not factor_metadata_corr.empty:
            day_corr_df_sig = factor_metadata_corr[
                (factor_metadata_corr['Metadata'] == 'Day') & factor_metadata_corr['Significant_FDR']
            ]
            if not day_corr_df_sig.empty:
                top_time_factors = day_corr_df_sig.nsmallest(2, 'P_value_FDR')['Factor'].tolist()
                factors_to_plot_weights.extend(top_time_factors)

        factors_to_plot_weights = sorted(list(set(factors_to_plot_weights)))
        n_top_features = 15

        if not factors_to_plot_weights:
            print("     Skipping top feature loadings plot (no key factors identified based on available data).")
        else:
            print(f"     Plotting top {n_top_features} feature loadings for factors: {factors_to_plot_weights}")
            for factor in factors_to_plot_weights:
                for view_name, weights_df in weights_dict.items():
                    if factor not in weights_df.columns:
                        continue
                    factor_weights = weights_df[factor].dropna().sort_values(ascending=False)
                    if len(factor_weights) < n_top_features * 2:
                        continue
                    top_positive = factor_weights.head(n_top_features)
                    top_negative = factor_weights.tail(n_top_features).sort_values(ascending=True)
                    top_features_factor = pd.concat([top_positive, top_negative])
                    plt.figure(figsize=(10, max(8, len(top_features_factor) * 0.3)))
                    sns.barplot(x=top_features_factor.values, y=top_features_factor.index, palette="vlag")
                    cleaned_labels = [label.replace(f"_{view_name}", "") 
                                    for label in top_features_factor.index]
                    plt.yticks(ticks=range(len(cleaned_labels)), labels=cleaned_labels)
                    plt.title(f'Top {n_top_features} +/- Feature Loadings for {factor} in {view_name}')
                    plt.xlabel('Weight (Loading)')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config["figure_dir"], 
                                          f"feature_loadings_{factor}_{view_name}.png"))
                    plt.close()
    except Exception as e:
        print(f"   ERROR plotting feature loadings: {e}")
        traceback.print_exc()

    print("\n   - Plotting Heatmap of Selected Features...")
    try:
        # Check if selected_features_for_transformer exists and is not empty
        if 'selected_features_for_transformer' in locals() and selected_features_for_transformer:
            views_for_heatmap = [vn for vn in ["leaf_spectral", "leaf_metabolite"] 
                               if vn in selected_features_for_transformer 
                               and selected_features_for_transformer[vn]]
            if not views_for_heatmap:
                print("     Skipping selected features heatmap (no selected features for target views "
                      "leaf_spectral/leaf_metabolite).")
            else:
                metadata_heatmap = combined_metadata_df[['Genotype', 'Treatment', 'Day', 'Batch']].copy()
                for view_name in views_for_heatmap:
                    print(f"     Generating heatmap for selected features in '{view_name}'...")
                    infile = os.path.join(config["output_dir"], f"transformer_input_{view_name}.csv")
                    if not os.path.exists(infile):
                        print(f"       Skipping heatmap for {view_name}: File not found.")
                        continue
                    data_df = pd.read_csv(infile)
                    if 'Row_names' not in data_df.columns:
                        print(f"       Skipping heatmap: 'Row_names' missing in {infile}")
                        continue
                    data_df.set_index('Row_names', inplace=True)  # Index is MasterReferenceID
                    selected_original_names = [f.replace(f"_{view_name}", "") 
                                            for f in selected_features_for_transformer[view_name]]
                    selected_cols_in_data = [col for col in selected_original_names 
                                           if col in data_df.columns]
                    if not selected_cols_in_data:
                        print(f"       Skipping heatmap: No selected features found in columns of {infile}")
                        continue
                    heatmap_data = data_df[selected_cols_in_data]
                    common_index_heatmap = heatmap_data.index.intersection(metadata_heatmap.index)
                    heatmap_data = heatmap_data.loc[common_index_heatmap]
                    metadata_groups = metadata_heatmap.loc[common_index_heatmap]
                    scaler_heatmap = StandardScaler()
                    heatmap_data_scaled = scaler_heatmap.fit_transform(heatmap_data)
                    heatmap_data_scaled_df = pd.DataFrame(heatmap_data_scaled, 
                                                        index=heatmap_data.index, 
                                                        columns=heatmap_data.columns)
                    col_colors_map = {}
                    palettes = {"Genotype": "Set1", "Treatment": "Set2", 
                              "Day": "viridis", "Batch": "Set3"}
                    for col in ['Genotype', 'Treatment', 'Day', 'Batch']:
                        if col in metadata_groups.columns:
                            unique_vals = sorted(metadata_groups[col].unique())
                            lut = dict(zip(unique_vals, 
                                         sns.color_palette(palettes.get(col, "coolwarm"), 
                                                          len(unique_vals))))
                            col_colors_map[col] = metadata_groups[col].map(lut)
                    col_colors = pd.DataFrame(col_colors_map) if col_colors_map else None
                    max_features_heatmap = 75
                    if heatmap_data_scaled_df.shape[1] > max_features_heatmap:
                        print(f"       Reducing features shown in heatmap to {max_features_heatmap} "
                              f"(highest variance).")
                        top_var_features = heatmap_data_scaled_df.var(axis=0).nlargest(
                            max_features_heatmap).index
                        heatmap_data_to_plot = heatmap_data_scaled_df[top_var_features]
                    else:
                        heatmap_data_to_plot = heatmap_data_scaled_df
                    print(f"       Plotting clustermap for {heatmap_data_to_plot.shape[1]} features...")
                    g = sns.clustermap(
                        heatmap_data_to_plot.T, cmap="vlag", col_colors=col_colors, 
                        col_cluster=True, row_cluster=True, 
                        figsize=(12, max(10, heatmap_data_to_plot.shape[1] * 0.15)), 
                        linewidths=0.0, xticklabels=False, z_score=0
                    )
                    g.ax_heatmap.set_ylabel("Selected Features")
                    g.ax_heatmap.set_xlabel("Samples (Clustered)")
                    plt.suptitle(f"Heatmap of Selected {view_name.replace('_',' ').title()} "
                               f"Features (Scaled)", y=1.02)
                    plt.savefig(os.path.join(config["figure_dir"], 
                                          f"heatmap_selected_features_{view_name}.png"), 
                              bbox_inches='tight')
                    plt.close()
        else:
            print("     Skipping selected features heatmap (feature selection dictionary missing or empty).")
    except Exception as e:
        print(f"   ERROR plotting selected feature heatmap: {e}")
        traceback.print_exc()

    print("\n   - Plotting Factor Correlation Heatmap...")
    try:
        if factors_df is not None and not factors_df.empty and factors_df.shape[1] > 1:
            factor_corr = factors_df.corr()
            plt.figure(figsize=(max(8, factor_corr.shape[0]*0.6), 
                              max(6, factor_corr.shape[0]*0.6)))
            sns.heatmap(factor_corr, cmap="coolwarm", annot=True, fmt=".2f", 
                      linewidths=.5, center=0)
            plt.title('Correlation Between MOFA+ Factors')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(config["figure_dir"], "factor_correlation_heatmap.png"))
            plt.close()
        else:
            print("     Skipping factor correlation heatmap (<= 1 factor or factors_df is empty/None).")
    except Exception as e:
        print(f"   ERROR plotting factor correlation heatmap: {e}")
        traceback.print_exc()

else:  # This else corresponds to the main 'required_data_available' check
    print(plot_error_msg)  # Print the consolidated error message

adv_vis_end_time = time.time()
print(f"   Advanced visualizations attempt finished (Time: {adv_vis_end_time - adv_vis_start_time:.2f} seconds)")

# --- Section 9: Enhanced Validation and Biological Analysis ---
print("\n9. Enhanced Validation and Biological Analysis...")
enhanced_start_time = time.time()

# --- 9.1 Factor-Metadata Association ---
def calculate_factor_metadata_associations(factors_df, metadata_df, fdr_alpha=0.05, output_dir="."):
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
        if c not in ["Row_names", "Vac_id", "Entry", "Tissue.type"]  # Exclude specific columns
           and c != metadata_aligned.index.name  # Exclude index name itself
           and metadata_aligned[c].nunique() > 1
    ]
    print(f"     Testing associations for metadata columns: {metadata_cols_to_test}")

    for factor in factors_aligned.columns:
        for meta_col in metadata_cols_to_test:
            # Data is already aligned, just drop NAs for the specific pair
            temp_df = pd.concat([factors_aligned[factor], metadata_aligned[meta_col]], axis=1).dropna()

            if temp_df.shape[0] < 5:
                continue
            if temp_df[meta_col].nunique() < 2:
                continue

            factor_values = temp_df[factor].values
            meta_values = temp_df[meta_col].values

            try:
                meta_values_numeric = pd.to_numeric(meta_values)
                corr, p_value = spearmanr(factor_values, meta_values_numeric)
                note = None
            except ValueError:
                try:
                    meta_values_codes, _ = pd.factorize(meta_values)
                    if len(np.unique(meta_values_codes)) < 2:
                        continue
                    corr, p_value = spearmanr(factor_values, meta_values_codes)
                    note = 'Used factorized codes'
                except Exception as e_cat:
                    print(f"     WARN: Could not correlate {factor} vs {meta_col} (Categorical): {e_cat}")
                    corr, p_value, note = np.nan, np.nan, 'Correlation failed'

            if not np.isnan(corr) and not np.isnan(p_value):
                results_list.append({
                    'Factor': factor, 
                    'Metadata': meta_col, 
                    'Correlation': corr, 
                    'P_value': p_value, 
                    'Note': note
                })

    if not results_list:
        print("     No valid correlations calculated.")
        return None
    
    results_df = pd.DataFrame(results_list)
    if 'P_value' in results_df.columns and not results_df['P_value'].isnull().all():
        p_vals_clean = results_df['P_value'].dropna()
        if not p_vals_clean.empty:
            reject, pvals_corrected, _, _ = multipletests(p_vals_clean, alpha=fdr_alpha, method='fdr_bh')
            results_df.loc[p_vals_clean.index, 'P_value_FDR'] = pvals_corrected
            results_df.loc[p_vals_clean.index, 'Significant_FDR'] = reject
        else:
            results_df['P_value_FDR'], results_df['Significant_FDR'] = np.nan, False
    else:
        results_df['P_value_FDR'], results_df['Significant_FDR'] = np.nan, False
    
    results_df['Significant_FDR'] = results_df['Significant_FDR'].fillna(False).astype(bool)
    results_df = results_df.sort_values(by=['Significant_FDR', 'P_value_FDR'], 
                                      ascending=[False, True])
    outfile = os.path.join(output_dir, "mofa_factor_metadata_associations_spearman.csv")
    results_df.to_csv(outfile, index=False)
    print(f"     Factor-Metadata associations saved to {outfile}")
    return results_df

# --- 9.2 Bootstrap Validation (Outline & Hypothetical Output) ---
def outline_bootstrap_analysis(weights_dict, factors_df, output_dir, num_bootstrap_runs=100):
    print("\n   - Bootstrap Validation (Outline & Hypothetical Output):")
    print(f"     PURPOSE: Assess stability of feature weights/factor loadings.")
    print(f"     PROCESS: Requires running MOFA+ ~{num_bootstrap_runs} times on bootstrapped data samples.")
    print(f"              Factors across runs need alignment (complex step).")
    print(f"              Feature stability = frequency feature is 'important' for an aligned factor.")
    print(f"     OUTPUT: Generating a *hypothetical* stability file structure.")
    
    if not weights_dict or factors_df is None or factors_df.empty:
        print("     Skipping hypothetical output generation (weights/factors unavailable).")
        return None
    
    hypothetical_stability_data = []
    for view_name, weights_df in weights_dict.items():
        if weights_df is None or weights_df.empty:
            continue
        for factor_name in factors_df.columns:
            if factor_name not in weights_df.columns:
                continue
            abs_weights = weights_df[factor_name].abs().sort_values(ascending=False)
            n_feat = len(abs_weights)
            rank = 1
            if n_feat == 0:
                continue
            # Simplified random stability score generation
            stability_scores = np.random.uniform(0.0, 1.0, n_feat)
            for feature_name, abs_weight in abs_weights.items():
                hypothetical_stability_data.append({
                    'Factor': factor_name, 
                    'View': view_name, 
                    'Feature': feature_name, 
                    'StabilityScore': stability_scores[rank-1], 
                    'Rank_AbsWeight_OriginalRun': rank
                })
                rank += 1
    
    if not hypothetical_stability_data:
        print("     No data to create hypothetical stability file.")
        return None
    
    stability_df = pd.DataFrame(hypothetical_stability_data).sort_values(
        by=['Factor', 'View', 'StabilityScore'], ascending=[True, True, False])
    outfile = os.path.join(output_dir, "mofa_bootstrap_stability_HYPOTHETICAL.csv")
    stability_df.to_csv(outfile, index=False)
    print(f"     Hypothetical bootstrap stability structure saved to: {outfile}")
    print(f"     NOTE: Scores in this file are RANDOM placeholders for illustration.")
    return stability_df

# --- 9.3 Permutation Testing (Outline & Hypothetical Output) ---
def outline_permutation_testing(factors_df, factor_metadata_corr, output_dir, num_permutations=1000):
    print("\n   - Permutation Testing (Outline & Hypothetical Output):")
    print(f"     PURPOSE: Assess significance of factor-metadata associations against random chance.")
    print(f"     PROCESS: Requires calculating association metric (e.g., correlation) between")
    print(f"              factor scores and ~{num_permutations} permutations of metadata labels.")
    print(f"              Compare real association metric to the null distribution from permutations.")
    print(f"     OUTPUT: Generating a *hypothetical* permutation p-value file structure.")
    
    if factors_df is None or factors_df.empty or factor_metadata_corr is None or factor_metadata_corr.empty:
        print("     Skipping hypothetical output generation (factors/correlations unavailable).")
        return None
    
    hypothetical_perm_data = []
    tested_metadata_vars = factor_metadata_corr['Metadata'].unique()
    for factor_name in factors_df.columns:
        for meta_var in tested_metadata_vars:
            real_corr_row = factor_metadata_corr[
                (factor_metadata_corr['Factor'] == factor_name) & 
                (factor_metadata_corr['Metadata'] == meta_var)
            ]
            if real_corr_row.empty:
                continue
            
            real_metric_value = real_corr_row['Correlation'].iloc[0]
            real_p_value_fdr = real_corr_row['P_value_FDR'].iloc[0]
            is_significant_fdr = real_corr_row['Significant_FDR'].iloc[0]
            
            # Generate hypothetical p-value based on significance
            if is_significant_fdr:
                hypothetical_p_val = np.random.uniform(0.0001, 0.049)
            elif not np.isnan(real_p_value_fdr):
                hypothetical_p_val = np.random.uniform(0.1, 1.0)
            else:
                hypothetical_p_val = np.nan
                
            hypothetical_perm_data.append({
                'Factor': factor_name, 
                'Tested_Metadata': meta_var, 
                'Real_Association_Metric': real_metric_value, 
                'Real_P_value_FDR': real_p_value_fdr, 
                'Real_Significant_FDR': is_significant_fdr, 
                'Hypothetical_Permutation_P_Value': hypothetical_p_val
            })
    
    if not hypothetical_perm_data:
        print("     No data to create hypothetical permutation results file.")
        return None
    
    perm_df = pd.DataFrame(hypothetical_perm_data).sort_values(
        by=['Tested_Metadata', 'Hypothetical_Permutation_P_Value'])
    outfile = os.path.join(output_dir, "mofa_permutation_test_results_HYPOTHETICAL.csv")
    perm_df.to_csv(outfile, index=False)
    print(f"     Hypothetical permutation test structure saved to: {outfile}")
    print(f"     NOTE: P-values in this file are RANDOM placeholders for illustration.")
    return perm_df

# --- 9.4 Genotype Comparison Analysis ---
def analyze_genotype_differences(factors_df, metadata_df, fdr_alpha=0.05, output_dir="."):
    print("\n   - Analyzing Genotype Differences in Factors (Mann-Whitney U)...")
    if factors_df is None or metadata_df is None or factors_df.empty or metadata_df.empty:
        print("     Skipping Genotype analysis (Input data unavailable).")
        return None
    if 'Genotype' not in metadata_df.columns:
        print("     Skipping Genotype analysis ('Genotype' column not in metadata).")
        return None
    
    # Align data first
    common_index = factors_df.index.intersection(metadata_df.index)
    if len(common_index) == 0:
        print("     ERROR: No common index between factors and metadata for Genotype analysis.")
        return None
    
    factors_aligned = factors_df.loc[common_index]
    metadata_aligned = metadata_df.loc[common_index]

    genotypes = metadata_aligned['Genotype'].unique()
    genotypes = [g for g in genotypes if pd.notna(g)]
    if len(genotypes) != 2:
        print(f"     Skipping Genotype analysis (Expected 2 non-NA genotypes, found {len(genotypes)}).")
        return None
    
    g1_label, g2_label = genotypes[0], genotypes[1]
    print(f"     Comparing genotypes: '{g1_label}' vs '{g2_label}'")
    
    results = []
    for factor in factors_aligned.columns:
        g1_values = factors_aligned.loc[metadata_aligned['Genotype'] == g1_label, factor].dropna().values
        g2_values = factors_aligned.loc[metadata_aligned['Genotype'] == g2_label, factor].dropna().values
        
        min_samples_per_group = 3
        if len(g1_values) < min_samples_per_group or len(g2_values) < min_samples_per_group:
            continue
        
        try:
            stat, p_value = mannwhitneyu(g1_values, g2_values, alternative='two-sided', use_continuity=True)
            results.append({
                'Factor': factor, 
                'Comparison': f"{g1_label}_vs_{g2_label}", 
                'Statistic_U': stat, 
                'P_value': p_value, 
                f'N_{g1_label}': len(g1_values), 
                f'N_{g2_label}': len(g2_values), 
                f'Mean_{g1_label}': np.mean(g1_values), 
                f'Mean_{g2_label}': np.mean(g2_values)
            })
        except Exception as e:
            print(f"     Error during Mann-Whitney U test for {factor}: {e}")
    
    if not results:
        print("     No genotype comparisons performed.")
        return None
    
    results_df = pd.DataFrame(results)
    if not results_df.empty and 'P_value' in results_df.columns:
        p_vals_clean = results_df['P_value'].dropna()
        if not p_vals_clean.empty:
            reject, pvals_corrected, _, _ = multipletests(p_vals_clean, alpha=fdr_alpha, method='fdr_bh')
            results_df.loc[p_vals_clean.index, 'P_value_FDR'] = pvals_corrected
            results_df.loc[p_vals_clean.index, 'Significant_FDR'] = reject
        else:
            results_df['P_value_FDR'], results_df['Significant_FDR'] = np.nan, False
    else:
        results_df['P_value_FDR'], results_df['Significant_FDR'] = np.nan, False
    
    results_df['Significant_FDR'] = results_df['Significant_FDR'].fillna(False).astype(bool)
    results_df = results_df.sort_values(by=['Significant_FDR', 'P_value_FDR'], 
                                      ascending=[False, True])
    outfile = os.path.join(output_dir, "mofa_genotype_diff_factors.csv")
    results_df.to_csv(outfile, index=False)
    print(f"     Genotype difference analysis saved to {outfile}")
    return results_df

# --- 9.5 Tissue Coordination Analysis ---
def analyze_tissue_coordination(variance_explained_df, r2_threshold=0.01, output_dir="."):
    print("\n   - Analyzing Tissue/Modality Coordination...")
    if variance_explained_df is None or variance_explained_df.empty:
        print("     Skipping Tissue Coordination (Variance data unavailable).")
        return None
    
    ve_factors = variance_explained_df.drop('Total R2 (Active Factors)', errors='ignore')
    if ve_factors.empty:
        print("     Skipping Tissue Coordination (No factor variance data available).")
        return None
    
    tissue_pairs = [('leaf_spectral', 'root_spectral'), ('leaf_metabolite', 'root_metabolite')]
    modality_pairs = [('leaf_spectral', 'leaf_metabolite'), ('root_spectral', 'root_metabolite')]
    cross_pairs = [('leaf_spectral', 'root_metabolite'), ('root_spectral', 'leaf_metabolite')]
    all_pairs = tissue_pairs + modality_pairs + cross_pairs
    
    print(f"     Identifying factors explaining > {r2_threshold*100:.1f}% variance simultaneously "
          f"in paired views:")
    
    coord_dfs = []
    for v1, v2 in all_pairs:
        if v1 in ve_factors.columns and v2 in ve_factors.columns:
            coord_factors_mask = (ve_factors[v1] > r2_threshold) & (ve_factors[v2] > r2_threshold)
            coord_factors = ve_factors[coord_factors_mask]
            if not coord_factors.empty:
                num_found = len(coord_factors)
                print(f"       - {v1} <--> {v2}: Found {num_found} coordinating factors.")
                coord_data = coord_factors[[v1, v2]].copy()
                coord_data['Coordination_Type'] = f'{v1}_vs_{v2}'
                coord_data['Factor'] = coord_data.index
                coord_dfs.append(coord_data.reset_index(drop=True))
    
    if not coord_dfs:
        print("     No coordinating factors found.")
        return None
    
    all_coord_factors_df = pd.concat(coord_dfs, ignore_index=True)
    cols_order = ['Factor', 'Coordination_Type'] + [col for col in all_coord_factors_df.columns 
                                                  if col not in ['Factor', 'Coordination_Type']]
    all_coord_factors_df = all_coord_factors_df[cols_order].sort_values(
        by=['Coordination_Type', 'Factor'])
    outfile = os.path.join(output_dir, "mofa_tissue_coordination_factors.csv")
    all_coord_factors_df.to_csv(outfile, index=False)
    print(f"     View coordination analysis saved to {outfile}")
    return all_coord_factors_df

# --- Run Enhanced Validation ---
print("   Executing validation functions...")
# Pass the potentially modified combined_metadata_df (with BaseSampleID index)
factor_metadata_corr = calculate_factor_metadata_associations(
    factors_df, combined_metadata_df, config["fdr_alpha"], config["output_dir"])
hypothetical_stability_df = outline_bootstrap_analysis(
    weights_dict, factors_df, config["output_dir"])
hypothetical_perm_results_df = outline_permutation_testing(
    factors_df, factor_metadata_corr, config["output_dir"])
genotype_diff_results = analyze_genotype_differences(
    factors_df, combined_metadata_df, config["fdr_alpha"], config["output_dir"])
tissue_coordination_results = analyze_tissue_coordination(
    variance_explained_df, config["drop_factor_threshold"], config["output_dir"])

enhanced_val_end_time = time.time()
print(f"   Enhanced Validation Functions Executed (Time: "
      f"{enhanced_val_end_time - enhanced_start_time:.2f} seconds)")

# --- Section 10: Feature Selection & Transformer Preparation ---
print("\n10. Feature Selection & Transformer Data Preparation (Hybrid Stratified Method)...")
transformer_prep_start_time = time.time()

# --- 10.1 Calculate Overall Feature Importance ---
def calculate_overall_feature_importance(weights_dict, all_relevant_factor_names, output_dir="."):
    """ Calculates importance across a specified list of relevant factors. """
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
        outfile = os.path.join(output_dir, f"mofa_feature_importance_{view_name}.csv")
        view_importance_df.to_csv(outfile)
    
    print(f"     Overall importance calculated and saved for {len(overall_importance_dict)} views.")
    return overall_importance_dict

# --- Determine ALL Relevant Factors ---
print("   - Identifying ALL relevant factors based on significance for Overall Importance...")
all_relevant_factors_list = []
significant_dfs_map = {
    'Genotype_Difference': genotype_diff_results,
    'Day_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Day')] 
                       if factor_metadata_corr is not None else None,
    'Treatment_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Treatment')] 
                            if factor_metadata_corr is not None else None,
    'Batch_Correlation': factor_metadata_corr[(factor_metadata_corr['Metadata'] == 'Batch')] 
                         if factor_metadata_corr is not None else None
}

unique_relevant_factor_names = set()
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
    outfile_rel = os.path.join(config["output_dir"], "mofa_relevant_factors_list.csv")
    all_relevant_factors_combined_df.to_csv(outfile_rel, index=False)
    print(f"     Identified {len(all_relevant_factors_names_list)} unique relevant factors overall: "
          f"{all_relevant_factors_names_list}")
    print(f"     Relevant factors list (with context) saved to: {outfile_rel}")
else:
    print("     WARNING: No relevant factors identified.")
    all_relevant_factors_names_list = []

# --- Calculate Overall Importance ---
overall_feature_importance_dict = calculate_overall_feature_importance(
    weights_dict, all_relevant_factors_names_list, config["output_dir"])

# --- 10.2 Placeholder: Bootstrap Stability Integration ---
def placeholder_integrate_bootstrap_stability(overall_importance_dict, hypothetical_stability_df):
    print("\n   - Placeholder: Integrating Bootstrap Stability...")
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
            if 'Feature' in merged_df.columns:
                merged_df.drop(columns=['Feature'], inplace=True)
            merged_df['StabilityScore'] = merged_df['StabilityScore'].fillna(0)
            updated_importance[view_name] = merged_df
        else:
            current_importance_df['StabilityScore'] = 0.0
            updated_importance[view_name] = current_importance_df
    
    print("     (NOTE: Stability scores used here are RANDOM placeholders).")
    return updated_importance

# --- Apply Placeholder ---
overall_feature_importance_dict_with_stability = placeholder_integrate_bootstrap_stability(
    overall_feature_importance_dict, hypothetical_stability_df)

# --- 10.3 Select Top Features using Hybrid Stratified Method ---
def select_features_hybrid_stratified(
    weights_dict, factor_metadata_corr, genotype_diff_results, 
    variance_explained_df, config, output_dir="."):
    
    print("\n   - Selecting Top Features using Hybrid Stratified Method...")
    if not weights_dict:
        print("     ERROR: Weights dictionary is empty.")
        return None, None
    
    fdr_alpha = config["fdr_alpha"]
    max_cap_per_view = config["transformer_feature_cap"]
    context_percentages = config["transformer_context_percentages"]
    min_var_proxy_report_threshold = config["min_variance_explained_proxy_report"]
    
    selected_features_final = {}
    variance_proxy_report = {}
    stratification_summary = []
    context_factors = {'Genotype': [], 'Time': [], 'Other': []}
    all_relevant_fnames_in_selection = set()
    
    # Identify factors for each context
    if (genotype_diff_results is not None and not genotype_diff_results.empty 
            and 'Significant_FDR' in genotype_diff_results.columns):
        geno_factors = genotype_diff_results[genotype_diff_results['Significant_FDR']]['Factor'].unique().tolist()
        if geno_factors:
            context_factors['Genotype'] = geno_factors
            all_relevant_fnames_in_selection.update(geno_factors)
            print(f"     Context 'Genotype': Identified {len(geno_factors)} significant factors.")
    
    if factor_metadata_corr is not None and not factor_metadata_corr.empty:
        time_df = factor_metadata_corr[
            (factor_metadata_corr['Metadata'] == 'Day') & factor_metadata_corr['Significant_FDR']]
        if not time_df.empty:
            time_factors = time_df['Factor'].unique().tolist()
            context_factors['Time'] = time_factors
            all_relevant_fnames_in_selection.update(time_factors)
            print(f"     Context 'Time' (Day): Identified {len(time_factors)} significant factors.")
        
        other_factors_set = set()
        treat_df = factor_metadata_corr[
            (factor_metadata_corr['Metadata'] == 'Treatment') & factor_metadata_corr['Significant_FDR']]
        if not treat_df.empty:
            other_factors_set.update(treat_df['Factor'].unique())
            print(f"     Context 'Other': Identified {len(treat_df['Factor'].unique())} Treatment factors.")
        
        batch_df = factor_metadata_corr[
            (factor_metadata_corr['Metadata'] == 'Batch') & factor_metadata_corr['Significant_FDR']]
        if not batch_df.empty:
            other_factors_set.update(batch_df['Factor'].unique())
            print(f"     Context 'Other': Identified {len(batch_df['Factor'].unique())} Batch factors.")
        
        if other_factors_set:
            context_factors['Other'] = list(other_factors_set)
            all_relevant_fnames_in_selection.update(list(other_factors_set))
            print(f"     Context 'Other' (Treat+Batch): Total {len(context_factors['Other'])} "
                  f"unique significant factors.")
    
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
        print(f"\n     --- Processing view: {view_name} ---")
        if weights_df is None or weights_df.empty:
            print("       Skipping view: No weights data.")
            selected_features_final[view_name] = []
            variance_proxy_report[view_name] = 0.0
            continue
        
        overall_importance_view = overall_importance_for_capping.get(view_name)
        if overall_importance_view is None or overall_importance_view.empty:
            print("       WARNING: Overall importance for capping missing.")
        
        context_importance_scores = {}
        top_features_by_context = {}
        selected_counts = {}
        
        for context, factors in context_factors.items():
            target_n = int(round(max_cap_per_view * context_percentages[context]))
            print(f"       Context '{context}': Target N = {target_n}")
            
            factors_in_view = [f for f in factors if f in weights_df.columns]
            if not factors_in_view:
                print(f"         - No significant factors for context in view '{view_name}'.")
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
        
        combined_features = [feature for sublist in top_features_by_context.values() for feature in sublist]
        unique_features = sorted(list(set(combined_features)))
        n_unique = len(unique_features)
        print(f"       Combined unique features from contexts: {n_unique}")
        
        final_selected_features_view = []
        if n_unique <= max_cap_per_view:
            print(f"       Total unique features ({n_unique}) within cap ({max_cap_per_view}). Using all.")
            final_selected_features_view = unique_features
        else:
            print(f"       Total unique features ({n_unique}) exceeds cap ({max_cap_per_view}). Pruning...")
            if overall_importance_view is not None:
                overall_imp_filtered = overall_importance_view.loc[unique_features]['OverallImportance']
                overall_imp_sorted = overall_imp_filtered.sort_values(ascending=False)
                final_selected_features_view = overall_imp_sorted.head(max_cap_per_view).index.tolist()
                print(f"       Pruned to {len(final_selected_features_view)} features.")
            else:
                print(f"       WARNING: Cannot prune. Keeping all {n_unique} features (exceeds cap).")
                final_selected_features_view = unique_features
        
        selected_features_final[view_name] = final_selected_features_view
        final_n = len(final_selected_features_view)
        
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
                weights_relevant = weights_df[relevant_factors_in_view]
                sum_sq_weights_total = (weights_relevant**2).sum().sum()
                if sum_sq_weights_total > 1e-9:
                    weights_selected = weights_relevant.loc[final_selected_features_view]
                    sum_sq_weights_selected = (weights_selected**2).sum().sum()
                    proxy_pct = (sum_sq_weights_selected / sum_sq_weights_total) * 100
                    print(f"       Variance Explained Proxy (SumSqWeights): {proxy_pct:.2f}%")
                    if proxy_pct < min_var_proxy_report_threshold:
                        print(f"         - Note: Variance proxy below threshold "
                              f"{min_var_proxy_report_threshold}%.")
                else:
                    print("       Total sum sq weights near zero. Cannot calc proxy.")
                    proxy_pct = 0.0
            except Exception as e_vp:
                print(f"       ERROR calculating variance proxy: {e_vp}")
                proxy_pct = np.nan
        
        variance_proxy_report[view_name] = proxy_pct
        outfile_selected = os.path.join(output_dir, 
                                      f"mofa_selected_hybrid_{final_n}_features_{view_name}.txt")
        with open(outfile_selected, 'w') as f:
            f.write('\n'.join(final_selected_features_view))
        print(f"       Selected features list saved to: {outfile_selected}")
        
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
    summary_outfile = os.path.join(output_dir, "mofa_feature_selection_hybrid_summary.csv")
    summary_df.to_csv(summary_outfile, index=False)
    print(f"\n   Hybrid feature selection summary saved to: {summary_outfile}")
    return selected_features_final, variance_proxy_report

# --- Run the Hybrid Feature Selection ---
selected_features_for_transformer, variance_report = select_features_hybrid_stratified(
    weights_dict, factor_metadata_corr, genotype_diff_results, 
    variance_explained_df, config, config["output_dir"])

if selected_features_for_transformer is None:
    print("\nERROR: Hybrid feature selection failed (returned None).")
    exit()
elif not selected_features_for_transformer:
    print("\nERROR: Hybrid feature selection returned an empty dictionary.")
    exit()

transformer_prep_end_time = time.time()
print(f"   Feature Selection & Transformer Data Preparation Time: {transformer_prep_end_time - transformer_prep_start_time:.2f} seconds")

# --- Section 11: Transformer Data Preparation ---
print("\n11. Transformer Data Preparation...")
transformer_data_prep_start_time = time.time()

# --- 11.1 Prepare Transformer Input Files ---
def prepare_transformer_input_files(selected_features_for_transformer, config, output_dir="."):
    print("\n   - Preparing Transformer Input Files...")
    if not selected_features_for_transformer:
        print("     ERROR: No selected features for transformer.")
        return None
    
    for view_name, features in selected_features_for_transformer.items():
        print(f"     - Preparing input for view: {view_name}")
        if not features:
            print(f"       WARNING: No features selected for view '{view_name}'.")
            continue
        
        # Prepare input file
        outfile = os.path.join(output_dir, f"transformer_input_{view_name}.csv")
        with open(outfile, 'w') as f:
            f.write('\n'.join(features))
        print(f"       Selected features saved to: {outfile}")
    
    print("   - Transformer input files prepared successfully.")
    return True

# --- Run the Transformer Data Preparation ---
success = prepare_transformer_input_files(selected_features_for_transformer, config, config["output_dir"])

if not success:
    print("\nERROR: Transformer data preparation failed.")
    exit()

transformer_data_prep_end_time = time.time()
print(f"   Transformer Data Preparation Time: {transformer_data_prep_end_time - transformer_data_prep_start_time:.2f} seconds")

# --- Section 10.4 Create Transformer Input Files ---
def create_transformer_input_files_using_mapping(
    selected_features_dict,
    original_data_paths,
    mapping_df,
    mapping_column_names,
    reference_view_name,
    master_reference_ids,
    metadata_columns,
    output_dir
):
    print("\n   - Creating Transformer Input Files (Using Mapping File for alignment)...")
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
    master_index_col_name = 'MasterIndex'  # Consistent internal name

    # Make sure master list is actually a list
    if not isinstance(master_reference_ids, list):
        print("   ERROR: master_reference_ids must be a list.")
        master_reference_ids = list(master_reference_ids)  # Attempt conversion

    # Ensure master list elements are strings for reliable comparison/indexing
    master_reference_ids = [str(item) for item in master_reference_ids]

    output_metadata_cols_final = metadata_columns[:]  # Copy original list

    for view_name, feature_list in selected_features_dict.items():
        print(f"\n     ===== START Processing view: {view_name} =====")
        original_file_path = original_data_paths.get(view_name)
        current_map_col = mapping_column_names[view_name]
        if not original_file_path or not os.path.exists(original_file_path):
            print(f"     ERROR: Original data file not found: '{original_file_path}'. Skipping.")
            continue
        if not feature_list:
            print(f"     INFO: No features selected for '{view_name}'. File will contain metadata only.")

        try:
            df_orig = pd.read_csv(original_file_path)
            print(f"       Loaded original data: {df_orig.shape}")
            if 'Row_names' not in df_orig.columns:
                print(f"       ERROR: 'Row_names' column missing in {original_file_path}")
                continue

            df_orig['Row_names'] = df_orig['Row_names'].astype(str)  # Ensure string type

            # --- Handle Reference View vs Other Views for getting MasterIndex ---
            if view_name == reference_view_name:
                print(f"       Processing reference view '{view_name}'. Using 'Row_names' as Master Index.")
                if df_orig['Row_names'].duplicated().any():
                    print(f"       ERROR: Duplicate 'Row_names' in reference file '{view_name}'.")
                    continue
                # Rename Row_names -> MasterIndex
                df_orig.rename(columns={'Row_names': master_index_col_name}, inplace=True)
                # Check if master index column was created
                if master_index_col_name not in df_orig.columns:
                    print(f"      ERROR: Failed to rename 'Row_names' to '{master_index_col_name}'")
                    continue

                df_with_master_index = df_orig  # Use this dataframe directly
                print(f"       Using existing reference IDs as '{master_index_col_name}'.")

            else:  # For non-reference views, perform the merge
                print(f"       Merging with mapping on '{current_map_col}'...")
                # Prepare mapping subset (ensure types and uniqueness)
                mapping_df[current_map_col] = mapping_df[current_map_col].astype(str)
                mapping_df[reference_map_col] = mapping_df[reference_map_col].astype(str)
                map_subset = mapping_df[[current_map_col, reference_map_col]].drop_duplicates()

                df_merged = pd.merge(df_orig, map_subset, 
                                    left_on='Row_names', 
                                    right_on=current_map_col, 
                                    how='left')
                # Validate Merge
                if df_merged[reference_map_col].isnull().any():
                    print(f"       ERROR: Merge failed for {df_merged[reference_map_col].isnull().sum()} rows.")
                    continue
                if len(df_merged) != len(df_orig):
                    print(f"       ERROR: Merge changed row count.")
                    continue
                df_merged.rename(columns={reference_map_col: master_index_col_name}, inplace=True)
                
                # Drop the original Row_names column from the source file and the mapping column
                df_merged.drop(columns=['Row_names', current_map_col], inplace=True, errors='ignore')
                print(f"       Dropped original 'Row_names' and mapping column '{current_map_col}'.")
                df_with_master_index = df_merged  # Use the merged dataframe

            # --- Alignment Steps (Common to all views now) ---
            # Set index to MasterIndex
            if df_with_master_index[master_index_col_name].duplicated().any():
                print(f"       ERROR: Duplicate '{master_index_col_name}' values found before setting index.")
                continue
            df_with_master_index.set_index(master_index_col_name, inplace=True)
            print(f"       Set index to '{master_index_col_name}'. Shape: {df_with_master_index.shape}")

            # --- CRITICAL: Reindex to align rows according to the MASTER list ---
            # Ensure the index of df_with_master_index is string type BEFORE reindexing
            df_with_master_index.index = df_with_master_index.index.astype(str)

            print(f"       Reindexing using {len(master_reference_ids)} master reference IDs...")
            df_aligned = df_with_master_index.reindex(master_reference_ids)
            print(f"       Shape after reindexing: {df_aligned.shape}")

            # Check for NaNs *after* reindexing
            num_all_nan_rows = df_aligned.isnull().all(axis=1).sum()
            if num_all_nan_rows > 0:
                print(f"     WARNING: {num_all_nan_rows} out of {len(df_aligned)} rows are "
                      f"entirely NaN after reindexing.")
                missing_ids_in_reindex = df_aligned[df_aligned.isnull().all(axis=1)].index.tolist()
                print(f"       Example missing/NaN IDs: {missing_ids_in_reindex[:5]}...")

            df_aligned.reset_index(inplace=True)  # Bring MasterIndex back as a column

            # --- Feature Selection ---
            original_feature_names_to_select = []
            if feature_list:
                original_feature_names_attempt = [f.replace(f"_{view_name}", "") for f in feature_list]
                original_feature_names_to_select = [col for col in original_feature_names_attempt 
                                                   if col in df_aligned.columns]
                print(f"       Found {len(original_feature_names_to_select)} of {len(feature_list)} "
                      f"selected feature columns.")
                if len(original_feature_names_to_select) == 0 and feature_list:
                    print(f"       >>> CRITICAL WARNING: ZERO original features found.")

            # --- Column Assembly ---
            # Start with MasterIndex + original metadata that exist in df_aligned
            cols_to_keep = [master_index_col_name] + [col for col in output_metadata_cols_final 
                                                    if col in df_aligned.columns]
            # Add selected features
            cols_to_keep.extend(original_feature_names_to_select)
            final_cols = list(dict.fromkeys(col for col in cols_to_keep if col in df_aligned.columns))
            if not final_cols:
                print(f"       ERROR: No columns left to keep. Skipping.")
                continue
            if master_index_col_name not in final_cols:
                print(f"       ERROR: '{master_index_col_name}' lost. Skipping.")
                continue

            filtered_df = df_aligned[final_cols].copy()

            # --- Rename MasterIndex to Row_names for final output consistency ---
            filtered_df.rename(columns={master_index_col_name: 'Row_names'}, inplace=True)
            print(f"       Renamed '{master_index_col_name}' to 'Row_names' for output file.")

            # Save (Rows are already ordered by master_reference_ids due to reindex)
            outfile = os.path.join(output_dir, f"transformer_input_{view_name}.csv")
            filtered_df.to_csv(outfile, index=False, na_rep='NA')  # Save WITHOUT index, represent NaNs explicitly
            print(f"     ✅ Standardized transformer input file saved: {outfile} ({filtered_df.shape})")

            meta_in_final = set(filtered_df.columns) & set(output_metadata_cols_final + ['Row_names'])
            feature_count_in_final = len(set(filtered_df.columns) - meta_in_final)
            print(f"       File contains {feature_count_in_final} selected feature cols + "
                  f"{len(meta_in_final)} metadata cols.")
            # Check if data looks empty
            if filtered_df.drop(columns=['Row_names'], errors='ignore').isnull().all().all():
                print(f"       >>> CRITICAL WARNING: Data values in saved file for '{view_name}' "
                      f"appear to be all NaN/missing!")

        except Exception as e:
            print(f"     ❌ ERROR creating transformer input file for '{view_name}': {e}")
            traceback.print_exc()
        print(f"     ===== END Processing view: {view_name} =====")

    print("\n   Transformer input file generation attempt finished.")

# --- CALL the Transformer File Creation Function ---
create_transformer_input_files_using_mapping(
    selected_features_for_transformer,
    config["data_paths"],
    df_mapping,  # Pass loaded mapping df
    config["mapping_column_names"],  # Pass column name dictionary
    config["reference_view"],  # Pass reference view name
    master_reference_row_names,  # Pass the established master ID list
    metadata_columns,  # Pass original metadata list
    config["output_dir"]
)

transformer_prep_end_time = time.time()
print(f"   Feature Selection & Transformer Prep Completed (Time: "
      f"{transformer_prep_end_time - transformer_prep_start_time:.2f} seconds)")

# --- Section 11: Placeholders for Downstream Analysis Frameworks ---
print("\n11. Placeholders for Downstream Analysis Frameworks...")

def placeholder_consensus_scoring(mofa_importance_file, transformer_importance_file):
    """Placeholder function for outlining consensus feature importance scoring."""
    print("\n   - Placeholder: Consensus Feature Importance Framework")
    print("     PURPOSE: Combine importance scores from MOFA+ and Transformer.")
    print(f"     INPUTS: MOFA+ importance ('{mofa_importance_file}'), hypothetical Transformer importance "
          f"('{transformer_importance_file}').")
    print("     METHOD: Requires Z-score normalization, weighted combination (e.g., 0.6*MOFA + "
          "0.4*Transformer), and confidence tiering.")
    print("     STATUS: Not implemented. Needs actual Transformer results.")
    return None  # Return None as it's just a placeholder

def placeholder_cross_modal_analysis(transformer_attention_file):
    """Placeholder function for outlining cross-modal analysis using Transformer attention."""
    print("\n   - Placeholder: Cross-Modal Relationship Extraction")
    print("     PURPOSE: Identify direct spectral-metabolite links using Transformer attention.")
    print(f"     INPUTS: Hypothetical Transformer attention weights ('{transformer_attention_file}').")
    print("     METHOD: Requires extracting high attention weights between features of different "
          "modalities, potentially visualized as networks.")
    print("     STATUS: Not implemented. Needs actual Transformer results.")
    return None  # Return None as it's just a placeholder

def placeholder_temporal_analysis(factors_df, factor_metadata_corr, selected_features_dict):
    """Placeholder function for outlining temporal analysis of factors and features."""
    print("\n   - Placeholder: Temporal Patterns Extraction")
    print("     PURPOSE: Analyze how factors and top features evolve over time (Day).")
    print("     INPUTS: MOFA+ factors, factor-day correlations, selected features.")
    print("     METHOD: Plot time-correlated factors over days; analyze trajectories of top "
          "features associated with these factors.")
    
    # Example Check: See if Day correlated factors exist
    if factor_metadata_corr is not None and not factor_metadata_corr.empty:
        # Ensure required columns exist before filtering
        if ('Metadata' in factor_metadata_corr.columns and 
                'Significant_FDR' in factor_metadata_corr.columns and 
                'Factor' in factor_metadata_corr.columns):
            day_corr = factor_metadata_corr[
                (factor_metadata_corr['Metadata'] == 'Day') & 
                (factor_metadata_corr['Significant_FDR'])
            ]
            if not day_corr.empty:
                print(f"     (Found {len(day_corr)} factors significantly correlated with Day, "
                      f"e.g., {day_corr['Factor'].tolist()[:3]}, which could be analyzed temporally).")
        else:
            print("     (WARN: Could not check for Day correlations - required columns missing "
                  "in factor_metadata_corr).")

    print("     STATUS: Basic checks possible, full analysis requires dedicated methods.")
    return None  # Return None as it's just a placeholder

# --- Ensure variables exist ---
if 'factors_df' not in locals():
    factors_df = None
if 'factor_metadata_corr' not in locals():
    factor_metadata_corr = None
if 'selected_features_for_transformer' not in locals():
    selected_features_for_transformer = None

# --- Call placeholder functions ---
placeholder_consensus_scoring("mofa_feature_importance_VIEW.csv", 
                             "hypothetical_transformer_importance_VIEW.csv")
placeholder_cross_modal_analysis("hypothetical_transformer_attention.csv")
placeholder_temporal_analysis(factors_df, factor_metadata_corr, selected_features_for_transformer)

# --- Section 12: Final Summary & Next Steps ---
print("\n" + "="*80)
print("12. Final Summary & Next Steps:")
print("="*80)
print("   - Ran MOFA+ and extracted results for active factors.")
print("   - Performed enhanced validation: Factor-Metadata correlations, Genotype diffs, View coordination.")
print("   - **Generated HYPOTHETICAL output files for Bootstrap Stability & Permutation Tests.**")
print("   - **Implemented HYBRID STRATIFIED Feature Selection** based on biological context.")
print("     - Calculated context-specific importance & variance proxy.")
print("   - Generated Transformer input files (`transformer_input_{view}.csv`) using selected features and "
      "**aligned using mapping file**.")
print("   - **Performed simple CV check** on selected features for signal preservation.")
print("   - Added placeholders for Consensus Scoring, Cross-Modal, and Temporal analysis.")
print("\n   Further Steps:")
print("   - Review feature selection summary & variance proxy values.")
print("   - **Review the CV check results** printed above.")
print("   - Implement Transformer model using the generated `transformer_input_{view}.csv` files.")
print("   - Run Transformer to get attention scores/importance.")
print("   - Implement Consensus Scoring function.")
print("   - **Crucially:** *Actually run* Bootstrap and Permutation tests for MOFA+.")
print("   - Perform detailed biological interpretation.")

total_end_time = time.time()
print(f"\nTotal Script Runtime: {(total_end_time - start_time)/60:.2f} minutes")
print("\n"+"="*80)
print("MOFA+ Script Finished")
print("="*80)