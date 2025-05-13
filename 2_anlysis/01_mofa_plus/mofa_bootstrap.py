#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOFA+ Bootstrap Stability Analysis Script

This script performs bootstrap analysis to assess the stability of features
in a Multi-Omics Factor Analysis (MOFA+) model across multiple runs.
It identifies which features are consistently selected across bootstrap iterations,
providing confidence measures for feature importance.
"""
import pandas as pd
import numpy as np
import os
import json
import h5py
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point
import traceback
import time
from collections import defaultdict

print("="*60)
print("MOFA+ Bootstrap Stability Analysis Script")
print("WARNING: This script re-runs MOFA+ multiple times and")
print("         will take a VERY LONG time to complete!")
print("="*60)
start_time = time.time()

# --- 1. Configuration ---
print("1. Configuring paths and parameters...")
config = {
    # --- Paths ---
    "output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa_bootstrap",
    "data_paths": {
        "leaf_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv",
        "root_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv",
        "leaf_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv",
        "root_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv",
    },
    "metadata_file": r"C:/Users/ms/Desktop/hyper/output/mofa/all_metadata.json",
    "view_names": ["leaf_spectral", "root_spectral", "leaf_metabolite", 
                   "root_metabolite"],
    "groups_names": ["group0"],
    # --- MOFA+ Parameters ---
    "num_factors": 20,
    "maxiter": 500,
    "convergence_mode": "medium",
    "drop_factor_threshold": 0.01,
    # --- Bootstrap Parameters ---
    "n_bootstrap": 100,
    "stability_weight_threshold": 0.1
}
os.makedirs(config["output_dir"], exist_ok=True)
print(f"Output directory: {config['output_dir']}")
print(f"Number of bootstrap iterations: {config['n_bootstrap']}")
print(f"Number of factors per run: {config['num_factors']}")
print(f"Max iterations per run: {config['maxiter']}")

# --- 2. Load Metadata Definitions ---
print("\n2. Loading metadata definitions...")
try:
    with open(config["metadata_file"], 'r') as f:
        all_metadata = json.load(f)
    metadata_columns = all_metadata['datasets']['leaf_spectral']['metadata_columns']
    print(f"   Identified {len(metadata_columns)} metadata columns.")
except Exception as e:
    print(f"ERROR loading/parsing metadata: {e}")
    traceback.print_exc()
    exit()

# --- 3. Load and Prepare ORIGINAL Data ONCE ---
print("\n3. Loading and preparing original data views...")
original_data_scaled = {}  # Store scaled original data (Samples x Features)
feature_names_dict = {}    # Store unique feature names
master_row_names_order = None
expected_num_rows = None

# Load, Scale, Store original data
for i, view_name in enumerate(config["view_names"]):
    print(f"   - Processing view: {view_name}")
    file_path = config["data_paths"].get(view_name)
    if not file_path or not os.path.exists(file_path):
        print(f"   ERROR: Data file not found: {file_path}")
        exit()
    try:
        df = pd.read_csv(file_path)
        if expected_num_rows is None:
            expected_num_rows = df.shape[0]
            if 'Row_names' in df.columns:
                master_row_names_order = df['Row_names'].astype(str).str.strip().tolist()
            else:
                print(f"   ERROR: 'Row_names' column not found in first file '{view_name}'.")
                exit()
        elif df.shape[0] != expected_num_rows:
            print(f"   ERROR: Row count mismatch in '{view_name}'.")
            exit()

        feature_df = df.drop(columns=metadata_columns)
        original_feature_names = feature_df.columns.tolist()
        # Create unique names *before* scaling/storing feature names list
        unique_feature_names = [f"{feat}_{view_name}" for feat in original_feature_names]
        feature_names_dict[view_name] = unique_feature_names

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)  # Samples x Features
        original_data_scaled[view_name] = scaled_features  # Store scaled data
        print(f"     Loaded and scaled {view_name}: {scaled_features.shape}")

    except Exception as e:
        print(f"   ERROR processing view '{view_name}': {e}")
        traceback.print_exc()
        exit()

print("   Original data loaded and scaled.")
n_samples = expected_num_rows
samples_names_list = [master_row_names_order]  # Keep original names for reference if needed
# Ensure features_names_list is created *after* feature_names_dict is fully populated
features_names_list = [feature_names_dict[view_name] for view_name in config["view_names"]]

# --- 4. Bootstrap Loop ---
print(f"\n4. Starting {config['n_bootstrap']} Bootstrap Iterations...")

# Use defaultdict to easily count feature selections
feature_selection_counts = defaultdict(lambda: defaultdict(int))  # {view: {feature: count}}
run_successful = [False] * config['n_bootstrap']

for i_boot in range(config['n_bootstrap']):
    iter_start_time = time.time()
    print(f"--- Iteration {i_boot+1}/{config['n_bootstrap']} ---")

    # Define temporary file path *before* the try block
    tmp_hdf5_path = os.path.join(config["output_dir"], f"temp_bootstrap_model_{i_boot+1}.hdf5")

    try:
        # 1. Create bootstrapped sample indices (with replacement)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # 2. Create bootstrapped data views
        bootstrap_data_views = []
        for view_name in config["view_names"]:
            # Select rows based on bootstrap indices from the *original scaled* data
            boot_view_data = original_data_scaled[view_name][bootstrap_indices, :]
            bootstrap_data_views.append([boot_view_data])  # MOFA expects list of lists (one inner list per group)

        # 3. Setup and Run MOFA+ for this bootstrap sample
        print(f"     Setting up MOFA+...")
        ent_boot = entry_point()
        ent_boot.set_data_options(scale_views=False)  # Data is already scaled
        # Provide unique feature names for this run
        ent_boot.set_data_matrix(
            bootstrap_data_views,
            views_names=config["view_names"],
            groups_names=config["groups_names"],
            features_names=features_names_list  # Use same unique feature names
        )

        ent_boot.set_model_options(
            factors=config["num_factors"], 
            ard_factors=True, 
            ard_weights=True
        )
        ent_boot.set_train_options(
            iter=config['maxiter'],
            convergence_mode=config['convergence_mode'],
            dropR2=config['drop_factor_threshold'],
            gpu_mode=False,
            seed=(42 + i_boot),  # Different seed per run
            verbose=True,
            startELBO=10
        )

        print(f"     Running MOFA+...")
        ent_boot.build()
        ent_boot.run()
        print(f"     MOFA+ run finished. Saving temporary model...")

        # Explicitly save the model to the temporary file
        ent_boot.save(tmp_hdf5_path, save_data=False)  # Save model structure and expectations
        print(f"     Temporary model saved to: {tmp_hdf5_path}")

        # 4. Extract Weights and Identify 'Important' Features from the saved file
        print(f"     Extracting weights and identifying important features...")
        boot_weights_dict = {}
        active_factors_found = False

        # Open the explicitly saved temporary file
        with h5py.File(tmp_hdf5_path, 'r') as hf:
            # Determine active factors for *this* bootstrap run
            group_name_hdf5 = config["groups_names"][0]
            var_exp_path = f'variance_explained/r2_per_factor/{group_name_hdf5}'
            boot_active_indices = np.arange(config['num_factors'])  # Default to all if check fails

            if var_exp_path in hf:
                var_exp_boot = hf[var_exp_path][()]  # Shape likely Views x Factors

                if var_exp_boot.ndim != 2:  # Basic sanity check
                    print(f"     WARN: Unexpected variance explained dimensions in bootstrap run {i_boot+1}.")
                    active_factors_found = False
                else:
                    # Correctly sum across views (axis 0) to get R2 per factor
                    total_r2_boot = var_exp_boot.sum(axis=0)

                    # Use the configured drop threshold to define active factors
                    boot_active_mask = total_r2_boot > config['drop_factor_threshold']
                    boot_active_indices = np.where(boot_active_mask)[0]  # Correct 0-based indices

                    if len(boot_active_indices) > 0:
                        active_factors_found = True
                        num_converged_boot = var_exp_boot.shape[1]  # Number of factors in this run
                        print(f"     Found {len(boot_active_indices)} active factors (out of {num_converged_boot}).")
                    else:
                        print("     WARN: No active factors met the R2 threshold in this bootstrap run.")
                        active_factors_found = False  # Explicitly set

            else:
                print(f"     WARN: Variance explained path '{var_exp_path}' not found in temp file.")
                active_factors_found = False  # Cannot determine activity

            if not active_factors_found:
                print("     Skipping weight extraction for this run as no active factors determined.")
                # No need for 'continue' here, flow will naturally proceed to finally block

            else:
                # Extract weights only for active factors
                for view_idx, view_name in enumerate(config["view_names"]):
                    weights_path = f'expectations/W/{view_name}'
                    if weights_path in hf:
                        # Weights are stored Factors x Features in HDF5
                        weights_all_factors = hf[weights_path][()]  # Shape: (NumFactors, NumFeatures)
                        # Select rows corresponding to active factors
                        weights_raw = weights_all_factors[boot_active_indices, :]  # ActiveFactors x Features
                        weights = weights_raw.T  # Transpose to Features x ActiveFactors

                        # Check dimensions
                        view_features = features_names_list[view_idx]  # Get unique names for this view
                        if weights.shape[0] != len(view_features):
                            print(f"     ERROR: Weight dimension mismatch for {view_name}!")
                            continue

                        # Check if any weight for a feature exceeds threshold across active factors
                        max_abs_weight_per_feature = np.max(np.abs(weights), axis=1)
                        important_features_mask = max_abs_weight_per_feature > config['stability_weight_threshold']
                        important_features_indices = np.where(important_features_mask)[0]

                        # Get the unique names of important features
                        important_feature_names = [view_features[idx] for idx in important_features_indices]

                        # Increment count for selected features
                        for feature_name in important_feature_names:
                            feature_selection_counts[view_name][feature_name] += 1
                    else:
                        print(f"     WARN: Weights path '{weights_path}' not found for {view_name}.")

        # Only mark as successful if active factors were found and weights processed
        if active_factors_found:
            run_successful[i_boot] = True  # Mark run as successful for weight extraction

    except Exception as e:
        print(f"     ERROR in bootstrap iteration {i_boot+1}: {e}")
        traceback.print_exc()
        # Keep run_successful[i_boot] as False

    finally:
        # Clean up the explicitly saved temporary HDF5 file
        if os.path.exists(tmp_hdf5_path):
            try:
                os.remove(tmp_hdf5_path)
            except OSError as e_rm:
                print(f"     WARN: Could not remove temp file {tmp_hdf5_path}: {e_rm}")

    iter_end_time = time.time()
    print(f"--- Iteration {i_boot+1} finished (Time: {iter_end_time - iter_start_time:.2f} sec) ---")


# --- 5. Calculate and Save Stability Scores ---
print("\n5. Calculating final stability scores...")

num_successful_runs = sum(run_successful)
if num_successful_runs == 0:
    print("ERROR: No bootstrap runs completed successfully. Cannot calculate stability.")
    exit()
elif num_successful_runs < config['n_bootstrap']:
    print(f"WARNING: Only {num_successful_runs}/{config['n_bootstrap']} runs completed successfully.")

stability_results_list = []
# Make sure to iterate through views using the correct list
for view_idx, view_name in enumerate(config["view_names"]):
    # Use the corresponding feature names list from features_names_list
    view_features = features_names_list[view_idx]
    for feature_name in view_features:
        count = feature_selection_counts[view_name].get(feature_name, 0)  # Get count, default 0
        stability_score = count / num_successful_runs  # Normalize by successful runs
        stability_results_list.append({
            'View': view_name,
            'Feature': feature_name,  # This is the unique feature name (e.g., "FeatureX_leaf_spectral")
            'SelectionFrequency': stability_score
        })

if not stability_results_list:
    print("ERROR: No stability results generated, possibly due to errors or no features meeting threshold.")
    exit()

stability_df = pd.DataFrame(stability_results_list)
stability_df = stability_df.sort_values(by=['View', 'SelectionFrequency'], ascending=[True, False])

# Add confidence tiers based on frequency (adjust thresholds if needed)
stability_df["ConfidenceTier"] = "very_low"  # Default
stability_df.loc[stability_df["SelectionFrequency"] >= 0.8, "ConfidenceTier"] = "high"
stability_df.loc[(stability_df["SelectionFrequency"] >= 0.5) & 
                 (stability_df["SelectionFrequency"] < 0.8), "ConfidenceTier"] = "medium"
stability_df.loc[(stability_df["SelectionFrequency"] >= 0.2) & 
                 (stability_df["SelectionFrequency"] < 0.5), "ConfidenceTier"] = "low"


outfile = os.path.join(config["output_dir"], f"mofa_bootstrap_stability_n{config['n_bootstrap']}.csv")
stability_df.to_csv(outfile, index=False)
print(f"\nBootstrap stability results saved to: {outfile}")
print(f"(Based on {num_successful_runs} successful runs where weights could be extracted)")
print(f"Stability defined as frequency feature had abs(weight) > {config['stability_weight_threshold']} on any active factor.")

total_end_time = time.time()
print(f"\nTotal Bootstrap Script Runtime: {(total_end_time - start_time)/60:.2f} minutes")
print("\n="*60)
print("Bootstrap Script Finished")
print("="*60)