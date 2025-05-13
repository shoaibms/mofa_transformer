#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOFA+ Permutation Test Script for Factor-Metadata Association

This script performs permutation testing to assess statistical significance of associations 
between MOFA+ latent factors and sample metadata. It uses the results from a previous 
MOFA+ analysis and calculates empirical p-values through random permutations of the metadata.

The script requires that a main MOFA+ analysis has already been completed, as it uses
factor values and metadata correlations from that analysis.
"""
import pandas as pd
import numpy as np
import os
import json
import h5py
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point
import traceback
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import time


print("=" * 60)
print("MOFA+ Permutation Test Script (Factor-Metadata Association)")
print("=" * 60)
start_time = time.time()

# --- 1. Configuration ---
print("1. Configuring paths and parameters...")
config = {
    # --- Paths (Use FULL dataset paths here) ---
    "output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa/mofa_permutation",
    "main_mofa_results_dir": r"C:/Users/ms/Desktop/hyper/output/mofa",
    "data_paths": {
        "leaf_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv",
        "root_spectral": r"C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv",
        "leaf_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv",
        "root_metabolite": r"C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv",
    },
    "metadata_file": r"C:/Users/ms/Desktop/hyper/output/mofa/all_metadata.json",
    "view_names": ["leaf_spectral", "root_spectral", "leaf_metabolite", "root_metabolite"],
    # --- Permutation Parameters ---
    "n_permutations": 1000,  # Number of permutations
    "fdr_alpha": 0.05
}
os.makedirs(config["output_dir"], exist_ok=True)
print(f"Output directory: {config['output_dir']}")
print(f"Reading main MOFA results from: {config['main_mofa_results_dir']}")
print(f"Number of permutations: {config['n_permutations']}")

# --- 2. Load Results from Main MOFA+ Run ---
print("\n2. Loading results from the main MOFA+ run...")

try:
    # Load Factors
    factors_file = os.path.join(config['main_mofa_results_dir'], 
                               "mofa_latent_factors_active.csv")
    factors_df = pd.read_csv(factors_file, index_col="Sample_Index_Label")
    print(f"   Loaded factors: {factors_df.shape}")

    # Load Real Correlations
    corr_file = os.path.join(config['main_mofa_results_dir'], 
                            "mofa_factor_metadata_associations_spearman.csv")
    real_corr_df = pd.read_csv(corr_file)
    print(f"   Loaded real correlations: {real_corr_df.shape}")

    # Load Combined Metadata
    metadata_file_aligned = os.path.join(config['main_mofa_results_dir'], 
                                        "aligned_combined_metadata.csv")
    combined_metadata_df = pd.read_csv(metadata_file_aligned, 
                                      index_col="Sample_Index_Label")
    print(f"   Loaded aligned metadata: {combined_metadata_df.shape}")

except FileNotFoundError as e:
    print(f"ERROR: Required file not found from main MOFA run: {e}")
    print("Please ensure the main MOFA+ script was run successfully first.")
    exit()
except Exception as e:
    print(f"ERROR loading files from main MOFA run: {e}")
    traceback.print_exc()
    exit()

# --- 3. Perform Permutation Testing ---
def perform_permutation_testing(factors_df, combined_metadata_df, real_corr_df, 
                               output_dir, num_permutations=1000, fdr_alpha=0.05):
    """
    Performs permutation testing for factor-metadata associations based on Spearman correlation.
    
    Args:
        factors_df: DataFrame containing MOFA+ latent factors
        combined_metadata_df: DataFrame containing sample metadata
        real_corr_df: DataFrame containing real correlations from main MOFA+ run
        output_dir: Directory to save results
        num_permutations: Number of permutations to perform
        fdr_alpha: Alpha value for FDR correction
        
    Returns:
        DataFrame containing permutation test results
    """
    print("\n3. Performing Permutation Testing for Factor-Metadata Correlations...")

    # --- Initial Data Checks ---
    if factors_df is None or factors_df.empty:
        print("     Skipping: factors_df is missing or empty.")
        return None
    if combined_metadata_df is None or combined_metadata_df.empty:
        print("     Skipping: combined_metadata_df is missing or empty.")
        return None
    if real_corr_df is None or real_corr_df.empty:
        print("     Skipping: real_corr_df is missing or empty.")
        return None

    permutation_results = []
    tested_metadata_vars = real_corr_df['Metadata'].unique()
    print(f"     Testing associations for metadata found in correlation results: "
          f"{list(tested_metadata_vars)}")

    try:
        aligned_metadata = combined_metadata_df.loc[factors_df.index]
    except Exception as e_align:
        print(f"     ERROR aligning metadata: {e_align}")
        return None

    total_pairs_to_test = len(factors_df.columns) * len(tested_metadata_vars)
    print(f"     Total factor-metadata pairs to potentially test: {total_pairs_to_test}")
    pair_count = 0
    skipped_count = 0

    # --- Main Loops ---
    for factor in factors_df.columns:
        if factor not in factors_df:
            continue
        factor_values = factors_df[factor].values

        for meta_var in tested_metadata_vars:
            pair_count += 1
            if meta_var not in aligned_metadata.columns:
                skipped_count += 1
                continue

            # Check real correlation
            real_corr_row = real_corr_df[(real_corr_df['Factor'] == factor) & 
                                        (real_corr_df['Metadata'] == meta_var)]
            if real_corr_row.empty:
                skipped_count += 1
                continue
            real_corr = real_corr_row['Correlation'].iloc[0]
            if np.isnan(real_corr):
                skipped_count += 1
                continue

            # Check for NaNs and sufficient data
            meta_values_orig = aligned_metadata[meta_var].values
            valid_idx_real = ~pd.isna(factor_values) & ~pd.isna(meta_values_orig)
            if not np.any(valid_idx_real):
                skipped_count += 1
                continue
            factor_values_clean = factor_values[valid_idx_real]
            meta_values_clean = meta_values_orig[valid_idx_real]
            if len(factor_values_clean) < 3:
                skipped_count += 1
                continue

            # Check metadata variability and convert to numeric
            meta_values_numeric = None
            try:
                numeric_try = pd.to_numeric(meta_values_clean, errors='coerce')
                if not np.isnan(numeric_try).all():
                    valid_numeric_idx = ~np.isnan(numeric_try)
                    if np.sum(valid_numeric_idx) < 3:
                        skipped_count += 1
                        continue
                    meta_values_numeric = numeric_try[valid_numeric_idx]
                    current_factor_values_clean = factor_values_clean[valid_numeric_idx]
                    if len(np.unique(meta_values_numeric)) < 2:
                        skipped_count += 1
                        continue
                else:
                    raise ValueError("Direct numeric failed")
            except ValueError:
                try:
                    if not all(isinstance(x, (int, float, str, bool, bytes, type(None))) 
                              for x in meta_values_clean):
                        skipped_count += 1
                        continue
                    codes = pd.Categorical(meta_values_clean).codes
                    if len(np.unique(codes)) < 2:
                        skipped_count += 1
                        continue
                    meta_values_numeric = codes
                    current_factor_values_clean = factor_values_clean
                except Exception:
                    skipped_count += 1
                    continue

            # --- Checks passed, proceed ---
            null_distribution = np.zeros(num_permutations) * np.nan
            n_valid_perms = 0
            permutation_errors = 0  # Keep error count in case needed later

            # --- Permutation Loop ---
            for i in range(num_permutations):
                permuted_meta_numeric = np.random.permutation(meta_values_numeric)
                try:
                    if (len(np.unique(current_factor_values_clean)) < 2 or 
                        len(np.unique(permuted_meta_numeric)) < 2):
                        corr_perm = np.nan
                    else:
                        if len(current_factor_values_clean) != len(permuted_meta_numeric):
                            continue
                        # Use underscore for p-value
                        corr_perm, _ = spearmanr(current_factor_values_clean, 
                                                permuted_meta_numeric)

                    if not np.isnan(corr_perm):
                        null_distribution[i] = corr_perm
                        n_valid_perms += 1
                except Exception:
                    permutation_errors += 1  # Count errors silently

            # --- P-value Calculation ---
            if n_valid_perms == 0:
                p_value = np.nan
            else:
                null_distribution_clean = null_distribution[~np.isnan(null_distribution)]
                p_value = ((np.sum(np.abs(null_distribution_clean) >= np.abs(real_corr)) + 1) 
                          / (n_valid_perms + 1))

            permutation_results.append({
                'Factor': factor, 
                'Tested_Metadata': meta_var, 
                'Real_Spearman_Corr': real_corr,
                'Permutation_P_Value': p_value, 
                'Num_Valid_Permutations': n_valid_perms
            })
            # Simple progress indicator
            if (len(permutation_results) % 20 == 0) and len(permutation_results) > 0:
                print(f"     ... completed permutations for {len(permutation_results)} "
                      f"factor-metadata pairs.")

    print(f"\n     Finished checking {pair_count} potential pairs. "
          f"Skipped {skipped_count} pairs before permutation loop.")
    if not permutation_results:
        print("     No permutation tests completed successfully.")
        return None

    # --- FDR Correction and Saving ---
    perm_df = pd.DataFrame(permutation_results)
    if not perm_df.empty and 'Permutation_P_Value' in perm_df.columns:
        p_vals_clean = perm_df['Permutation_P_Value'].dropna()
        if not p_vals_clean.empty:
            reject, pvals_corrected, _, _ = multipletests(
                p_vals_clean, alpha=fdr_alpha, method='fdr_bh')
            perm_df.loc[p_vals_clean.index, 'Permutation_P_Value_FDR'] = pvals_corrected
            perm_df.loc[p_vals_clean.index, 'Significant_FDR'] = reject
            perm_df['Significant_FDR'] = perm_df['Significant_FDR'].fillna(False)
        else:
            perm_df['Permutation_P_Value_FDR'] = np.nan
            perm_df['Significant_FDR'] = False
    else:
        perm_df['Permutation_P_Value_FDR'] = np.nan
        perm_df['Significant_FDR'] = False

    perm_df = perm_df.sort_values(
        by=['Significant_FDR', 'Permutation_P_Value_FDR'], 
        ascending=[False, True]
    )
    outfile = os.path.join(
        output_dir, 
        f"mofa_permutation_test_factor_metadata_corr_n{num_permutations}.csv"
    )
    perm_df.to_csv(outfile, index=False)
    print(f"     Permutation test results saved to: {outfile}")

    return perm_df


# Run the permutation test function
perm_results_df = perform_permutation_testing(
    factors_df,
    combined_metadata_df,
    real_corr_df,
    config["output_dir"],
    num_permutations=config["n_permutations"],
    fdr_alpha=config["fdr_alpha"]
)

# --- 4. Final Summary ---
total_end_time = time.time()
print(f"\nTotal Permutation Script Runtime: {(total_end_time - start_time)/60:.2f} minutes")
print("\n" + "=" * 60)
print("Permutation Script Finished")
print("=" * 60)