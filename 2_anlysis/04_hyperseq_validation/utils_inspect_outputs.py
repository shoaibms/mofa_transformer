# -*- coding: utf-8 -*-
"""
read_and_summarize_results.py

A simple utility script to read the binary outputs (HDF5 and Feather) from
the validation pipeline and print their key contents in a human-readable format.
"""
import pandas as pd
import numpy as np
import h5py
import os

print("="*80)
print("Reading and Summarizing Validation Results")
print("="*80)

# --- Configuration ---
# Point this to the base output directory of your validation run
BASE_DIR = r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val"
MOFA_RESULTS_DIR = os.path.join(BASE_DIR, "mofa_results")
TRANSFORMER_RESULTS_DIR = os.path.join(BASE_DIR, "transformer_results", "results")

# --- Define File Paths ---
MOFA_MODEL_PATH = os.path.join(MOFA_RESULTS_DIR, "mofa_model_hyperseq.hdf5")
METADATA_PATH = os.path.join(TRANSFORMER_RESULTS_DIR, "raw_attention_metadata_HyperSeq.feather")

# --- Function to Read MOFA HDF5 ---
def summarize_mofa_hdf5(filepath):
    print("\n" + "-"*20 + " SUMMARY OF MOFA+ HDF5 FILE " + "-"*20)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return

    try:
        with h5py.File(filepath, 'r') as hf:
            # 1. Active Factors and Variance Explained
            print("\n--- [1] Variance Explained per Factor ---")
            variance_data = hf['variance_explained/r2_per_factor/group0'][()]
            active_factors_mask = (variance_data.sum(axis=0) > 0.02) # Using 2% threshold from script
            active_indices = np.where(active_factors_mask)[0]
            
            ve_active = variance_data[:, active_indices] * 100 # As percentage
            ve_df = pd.DataFrame(
                ve_active.T,
                columns=['Spectral_VE', 'Transcriptomics_VE'],
                index=[f"Factor_{i+1}" for i in active_indices]
            )
            print(ve_df.to_string())

            # 2. Top 10 Feature Weights for each Active Factor
            print("\n--- [2] Top 5 Feature Weights per Active Factor ---")
            for factor_col_idx, hdf5_factor_idx in enumerate(active_indices):
                print(f"\n--- FACTOR {hdf5_factor_idx + 1} ---")
                
                # Spectral Weights
                weights_spec = hf['expectations/W/spectral'][hdf5_factor_idx, :]
                features_spec = [s.decode() for s in hf['features/spectral'][()]]
                spec_series = pd.Series(weights_spec, index=features_spec).abs().nlargest(5)
                print("  Top 5 Spectral Features:")
                print(spec_series.to_string())

                # Transcriptomics Weights
                weights_tx = hf['expectations/W/transcriptomics'][hdf5_factor_idx, :]
                features_tx = [s.decode() for s in hf['features/transcriptomics'][()]]
                tx_series = pd.Series(weights_tx, index=features_tx).abs().nlargest(5)
                print("\n  Top 5 Transcriptomics Features:")
                print(tx_series.to_string())

    except Exception as e:
        print(f"ERROR reading HDF5 file: {e}")

# --- Function to Read Metadata Feather ---
def summarize_metadata_feather(filepath):
    print("\n" + "-"*20 + " SUMMARY OF METADATA FEATHER FILE " + "-"*20)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return
        
    try:
        df = pd.read_feather(filepath)
        print("First 5 rows of metadata:")
        print(df.head().to_string())
        print("\nValue counts for key columns:")
        if 'Batch' in df.columns:
            print("\n--- Batch ---")
            print(df['Batch'].value_counts().to_string())
        if 'Grid' in df.columns:
            print("\n--- Grid (Top 5) ---")
            print(df['Grid'].value_counts().head(5).to_string())
    except Exception as e:
        print(f"ERROR reading Feather file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    summarize_mofa_hdf5(MOFA_MODEL_PATH)
    summarize_metadata_feather(METADATA_PATH)
    print("\n" + "="*80)
    print("Summary complete. Please copy the entire output from your terminal.")
    print("="*80)