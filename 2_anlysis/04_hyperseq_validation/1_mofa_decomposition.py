# -*- coding: utf-8 -*-
"""
MOFA+ Decomposition and Validation

This script runs MOFA+ (Multi-Omics Factor Analysis) on the pre-processed HyperSeq dataset.
It performs the following steps:
1.  Data loading and validation (spectral and transcriptomics views).
2.  Metadata column identification.
3.  MOFA+ model training.
4.  Results extraction (factors, weights, variance explained).
5.  Visualization of variance explained and factor-metadata associations.
6.  Feature selection based on factor weights for downstream Transformer models.

"""

import pandas as pd
import numpy as np
import os
import h5py
import traceback
import time
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from mofapy2.run.entry_point import entry_point
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    print(f"Error: Required library missing. {e}")
    sys.exit(1)

# --- Configuration ---

# Visualization Settings
PLOT_SETTINGS = {
    "figsize_heatmap": (10, 7),
    "figsize_factor": (8, 6),
    "heatmap_cmap": "viridis",
    "scatter_alpha": 0.7,
    "box_color": "lightgray",
    "box_alpha": 0.5,
    "dpi": 300
}

# Analysis Configuration
CONFIG = {
    "base_dir": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val",
    "output_dir": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val/mofa_results",
    "figure_dir": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val/mofa_results/figures",
    
    "data_paths": {
        "spectral": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/transformer_input_hyperseq_spectral.csv",
        "transcriptomics": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/transformer_input_hyperseq_gene.csv",
    },
    
    "mapping_file": r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val/dummy_mapping.tsv",
    "view_names": ["spectral", "transcriptomics"],
    "reference_view": "spectral",
    
    "mapping_column_names": {
        "spectral": "Row_names",
        "transcriptomics": "Row_names",
    },

    "groups_names": ["group0"],
    "num_factors": 15,
    "maxiter": 1000,
    "convergence_mode": "medium",
    "drop_factor_threshold": 0.02,
    "outfile": "mofa_model_hyperseq.hdf5",
    "fdr_alpha": 0.05,
    
    "transformer_feature_cap": {
        "spectral": 15,
        "transcriptomics": 100
    },
}

def identify_metadata_columns(df):
    """
    Identify metadata columns based on data types and common naming patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        list: List of column names identified as metadata.
    """
    metadata_cols = ['Row_names']  # Always include Row_names
    
    for col in df.columns:
        if col == 'Row_names':
            continue
        
        # Check if column contains non-numeric data
        try:
            pd.to_numeric(df[col], errors='raise')
            # If numeric, likely a feature
        except (ValueError, TypeError):
            # Non-numeric, likely metadata
            metadata_cols.append(col)
            continue
        
        # Additional checks for metadata patterns
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['batch', 'group', 'condition', 'id', 'cell']):
            metadata_cols.append(col)
        elif df[col].nunique() <= 10 and df[col].dtype in ['int64', 'float64']:
            # Few unique values might indicate categorical metadata
            metadata_cols.append(col)
    
    return metadata_cols

def print_hdf5_structure(name, obj):
    """Helper to print HDF5 file structure."""
    print(f"   HDF5 item: {name}")

def main():
    print("="*80)
    print("MOFA+ Analysis on HyperSeq Data")
    print("="*80)
    start_time = time.time()
    
    # Ensure directories exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["figure_dir"], exist_ok=True)
    print(f"Output directory: {CONFIG['output_dir']}")
    
    # --- Section 1: Load and Prepare Data ---
    print("\n1. Loading and preparing data...")
    try:
        # Load datasets
        df_spectral = pd.read_csv(CONFIG["data_paths"]["spectral"])
        df_transcriptomics = pd.read_csv(CONFIG["data_paths"]["transcriptomics"])
        
        print(f"   - Loaded spectral data: {df_spectral.shape}")
        print(f"   - Loaded transcriptomics data: {df_transcriptomics.shape}")
        
        # Validate Row_names
        if 'Row_names' not in df_spectral.columns:
            raise KeyError("'Row_names' not found in spectral CSV.")
        if 'Row_names' not in df_transcriptomics.columns:
            raise KeyError("'Row_names' not found in transcriptomics CSV.")
        
        # Check alignment
        spectral_row_names = set(df_spectral['Row_names'].astype(str))
        transcriptomics_row_names = set(df_transcriptomics['Row_names'].astype(str))
        
        common_samples = spectral_row_names.intersection(transcriptomics_row_names)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between spectral and transcriptomics data.")
        
        print(f"   - Common samples: {len(common_samples)}")
        
        # Use common samples only
        master_reference_row_names = sorted(list(common_samples))
        
        # Create dummy mapping file
        dummy_map_df = pd.DataFrame({'Row_names': master_reference_row_names})
        dummy_map_df.to_csv(CONFIG["mapping_file"], sep='\t', index=False)
        
        # Identify metadata columns using spectral as reference
        ref_df = df_spectral
        metadata_columns = identify_metadata_columns(ref_df)
        print(f"   - Identified metadata columns: {metadata_columns}")
        
        # Extract metadata
        combined_metadata_df = ref_df[metadata_columns].copy()
        combined_metadata_df.set_index('Row_names', inplace=True)
        combined_metadata_df = combined_metadata_df.reindex(master_reference_row_names)
        
        # Process each view
        data_views_input_format = []
        feature_names_dict = {}
        
        for view_name in CONFIG["view_names"]:
            print(f"   - Processing view: {view_name}")
            
            if view_name == "spectral":
                df_view = df_spectral
            else:
                df_view = df_transcriptomics
            
            # Extract feature columns (exclude metadata)
            all_cols = df_view.columns.tolist()
            feature_cols = [col for col in all_cols if col not in metadata_columns]
            
            # Align samples and extract features
            df_view_aligned = df_view.set_index('Row_names').reindex(master_reference_row_names)
            feature_df_aligned = df_view_aligned[feature_cols]
            
            # Handle missing values
            if feature_df_aligned.isnull().values.any():
                missing_count = feature_df_aligned.isnull().sum().sum()
                print(f"     - Note: {missing_count} missing values found. Filling with column mean.")
                feature_df_aligned = feature_df_aligned.fillna(feature_df_aligned.mean())
            
            # Validate numeric data
            non_numeric_cols = []
            for col in feature_df_aligned.columns:
                try:
                    pd.to_numeric(feature_df_aligned[col], errors='raise')
                except (ValueError, TypeError):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                raise ValueError(f"Non-numeric feature columns found in {view_name}: {non_numeric_cols}")
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_df_aligned)
            
            # Prepare for MOFA+
            data_views_input_format.append([scaled_features])
            unique_feature_names = [f"{feat}_{view_name}" for feat in feature_df_aligned.columns]
            feature_names_dict[view_name] = unique_feature_names
            
            print(f"     - Processed {scaled_features.shape[1]} features, {scaled_features.shape[0]} samples")

    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Section 2: Run MOFA+ ---
    print("\n2. Initializing and running MOFA+ model...")
    mofa_start_time = time.time()
    model_outfile = os.path.join(CONFIG["output_dir"], CONFIG["outfile"])
    
    try:
        ent = entry_point()
        ent.set_data_options(scale_views=False)
        
        ent.set_data_matrix(
            data_views_input_format,
            views_names=CONFIG["view_names"],
            features_names=[feature_names_dict[v] for v in CONFIG["view_names"]],
            samples_names=[master_reference_row_names]
        )
        
        ent.set_model_options(
            factors=CONFIG["num_factors"],
            spikeslab_weights=False,
            ard_factors=True,
            ard_weights=True
        )
        
        ent.set_train_options(
            iter=CONFIG["maxiter"],
            convergence_mode=CONFIG["convergence_mode"],
            dropR2=CONFIG["drop_factor_threshold"],
            gpu_mode=False,
            seed=42,
            verbose=False
        )
        
        print("   - Building and training model...")
        ent.build()
        ent.run()
        
        ent.save(model_outfile, save_data=False)
        print(f"   - Model saved. Training Time: {time.time() - mofa_start_time:.2f} seconds")

    except Exception as e:
        print(f"Error during MOFA+ execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Section 3: Results Extraction ---
    print("\n3. Extracting results...")
    factors_df = None
    weights_dict = {}
    variance_explained_df = None
    
    try:
        with h5py.File(model_outfile, 'r') as hf:
            # hf.visititems(print_hdf5_structure) # Optional: uncomment to debug HDF5 structure
            
            # Extract factors
            try:
                factors = hf['expectations/Z/group0'][()].T
            except KeyError:
                factors = hf['expectations/Z'][()].T
            
            # Extract variance explained
            try:
                ve_data = hf['variance_explained/r2_per_factor/group0'][()]
            except KeyError:
                ve_data = hf['variance_explained/r2_per_factor'][()]
            
            # Identify active factors
            active_factors_mask = (ve_data.sum(axis=0) > CONFIG["drop_factor_threshold"])
            active_factors_indices = np.where(active_factors_mask)[0]
            
            print(f"   - Active factors: {len(active_factors_indices)}")
            
            if len(active_factors_indices) == 0:
                raise ValueError("No active factors found.")
            
            factor_column_names = [f"Factor{i+1}" for i in active_factors_indices]
            factors_df = pd.DataFrame(
                factors[:, active_factors_indices],
                index=master_reference_row_names,
                columns=factor_column_names
            )
            factors_df.index.name = "Row_names"
            
            # Extract weights
            for view in CONFIG["view_names"]:
                try:
                    weights_full = hf[f'expectations/W/{view}'][()].T
                    weights_dict[view] = weights_full[:, active_factors_indices]
                except KeyError as e:
                    print(f"   - Warning: Could not extract weights for view {view}: {e}")
                    # Try alternative path
                    weights_full = hf[f'expectations/W'][view][()].T
                    weights_dict[view] = weights_full[:, active_factors_indices]
            
            # Variance explained dataframe
            ve_active = ve_data[:, active_factors_indices]
            variance_explained_df = pd.DataFrame(
                ve_active.T,
                index=factor_column_names,
                columns=CONFIG["view_names"]
            )

    except Exception as e:
        print(f"Error extracting results: {e}")
        traceback.print_exc()

    # --- Section 4: Visualizations ---
    print("\n4. Generating visualizations...")
    try:
        if variance_explained_df is not None and not variance_explained_df.empty:
            plt.figure(figsize=PLOT_SETTINGS["figsize_heatmap"])
            sns.heatmap(variance_explained_df * 100, cmap=PLOT_SETTINGS["heatmap_cmap"], annot=True, fmt=".1f")
            plt.title('Variance Explained (%) per Factor and View')
            plt.ylabel('Factor')
            plt.xlabel('View')
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["figure_dir"], "variance_explained_heatmap.png"), dpi=PLOT_SETTINGS["dpi"])
            plt.close()
            
            for view in CONFIG["view_names"]:
                total_var = variance_explained_df[view].sum()
                print(f"   - Total variance explained ({view}): {total_var*100:.1f}%")

        # Plot Factors vs Metadata
        if factors_df is not None and not factors_df.empty and combined_metadata_df is not None:
            plot_factors_df = pd.merge(factors_df, combined_metadata_df, left_index=True, right_index=True)
            num_factors_to_plot = min(factors_df.shape[1], 4)
            
            metadata_to_plot = []
            for col in combined_metadata_df.columns:
                unique_vals = combined_metadata_df[col].nunique()
                if 2 <= unique_vals <= 20:
                    metadata_to_plot.append(col)
            
            if metadata_to_plot:
                print(f"   - Plotting factors against metadata: {metadata_to_plot}")
                for factor in plot_factors_df.columns[:num_factors_to_plot]:
                    if factor in factor_column_names:
                        for meta_col in metadata_to_plot:
                            plt.figure(figsize=PLOT_SETTINGS["figsize_factor"])
                            try:
                                sns.stripplot(
                                    data=plot_factors_df, 
                                    x=meta_col, 
                                    y=factor, 
                                    jitter=True, 
                                    alpha=PLOT_SETTINGS["scatter_alpha"]
                                )
                                sns.boxplot(
                                    data=plot_factors_df, 
                                    x=meta_col, 
                                    y=factor, 
                                    showfliers=False, 
                                    color=PLOT_SETTINGS["box_color"], 
                                    boxprops=dict(alpha=PLOT_SETTINGS["box_alpha"])
                                )
                                plt.title(f'{factor} vs {meta_col}')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                plt.savefig(
                                    os.path.join(CONFIG["figure_dir"], f"factor_plot_{factor}_vs_{meta_col}.png"), 
                                    dpi=PLOT_SETTINGS["dpi"]
                                )
                                plt.close()
                            except Exception as e:
                                print(f"     - Could not plot {factor} vs {meta_col}: {e}")
                                plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")
        traceback.print_exc()

    # --- Section 5: Feature Selection & File Prep ---
    print("\n5. Selecting features and creating Transformer input files...")
    try:
        if not weights_dict:
            raise ValueError("No weights available for feature selection.")
        
        selected_features_final = {}
        
        for view_name in CONFIG["view_names"]:
            if view_name not in weights_dict:
                continue
                
            weights_view = pd.DataFrame(weights_dict[view_name], index=feature_names_dict[view_name])
            
            # Calculate feature importance
            importance = weights_view.abs().sum(axis=1)
            n_select = CONFIG["transformer_feature_cap"][view_name]
            n_select = min(n_select, len(importance))
            
            top_features = importance.nlargest(n_select).index.tolist()
            selected_features_final[view_name] = top_features
            
            print(f"   - Selected {len(top_features)} features for view '{view_name}'")
        
        # Create transformer input files
        for view_name, selected_features in selected_features_final.items():
            original_feature_names = [f.replace(f"_{view_name}", "") for f in selected_features]
            
            if view_name == "spectral":
                df_orig = df_spectral
            else:
                df_orig = df_transcriptomics
            
            cols_to_keep = metadata_columns + original_feature_names
            cols_to_keep_existing = [c for c in cols_to_keep if c in df_orig.columns]
            
            if len(cols_to_keep_existing) != len(cols_to_keep):
                missing_cols = set(cols_to_keep) - set(cols_to_keep_existing)
                print(f"     - Warning: Missing columns for {view_name}: {missing_cols}")
            
            df_transformer = df_orig[cols_to_keep_existing].copy()
            df_transformer = df_transformer.set_index('Row_names').reindex(master_reference_row_names).reset_index()
            
            # Fill missing
            if df_transformer.isnull().values.any():
                numeric_cols = df_transformer.select_dtypes(include=[np.number]).columns
                df_transformer[numeric_cols] = df_transformer[numeric_cols].fillna(df_transformer[numeric_cols].mean())
            
            outfile = os.path.join(CONFIG["output_dir"], f"transformer_input_{view_name}_hyperseq.csv")
            df_transformer.to_csv(outfile, index=False)
            print(f"   - Saved: {outfile}")

    except Exception as e:
        print(f"Error during feature selection: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    if variance_explained_df is not None:
        print("Status: Success")
        print(f"Total Runtime: {(time.time() - start_time)/60:.2f} minutes")
    else:
        print("Status: Failed")

if __name__ == "__main__":
    main()
