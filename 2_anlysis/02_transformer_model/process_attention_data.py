# -*- coding: utf-8 -*-
"""
Process Raw Attention Data (v3.0)

This script processes feature-level attention data from transformer models
analyzing multi-omics datasets. It handles both spectral-to-metabolite (S->M)
and metabolite-to-spectral (M->S) attention tensors.

Main functions:
1. Loads raw 4D attention tensors from HDF5 files and metadata from Feather/CSV
2. Validates data alignment
3. Calculates view-level statistics (Mean, StdDev, P95)
4. Processes feature-pair attention scores
5. Generates conditional attention metrics grouped by metadata fields
6. Exports all data to CSV files for further analysis

Input: HDF5 files containing attention tensors and feature names
Output: CSV files with processed attention statistics and feature pairs
"""

# ===== IMPORTS =====
import os
import sys
import time
import logging
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import h5py
try:
    # Try importing pyarrow for Feather file support
    import pyarrow
    # Check if necessary modules are available
    if not hasattr(pd, 'read_feather'):
        pyarrow = None # Disable if pandas integration is missing
        print("WARNING: Pandas Feather support not fully available. Will attempt .csv fallback.")
except ImportError:
    pyarrow = None
    print("WARNING: `pyarrow` library not found. Will not be able to read .feather files. Will attempt .csv fallback.")

# Import SciPy conditionally for percentile calculation
try:
    from scipy.stats import percentileofscore # For potential future use
except ImportError:
    print("WARNING: `scipy` not found. Percentile calculations might rely solely on NumPy.")
    # NumPy's percentile function is sufficient here, so no critical failure.

# ===== CONFIGURATION =====

# --- Script Info ---
SCRIPT_NAME = "process_attention_data_3"
VERSION = "3.0.0"

# --- Paths ---
BASE_OUTPUT_DIR = r"C:/Users/ms/Desktop/hyper/output/transformer/v3_feature_attention"

# Input HDF5 files
HDF5_PATHS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, r"leaf/results/raw_attention_data_Leaf.h5"),
    "Root": os.path.join(BASE_OUTPUT_DIR, r"root/results/raw_attention_data_Root.h5")
}

# Input Metadata Files
METADATA_PATHS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, r"leaf/results/raw_attention_metadata_Leaf.feather"),
    "Root": os.path.join(BASE_OUTPUT_DIR, r"root/results/raw_attention_metadata_Root.feather")
}
METADATA_CSV_FALLBACK_PATHS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, r"leaf/results/raw_attention_metadata_Leaf.csv"),
    "Root": os.path.join(BASE_OUTPUT_DIR, r"root/results/raw_attention_metadata_Root.csv")
}

# Output Directories for Processed Data
OUTPUT_DIRS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, r"processed_attention_leaf"),
    "Root": os.path.join(BASE_OUTPUT_DIR, r"processed_attention_root")
}
OUTPUT_STRUCTURE = "PerPairing"

# Log directory within the main script's output path
LOG_DIR = os.path.join(BASE_OUTPUT_DIR, f"{SCRIPT_NAME}_logs")

# --- Processing Parameters ---
TOP_K_PAIRS = 500
CONDITIONAL_GROUPING_COLS = ['Genotype', 'Treatment', 'Day']
METADATA_INDEX_COL = 'Row_names'
PERCENTILE_VALUE = 95  # Percentile to calculate for view-level stats

# ===== LOGGING =====
def setup_logging(log_dir: str, script_name: str, version: str) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        log_dir: Directory to store log files
        script_name: Name of the script for log filename
        version: Version string for log filename
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{script_name}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    logger.info(f"Logging setup complete. Log file: {log_filepath}")
    return logger

logger = setup_logging(LOG_DIR, SCRIPT_NAME, VERSION)

logger.info("="*60)
logger.info(f"Starting Script: {SCRIPT_NAME} v{VERSION}")
logger.info(f"Processing Attention Data")
logger.info(f"Base Output Dir: {BASE_OUTPUT_DIR}")
logger.info(f"Top K Pairs to Save separately: {TOP_K_PAIRS}")
logger.info(f"Conditional Grouping Columns: {CONDITIONAL_GROUPING_COLS}")
logger.info(f"Metadata Index Column: {METADATA_INDEX_COL}")
logger.info(f"Calculating {PERCENTILE_VALUE}th percentile for view-level stats.")
logger.info("="*60)


# ===== HELPER FUNCTIONS =====
def load_h5_tensor_data(filepath: str) -> dict or None:
    """
    Load raw attention tensors and feature names from HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        
    Returns:
        Dictionary with tensors and feature names or None if loading failed
    """
    logger.info(f"Loading tensor/feature data from HDF5: {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"HDF5 file not found: {filepath}")
        return None
        
    data = {}
    try:
        with h5py.File(filepath, 'r') as f:
            logger.info("Keys found in HDF5: " + str(list(f.keys())))
            
            if 'attention_spec_to_metab' in f:
                data['attn_s2m'] = f['attention_spec_to_metab'][()]
                logger.info(f"  Loaded 'attention_spec_to_metab' shape: {data['attn_s2m'].shape}")
            else:
                logger.error("  'attention_spec_to_metab' not found.")
                return None
                
            if 'attention_metab_to_spec' in f:
                data['attn_m2s'] = f['attention_metab_to_spec'][()]
                logger.info(f"  Loaded 'attention_metab_to_spec' shape: {data['attn_m2s'].shape}")
            else:
                logger.warning("  'attention_metab_to_spec' not found.")
                data['attn_m2s'] = None
                
            if 'spectral_feature_names' in f:
                data['spec_features'] = [s.decode('utf-8') for s in f['spectral_feature_names'][()]]
                logger.info(f"  Loaded {len(data['spec_features'])} spectral features.")
            else:
                logger.error("  'spectral_feature_names' not found.")
                return None
                
            if 'metabolite_feature_names' in f:
                data['metab_features'] = [s.decode('utf-8') for s in f['metabolite_feature_names'][()]]
                logger.info(f"  Loaded {len(data['metab_features'])} metabolite features.")
            else:
                logger.error("  'metabolite_feature_names' not found.")
                return None

        # Validation for 4D tensors
        tensor_shape = data['attn_s2m'].shape
        if len(tensor_shape) != 4:
            logger.error(f"Expected 4D S->M tensor, got {len(tensor_shape)}D.")
            return None
            
        n_samples, n_heads, n_spec, n_metab = tensor_shape
        if n_spec != len(data['spec_features']):
            logger.error(f"S->M Spectral dim mismatch: Tensor={n_spec}, Features={len(data['spec_features'])}")
            return None
            
        if n_metab != len(data['metab_features']):
            logger.error(f"S->M Metabolite dim mismatch: Tensor={n_metab}, Features={len(data['metab_features'])}")
            return None
            
        if data.get('attn_m2s') is not None:
            m2s_shape = data['attn_m2s'].shape
            if len(m2s_shape) != 4:
                logger.warning(f"Expected 4D M->S tensor, got {len(m2s_shape)}D. M->S calculations might fail.")
            elif m2s_shape != (n_samples, n_heads, n_metab, n_spec):
                logger.error(f"M->S shape {m2s_shape} inconsistent with S->M {tensor_shape}.")
                return None
                
        logger.info("HDF5 tensor/feature data loaded and validated successfully.")
        return data
        
    except Exception as e:
        logger.error(f"Failed HDF5 load/process {filepath}: {e}", exc_info=True)
        return None


def load_metadata_file(feather_path: str, csv_fallback_path: str, index_col: str) -> pd.DataFrame or None:
    """
    Load metadata from Feather file with CSV fallback.
    
    Args:
        feather_path: Path to Feather file
        csv_fallback_path: Path to CSV file as fallback
        index_col: Column to use as DataFrame index
        
    Returns:
        Metadata DataFrame or None if loading failed
    """
    metadata_df = None
    
    # Try Feather
    if pyarrow and os.path.exists(feather_path):
        logger.info(f"Attempting metadata load from Feather: {feather_path}")
        try:
            metadata_df = pd.read_feather(feather_path)
            # Feather might save index as a column, check and set
            if index_col in metadata_df.columns:
                metadata_df.set_index(index_col, inplace=True)
                logger.info(f"  Loaded Feather, shape: {metadata_df.shape}. Index '{index_col}'.")
            elif metadata_df.index.name == index_col:
                 logger.info(f"  Loaded Feather, shape: {metadata_df.shape}. Index '{index_col}' already set.")
            else:
                 logger.warning(f"  Feather loaded, but index '{index_col}' not found as column or index name. Current index: '{metadata_df.index.name}'.")
            return metadata_df
        except Exception as e_feather:
            logger.warning(f"  Feather load failed: {e_feather}. Trying CSV fallback.")
            metadata_df = None
            
    # Fallback to CSV
    if metadata_df is None and os.path.exists(csv_fallback_path):
        logger.info(f"Attempting metadata load from CSV fallback: {csv_fallback_path}")
        try:
            metadata_df = pd.read_csv(csv_fallback_path, index_col=index_col)
            if metadata_df.index.name == index_col:
                logger.info(f"  Loaded CSV, shape: {metadata_df.shape}. Index '{index_col}'.")
            else:
                logger.warning(f"  CSV loaded, but failed set index '{index_col}'. Current: '{metadata_df.index.name}'.")
            return metadata_df
        except Exception as e_csv:
            logger.error(f"  CSV load failed: {e_csv}")
            return None
            
    if metadata_df is None:
        logger.error(f"Metadata failed load. Checked Feather: {feather_path}, CSV: {csv_fallback_path}")
        return None
        
    return metadata_df

# ===== MAIN PROCESSING FUNCTION =====
def process_pairing_attention(pairing_name: str,
                             hdf5_path: str,
                             metadata_path: str,
                             metadata_csv_fallback: str,
                             output_path: str,
                             config: dict):
    """
    Process attention data for a specific tissue pairing.
    
    Args:
        pairing_name: Name of the tissue pairing (e.g., "Leaf", "Root")
        hdf5_path: Path to HDF5 file with attention tensors
        metadata_path: Path to Feather file with metadata
        metadata_csv_fallback: Path to CSV file with metadata (fallback)
        output_path: Directory to save processed outputs
        config: Dictionary with processing parameters
        
    Returns:
        None
    """
    logger.info(f"\n===== Processing Pairing: {pairing_name} =====")
    proc_start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Load Raw Tensor Data (Expect 4D)
    tensor_data = load_h5_tensor_data(hdf5_path)
    if tensor_data is None:
        logger.error(f"Failed tensor load for {pairing_name}. Skip.")
        return
        
    attn_s2m_raw = tensor_data['attn_s2m']
    attn_m2s_raw = tensor_data.get('attn_m2s', None)
    spec_features = tensor_data['spec_features']
    metab_features = tensor_data['metab_features']

    # Load Metadata
    metadata_df = load_metadata_file(metadata_path, metadata_csv_fallback, config['METADATA_INDEX_COL'])
    if metadata_df is None:
        logger.error(f"Failed metadata load for {pairing_name}. Skip.")
        return

    # Critical Validation
    n_samples_attn = attn_s2m_raw.shape[0]
    n_samples_meta = len(metadata_df)
    if n_samples_attn != n_samples_meta:
        logger.error(f"SAMPLE COUNT MISMATCH for {pairing_name}! Tensor N={n_samples_attn}, Metadata Rows={n_samples_meta}. Aborting.")
        return
    else:
        logger.info(f"Validation PASSED: Tensor samples ({n_samples_attn}) match metadata rows ({n_samples_meta}).")

    # Common parameters
    n_samples, n_heads, n_spec, n_metab = attn_s2m_raw.shape
    top_k = config['TOP_K_PAIRS']
    pct_val = config.get('PERCENTILE_VALUE', 95)  # Use default if not in config
    grouping_cols = [col for col in config['CONDITIONAL_GROUPING_COLS'] if col in metadata_df.columns]
    logger.info(f"Processing {n_samples} samples, {n_heads} heads, {n_spec} spectral features, {n_metab} metabolite features.")
    logger.info(f"Calculating {pct_val}th percentile.")

    # Initialize variables for M->S calculations
    attn_m2s_avg_heads = None
    pairs_overall_m2s_df = None
    conditional_mean_pairs_s2m_df = None
    conditional_mean_pairs_m2s_df = None
    m2s_data_valid = False  # Flag to track if M->S data is usable

    # Pre-check M->S data validity
    if attn_m2s_raw is not None and len(attn_m2s_raw.shape) == 4:
        if attn_m2s_raw.shape == (n_samples, n_heads, n_metab, n_spec):
            m2s_data_valid = True
            logger.info("M->S tensor found and shape is valid.")
        else:
            logger.warning(f"M->S tensor shape {attn_m2s_raw.shape} incorrect. Expected {(n_samples, n_heads, n_metab, n_spec)}. Skipping M->S calculations.")
    else:
        logger.warning("M->S attention tensor not found or not 4D. Skipping M->S calculations.")


    try:
        # 1. Calculate and Save Per-Sample View-Level Stats (Mean, Std, Pct)
        logger.info("Calculating per-sample view-level statistics...")
        
        # S->M Stats
        avg_attn_s2m_per_sample = np.mean(attn_s2m_raw, axis=(1, 2, 3))
        std_attn_s2m_per_sample = np.std(attn_s2m_raw, axis=(1, 2, 3))
        pXX_attn_s2m_per_sample = np.percentile(attn_s2m_raw, q=pct_val, axis=(1, 2, 3))
        logger.info(f"  Calculated AvgAttn_S2M (shape: {avg_attn_s2m_per_sample.shape})")
        logger.info(f"  Calculated StdAttn_S2M (shape: {std_attn_s2m_per_sample.shape})")
        logger.info(f"  Calculated P{pct_val}Attn_S2M (shape: {pXX_attn_s2m_per_sample.shape})")

        # M->S Stats (Conditional)
        avg_attn_m2s_per_sample = None
        std_attn_m2s_per_sample = None
        pXX_attn_m2s_per_sample = None
        if m2s_data_valid:
            avg_attn_m2s_per_sample = np.mean(attn_m2s_raw, axis=(1, 2, 3))
            std_attn_m2s_per_sample = np.std(attn_m2s_raw, axis=(1, 2, 3))
            pXX_attn_m2s_per_sample = np.percentile(attn_m2s_raw, q=pct_val, axis=(1, 2, 3))
            logger.info(f"  Calculated AvgAttn_M2S (shape: {avg_attn_m2s_per_sample.shape})")
            logger.info(f"  Calculated StdAttn_M2S (shape: {std_attn_m2s_per_sample.shape})")
            logger.info(f"  Calculated P{pct_val}Attn_M2S (shape: {pXX_attn_m2s_per_sample.shape})")

        # Combine with metadata
        view_level_df = metadata_df.copy()
        view_level_df['AvgAttn_S2M'] = avg_attn_s2m_per_sample
        view_level_df['StdAttn_S2M'] = std_attn_s2m_per_sample
        view_level_df[f'P{pct_val}Attn_S2M'] = pXX_attn_s2m_per_sample
        view_level_df['AvgAttn_M2S'] = avg_attn_m2s_per_sample if m2s_data_valid else np.nan
        view_level_df['StdAttn_M2S'] = std_attn_m2s_per_sample if m2s_data_valid else np.nan
        view_level_df[f'P{pct_val}Attn_M2S'] = pXX_attn_m2s_per_sample if m2s_data_valid else np.nan

        outfile_view_level = os.path.join(output_path, f"processed_view_level_attention_{pairing_name}.csv")
        view_level_df.to_csv(outfile_view_level, index=True, index_label=config['METADATA_INDEX_COL'], float_format='%.6e')
        logger.info(f"Saved per-sample view-level statistics (Mean, Std, P{pct_val}) to: {outfile_view_level}")


        # 2. Calculate Feature-Pair Averages (Overall & Conditional)

        # S->M Calculations
        logger.info("Averaging S->M tensor over heads for S->M feature-pair analysis...")
        attn_s2m_avg_heads = np.mean(attn_s2m_raw, axis=1)  # Shape (N, N_spec, N_metab)
        logger.info(f"  S->M Shape after averaging heads: {attn_s2m_avg_heads.shape}")

        logger.info("Calculating overall mean S->M feature-pair attention...")
        mean_attn_overall_s2m = np.mean(attn_s2m_avg_heads, axis=0)  # (N_spec, N_metab)
        pairs_overall_s2m_list = []
        for i in range(n_spec):
            for j in range(n_metab):
                pairs_overall_s2m_list.append({
                    'Spectral_Feature': spec_features[i], 
                    'Metabolite_Feature': metab_features[j],
                    'Mean_Attention_S2M_AvgHeads': mean_attn_overall_s2m[i, j]
                })
        pairs_overall_s2m_df = pd.DataFrame(pairs_overall_s2m_list)
        pairs_overall_s2m_df.sort_values('Mean_Attention_S2M_AvgHeads', ascending=False, inplace=True)
        outfile_mean_overall_s2m = os.path.join(output_path, f"processed_mean_attention_overall_{pairing_name}.csv")
        pairs_overall_s2m_df.to_csv(outfile_mean_overall_s2m, index=False, float_format='%.6e')
        logger.info(f"Saved FULL overall S->M mean attention ({len(pairs_overall_s2m_df)} pairs) to: {outfile_mean_overall_s2m}")
        
        outfile_top_overall_s2m = os.path.join(output_path, f"processed_top_{top_k}_pairs_overall_{pairing_name}.csv")
        pairs_overall_s2m_df.head(top_k).to_csv(outfile_top_overall_s2m, index=False, float_format='%.6e')
        logger.info(f"Saved overall top {top_k} S->M pairs subset to: {outfile_top_overall_s2m}")
        
        top_k_pairs_tuples_s2m = list(zip(
            pairs_overall_s2m_df['Spectral_Feature'].head(top_k), 
            pairs_overall_s2m_df['Metabolite_Feature'].head(top_k)
        ))
        top_k_index_s2m = pd.MultiIndex.from_tuples(
            top_k_pairs_tuples_s2m, 
            names=['Spectral_Feature', 'Metabolite_Feature']
        )
        logger.info(f"Identified overall top {len(top_k_index_s2m)} S->M pairs for trend analysis.")

        # M->S Calculations
        if m2s_data_valid:
            logger.info("Averaging M->S tensor over heads for M->S feature-pair analysis...")
            attn_m2s_avg_heads = np.mean(attn_m2s_raw, axis=1)  # Shape (N, N_metab, N_spec)
            logger.info(f"  M->S Shape after averaging heads: {attn_m2s_avg_heads.shape}")

            logger.info("Calculating overall mean M->S feature-pair attention...")
            mean_attn_overall_m2s = np.mean(attn_m2s_avg_heads, axis=0)  # (N_metab, N_spec)
            pairs_overall_m2s_list = []
            for j in range(n_metab):  # Outer loop metab
                for i in range(n_spec):  # Inner loop spec
                    pairs_overall_m2s_list.append({
                        'Metabolite_Feature': metab_features[j], 
                        'Spectral_Feature': spec_features[i],
                        'Mean_Attention_M2S_AvgHeads': mean_attn_overall_m2s[j, i]
                    })
            pairs_overall_m2s_df = pd.DataFrame(pairs_overall_m2s_list)
            pairs_overall_m2s_df.sort_values('Mean_Attention_M2S_AvgHeads', ascending=False, inplace=True)

            # Save M->S Overall Files
            outfile_mean_overall_m2s = os.path.join(output_path, f"processed_mean_attention_overall_M2S_{pairing_name}.csv")
            pairs_overall_m2s_df.to_csv(outfile_mean_overall_m2s, index=False, float_format='%.6e')
            logger.info(f"Saved FULL overall M->S mean attention ({len(pairs_overall_m2s_df)} pairs) to: {outfile_mean_overall_m2s}")
            
            outfile_top_overall_m2s = os.path.join(output_path, f"processed_top_{top_k}_pairs_overall_M2S_{pairing_name}.csv")
            pairs_overall_m2s_df.head(top_k).to_csv(outfile_top_overall_m2s, index=False, float_format='%.6e')
            logger.info(f"Saved overall top {top_k} M->S pairs subset to: {outfile_top_overall_m2s}")
        else:
            logger.warning("Skipping M->S Overall Mean calculations (M->S data invalid).")


        # Conditional Mean Attention (Combined Loop for S->M and M->S)
        logger.info(f"Calculating conditional mean feature-pair attention grouped by: {grouping_cols}")
        if not grouping_cols:
            logger.warning("No valid grouping columns. Skipping conditional analysis.")
            conditional_mean_pairs_s2m_df = None
            conditional_mean_pairs_m2s_df = None
        else:
            metadata_for_grouping = metadata_df.copy()
            for col in list(grouping_cols):  # Iterate over a copy for safe removal
                if col in metadata_for_grouping.columns:
                    dtype = metadata_for_grouping[col].dtype
                    # Convert non-string/object/category columns to string for grouping
                    if not pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype) and not pd.api.types.is_categorical_dtype(dtype):
                        logger.debug(f"Converting grouping column '{col}' (dtype: {dtype}) to string.")
                        try:
                            metadata_for_grouping[col] = metadata_for_grouping[col].astype(str)
                        except Exception as e_conv:
                            logger.error(f"Failed to convert '{col}' to string: {e_conv}. Removing from grouping.")
                            grouping_cols.remove(col)
                else:
                    logger.warning(f"Grouping column '{col}' not found in metadata, removing from list.")
                    grouping_cols.remove(col)

            if not grouping_cols:
                logger.warning("No valid grouping columns remain. Skipping conditional analysis.")
                conditional_mean_pairs_s2m_df = None
                conditional_mean_pairs_m2s_df = None
            else:
                logger.info(f"Using final grouping columns: {grouping_cols}")
                if metadata_for_grouping.index.name != config['METADATA_INDEX_COL']:
                    logger.error(f"Metadata index name issue.")
                    return
                try:
                    sample_id_to_int_pos = {sid: i for i, sid in enumerate(metadata_for_grouping.index)}
                except Exception as e_map:
                    logger.error(f"Failed index mapping: {e_map}")
                    return

                grouped_metadata = metadata_for_grouping.groupby(grouping_cols)
                logger.info(f"Found {len(grouped_metadata)} unique condition groups.")

                all_conditional_mean_pairs_s2m = []
                all_conditional_mean_pairs_m2s = []
                group_counter = 0
                
                for group_keys, group_sample_ids in grouped_metadata.groups.items():
                    group_counter += 1
                    group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
                    group_label_dict = dict(zip(grouping_cols, group_keys))
                    group_label_str = "_".join([f"{k}={v}" for k, v in group_label_dict.items()])
                    logger.debug(f"Processing group {group_counter}/{len(grouped_metadata)}: {group_label_dict} ({len(group_sample_ids)} samples)")
                    
                    if len(group_sample_ids) == 0:
                        continue
                        
                    integer_indices = [sample_id_to_int_pos.get(sid) for sid in group_sample_ids if sample_id_to_int_pos.get(sid) is not None]
                    if not integer_indices:
                        continue
                        
                    if len(integer_indices) < len(group_sample_ids):
                        logger.warning(f" {len(group_sample_ids) - len(integer_indices)} SampleIDs not mapped.")

                    # S->M Calculation
                    group_attn_s2m = attn_s2m_avg_heads[integer_indices, :, :]
                    mean_attn_group_s2m = np.mean(group_attn_s2m, axis=0)
                    group_pairs_s2m_list = []
                    for i in range(n_spec):
                        for j in range(n_metab):
                            group_pairs_s2m_list.append({
                                'Spectral_Feature': spec_features[i],
                                'Metabolite_Feature': metab_features[j],
                                'Mean_Attention_S2M_Group_AvgHeads': mean_attn_group_s2m[i, j]
                            })
                    group_pairs_s2m_df = pd.DataFrame(group_pairs_s2m_list)
                    for col, value in group_label_dict.items():
                        group_pairs_s2m_df[col] = value
                    group_pairs_s2m_df['N_Samples_Group'] = len(integer_indices)
                    all_conditional_mean_pairs_s2m.append(group_pairs_s2m_df)

                    # M->S Calculation (Conditional)
                    if m2s_data_valid and attn_m2s_avg_heads is not None:  # Double check avg_heads was created
                        group_attn_m2s = attn_m2s_avg_heads[integer_indices, :, :]
                        mean_attn_group_m2s = np.mean(group_attn_m2s, axis=0)
                        group_pairs_m2s_list = []
                        for j in range(n_metab):
                            for i in range(n_spec):
                                group_pairs_m2s_list.append({
                                    'Metabolite_Feature': metab_features[j],
                                    'Spectral_Feature': spec_features[i],
                                    'Mean_Attention_M2S_Group_AvgHeads': mean_attn_group_m2s[j, i]
                                })
                        group_pairs_m2s_df = pd.DataFrame(group_pairs_m2s_list)
                        for col, value in group_label_dict.items():
                            group_pairs_m2s_df[col] = value
                        group_pairs_m2s_df['N_Samples_Group'] = len(integer_indices)
                        all_conditional_mean_pairs_m2s.append(group_pairs_m2s_df)

                # Concatenate S->M results
                conditional_mean_pairs_s2m_df = None
                if all_conditional_mean_pairs_s2m:
                    conditional_mean_pairs_s2m_df = pd.concat(all_conditional_mean_pairs_s2m, ignore_index=True)
                    id_cols_s2m = grouping_cols + ['N_Samples_Group']
                    value_cols_s2m = ['Spectral_Feature', 'Metabolite_Feature', 'Mean_Attention_S2M_Group_AvgHeads']
                    conditional_mean_pairs_s2m_df = conditional_mean_pairs_s2m_df[id_cols_s2m + value_cols_s2m]
                    outfile_mean_conditional_s2m = os.path.join(output_path, f"processed_mean_attention_conditional_{pairing_name}.csv")
                    conditional_mean_pairs_s2m_df.to_csv(outfile_mean_conditional_s2m, index=False, float_format='%.6e')
                    logger.info(f"Saved FULL conditional S->M mean attention ({len(conditional_mean_pairs_s2m_df)} rows) to: {outfile_mean_conditional_s2m}")
                else:
                    logger.warning("No S->M conditional mean pairs generated.")

                # Concatenate M->S results
                conditional_mean_pairs_m2s_df = None
                if all_conditional_mean_pairs_m2s:
                    conditional_mean_pairs_m2s_df = pd.concat(all_conditional_mean_pairs_m2s, ignore_index=True)
                    id_cols_m2s = grouping_cols + ['N_Samples_Group']
                    value_cols_m2s = ['Metabolite_Feature', 'Spectral_Feature', 'Mean_Attention_M2S_Group_AvgHeads']
                    conditional_mean_pairs_m2s_df = conditional_mean_pairs_m2s_df[id_cols_m2s + value_cols_m2s]
                    outfile_mean_conditional_m2s = os.path.join(output_path, f"processed_mean_attention_conditional_M2S_{pairing_name}.csv")
                    conditional_mean_pairs_m2s_df.to_csv(outfile_mean_conditional_m2s, index=False, float_format='%.6e')
                    logger.info(f"Saved FULL conditional M->S mean attention ({len(conditional_mean_pairs_m2s_df)} rows) to: {outfile_mean_conditional_m2s}")
                else:
                    logger.warning("No M->S conditional mean pairs generated.")


        # Attention Trends for Overall Top K S->M Pairs
        logger.info(f"Extracting S->M attention trends for overall top {top_k} S->M pairs...")
        if conditional_mean_pairs_s2m_df is not None and not conditional_mean_pairs_s2m_df.empty and not top_k_index_s2m.empty:
            try:
                required_trend_cols = ['Spectral_Feature', 'Metabolite_Feature'] + grouping_cols
                missing_trend_cols = [c for c in required_trend_cols if c not in conditional_mean_pairs_s2m_df.columns]
                if missing_trend_cols:
                    logger.error(f"Cannot calc S->M trends. Missing columns: {missing_trend_cols}")
                else:
                    conditional_mean_pairs_s2m_df_indexed = conditional_mean_pairs_s2m_df.set_index(
                        ['Spectral_Feature', 'Metabolite_Feature']
                    )
                    trends_df = conditional_mean_pairs_s2m_df_indexed[
                        conditional_mean_pairs_s2m_df_indexed.index.isin(top_k_index_s2m)
                    ].reset_index()
                    
                    if not trends_df.empty:
                        trends_df.sort_values(
                            by=grouping_cols + ['Spectral_Feature', 'Metabolite_Feature'], 
                            inplace=True
                        )
                        outfile_trends = os.path.join(output_path, f"processed_attention_trends_top_{top_k}_{pairing_name}.csv")
                        trends_df.to_csv(outfile_trends, index=False, float_format='%.6e')
                        logger.info(f"Saved S->M attention trends for top {top_k} pairs ({len(trends_df)} rows) to: {outfile_trends}")
                    else:
                        logger.warning("No S->M rows matched top K pairs in conditional data. Trends file not saved.")
            except Exception as e_trends:
                logger.error(f"Error during S->M trends calculation: {e_trends}", exc_info=True)
        else:
            logger.warning("Skipping S->M trends calc (missing conditional means or top K S->M pairs).")

        # Placeholder for Sync Metrics
        logger.info("Placeholder for Synchronization Metrics calculation.")

    except Exception as e:
        logger.error(f"An error occurred during processing for {pairing_name}: {e}", exc_info=True)

    proc_end_time = time.time()
    logger.info(f"===== Finished processing {pairing_name}. Duration: {(proc_end_time - proc_start_time):.2f} seconds =====")


# ===== MAIN EXECUTION =====
def main():
    """
    Main execution function that processes each pairing.
    """
    main_start_time = time.time()
    logger.info("--- Starting Main Execution ---")
    
    # Configuration dictionary
    global_config = {
        'TOP_K_PAIRS': TOP_K_PAIRS,
        'CONDITIONAL_GROUPING_COLS': CONDITIONAL_GROUPING_COLS,
        'METADATA_INDEX_COL': METADATA_INDEX_COL,
        'PERCENTILE_VALUE': PERCENTILE_VALUE
    }
    pairings_to_process = list(HDF5_PATHS.keys())

    for pairing in pairings_to_process:
        hdf5_file = HDF5_PATHS.get(pairing)
        metadata_file = METADATA_PATHS.get(pairing)
        metadata_csv_fallback_file = METADATA_CSV_FALLBACK_PATHS.get(pairing)
        output_dir_pairing = OUTPUT_DIRS.get(pairing)

        if not all([hdf5_file, (metadata_file or metadata_csv_fallback_file), output_dir_pairing]):
            logger.warning(f"Missing essential file/dir path for pairing '{pairing}'. Skipping.")
            if not hdf5_file:
                logger.warning("  Reason: HDF5 path missing.")
            if not (metadata_file or metadata_csv_fallback_file):
                logger.warning("  Reason: Both metadata Feather and CSV paths missing.")
            if not output_dir_pairing:
                logger.warning("  Reason: Output directory path missing.")
            continue

        process_pairing_attention(
            pairing_name=pairing,
            hdf5_path=hdf5_file,
            metadata_path=metadata_file,
            metadata_csv_fallback=metadata_csv_fallback_file,
            output_path=output_dir_pairing,
            config=global_config
        )

    main_end_time = time.time()
    total_duration_min = (main_end_time - main_start_time) / 60
    logger.info(f"--- Main Execution Finished --- Total Duration: {total_duration_min:.2f} minutes ---")
    logger.info("="*60)


# --- Entry Point ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(1)