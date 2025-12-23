# -*- coding: utf-8 -*-
"""
process_attention_data.py (HyperSeq Validation Version)

This script loads the raw attention data from the HyperSeq validation run.
It is configured to:
1. Load the raw 4D feature-level attention tensors from the HDF5 file.
2. Load the corresponding sample metadata.
3. Calculate and save overall and conditional attention statistics.
"""

# ===== IMPORTS =====
import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import h5py
try:
    import pyarrow
except ImportError:
    pyarrow = None
    print("WARNING: `pyarrow` library not found. Will use .csv fallback for metadata.")

# ===== CONFIGURATION FOR HYPERSEQ VALIDATION =====
print("="*60)
print("Processing Transformer Attention for HyperSeq Validation")
print("="*60)

# --- Paths ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val/transformer_results"
RESULTS_INPUT_DIR = os.path.join(BASE_DIR, "results")
PROCESSED_OUTPUT_DIR = os.path.join(BASE_DIR, "processed_attention")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- Input Files ---
# We have only one pairing for this validation run
PAIRING_NAME = "HyperSeq"

HDF5_PATH = os.path.join(RESULTS_INPUT_DIR, f"raw_attention_data_{PAIRING_NAME}.h5")
METADATA_PATH = os.path.join(RESULTS_INPUT_DIR, f"raw_attention_metadata_{PAIRING_NAME}.feather")
METADATA_CSV_FALLBACK_PATH = os.path.join(RESULTS_INPUT_DIR, f"raw_attention_metadata_{PAIRING_NAME}.csv")

# --- Output Directory ---
os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)

# --- Processing Parameters ---
TOP_K_PAIRS = 200
# For HyperSeq, the only meaningful metadata to group by is 'Batch'
CONDITIONAL_GROUPING_COLS = ['Batch']
METADATA_INDEX_COL = 'Row_names'
PERCENTILE_VALUE = 95

# ===== LOGGING =====
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"process_attention_hyperseq_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', '%H:%M:%S')
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fh = logging.FileHandler(log_filepath); fh.setFormatter(formatter); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(formatter); logger.addHandler(sh)
    return logger

logger = setup_logging(LOG_DIR)
logger.info(f"Processing Pairing: {PAIRING_NAME}")
logger.info(f"Input HDF5: {HDF5_PATH}")
logger.info(f"Output Directory: {PROCESSED_OUTPUT_DIR}")
logger.info(f"Grouping by: {CONDITIONAL_GROUPING_COLS}")

# ===== HELPER FUNCTIONS (Unchanged from your original script) =====
def load_h5_tensor_data(filepath: str) -> dict or None:
    logger.info(f"Loading tensor/feature data from HDF5: {filepath}")
    if not os.path.exists(filepath): logger.error(f"HDF5 file not found: {filepath}"); return None
    data = {}
    try:
        with h5py.File(filepath, 'r') as f:
            logger.info("Keys found in HDF5: " + str(list(f.keys())))
            data['attn_s2m'] = f['attention_spec_to_metab'][()]
            data['attn_m2s'] = f['attention_metab_to_spec'][()] if 'attention_metab_to_spec' in f else None
            data['spec_features'] = [s.decode('utf-8') for s in f['spectral_feature_names'][()]]
            data['metab_features'] = [s.decode('utf-8') for s in f['metabolite_feature_names'][()]]
        logger.info("HDF5 data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Failed HDF5 load: {e}", exc_info=True); return None

def load_metadata_file(feather_path: str, csv_fallback_path: str, index_col: str) -> pd.DataFrame or None:
    metadata_df = None
    if pyarrow and os.path.exists(feather_path):
        logger.info(f"Attempting metadata load from Feather: {feather_path}")
        try:
            metadata_df = pd.read_feather(feather_path).set_index(index_col)
            return metadata_df
        except Exception as e:
            logger.warning(f"Feather load failed: {e}. Trying CSV fallback.")
    if os.path.exists(csv_fallback_path):
        logger.info(f"Attempting metadata load from CSV fallback: {csv_fallback_path}")
        try:
            return pd.read_csv(csv_fallback_path, index_col=index_col)
        except Exception as e:
            logger.error(f"CSV load failed: {e}"); return None
    logger.error("Metadata loading failed for all paths."); return None

# ===== MAIN PROCESSING FUNCTION (Unchanged from your original script) =====
def process_pairing_attention(pairing_name, tensor_data, metadata_df, output_path, config):
    logger.info(f"\n===== Processing Pairing: {pairing_name} =====")
    proc_start_time = time.time()
    
    attn_s2m_raw, attn_m2s_raw = tensor_data['attn_s2m'], tensor_data.get('attn_m2s')
    spec_features, metab_features = tensor_data['spec_features'], tensor_data['metab_features']
    n_samples, n_heads, n_spec, n_metab = attn_s2m_raw.shape
    top_k, pct_val = config['TOP_K_PAIRS'], config.get('PERCENTILE_VALUE', 95)
    grouping_cols = [col for col in config['CONDITIONAL_GROUPING_COLS'] if col in metadata_df.columns]
    
    if attn_s2m_raw.shape[0] != len(metadata_df):
        logger.error("SAMPLE COUNT MISMATCH. Aborting."); return

    # --- 1. View-Level Stats ---
    logger.info("Calculating per-sample view-level statistics...")
    view_stats = pd.DataFrame(index=metadata_df.index)
    view_stats['AvgAttn_S2M'] = np.mean(attn_s2m_raw, axis=(1, 2, 3))
    view_stats['StdAttn_S2M'] = np.std(attn_s2m_raw, axis=(1, 2, 3))
    view_stats[f'P{pct_val}Attn_S2M'] = np.percentile(attn_s2m_raw, q=pct_val, axis=(1, 2, 3))
    if attn_m2s_raw is not None:
        view_stats['AvgAttn_M2S'] = np.mean(attn_m2s_raw, axis=(1, 2, 3))
        view_stats['StdAttn_M2S'] = np.std(attn_m2s_raw, axis=(1, 2, 3))
        view_stats[f'P{pct_val}Attn_M2S'] = np.percentile(attn_m2s_raw, q=pct_val, axis=(1, 2, 3))
    view_level_df = pd.concat([metadata_df, view_stats], axis=1)
    view_level_df.to_csv(os.path.join(output_path, f"processed_view_level_attention_{pairing_name}.csv"))

    # --- 2. Feature-Pair Averages ---
    logger.info("Calculating overall mean feature-pair attention...")
    attn_s2m_avg_heads = np.mean(attn_s2m_raw, axis=1)
    mean_attn_overall_s2m = np.mean(attn_s2m_avg_heads, axis=0)
    pairs_overall_s2m_df = pd.DataFrame(mean_attn_overall_s2m, index=spec_features, columns=metab_features).stack().reset_index()
    pairs_overall_s2m_df.columns = ['Spectral_Feature', 'Metabolite_Feature', 'Mean_Attention_S2M']
    pairs_overall_s2m_df.sort_values('Mean_Attention_S2M', ascending=False, inplace=True)
    pairs_overall_s2m_df.to_csv(os.path.join(output_path, f"processed_mean_attention_overall_{pairing_name}.csv"), index=False, float_format='%.6e')
    pairs_overall_s2m_df.head(top_k).to_csv(os.path.join(output_path, f"processed_top_{top_k}_pairs_overall_{pairing_name}.csv"), index=False, float_format='%.6e')
    
    # --- 3. Conditional Mean Attention ---
    if grouping_cols:
        logger.info(f"Calculating conditional mean attention grouped by: {grouping_cols}")
        all_conditional_means = []
        for group_keys, group_sample_ids in metadata_df.groupby(grouping_cols).groups.items():
            group_label_dict = dict(zip(grouping_cols, [group_keys] if not isinstance(group_keys, tuple) else group_keys))
            group_attn_s2m = attn_s2m_avg_heads[metadata_df.index.get_indexer(group_sample_ids)]
            mean_attn_group_s2m = np.mean(group_attn_s2m, axis=0)
            group_df = pd.DataFrame(mean_attn_group_s2m, index=spec_features, columns=metab_features).stack().reset_index()
            group_df.columns = ['Spectral_Feature', 'Metabolite_Feature', 'Mean_Attention_S2M_Group']
            for col, value in group_label_dict.items(): group_df[col] = value
            all_conditional_means.append(group_df)
        if all_conditional_means:
            conditional_df = pd.concat(all_conditional_means, ignore_index=True)
            conditional_df.to_csv(os.path.join(output_path, f"processed_mean_attention_conditional_{pairing_name}.csv"), index=False, float_format='%.6e')

    logger.info(f"===== Finished processing {pairing_name}. Duration: {(time.time() - proc_start_time):.2f} seconds =====")

# ===== MAIN EXECUTION =====
def main():
    main_start_time = time.time()
    logger.info("--- Starting Main Execution ---")
    
    global_config = {
        'TOP_K_PAIRS': TOP_K_PAIRS,
        'CONDITIONAL_GROUPING_COLS': CONDITIONAL_GROUPING_COLS,
        'METADATA_INDEX_COL': METADATA_INDEX_COL,
        'PERCENTILE_VALUE': PERCENTILE_VALUE
    }

    tensor_data = load_h5_tensor_data(HDF5_PATH)
    metadata = load_metadata_file(METADATA_PATH, METADATA_CSV_FALLBACK_PATH, METADATA_INDEX_COL)

    if tensor_data and metadata is not None:
        process_pairing_attention(
            pairing_name=PAIRING_NAME,
            tensor_data=tensor_data,
            metadata_df=metadata,
            output_path=PROCESSED_OUTPUT_DIR,
            config=global_config
        )
    else:
        logger.error("Could not load necessary data. Aborting.")

    total_duration_min = (time.time() - main_start_time) / 60
    logger.info(f"--- Main Execution Finished --- Total Duration: {total_duration_min:.2f} minutes ---")

if __name__ == '__main__':
    main()