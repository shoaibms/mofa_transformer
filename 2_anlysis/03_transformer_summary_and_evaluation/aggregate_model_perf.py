import pandas as pd
import numpy as np
import os
import sys

# --- Configuration ---
TRANSFORMER_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer"
# --- !!! SET THE CORRECT METRIC NAME FOUND IN YOUR CSV FILES !!! ---
TARGET_METRIC_NAME = 'F1_Macro'  # <--- CHANGE THIS if needed after checking CSVs
# ---

# --- Helper Function for Debug Printing ---
def debug_print(*args, **kwargs):
    """Prints debug messages to stderr."""
    print("DEBUG:", *args, file=sys.stderr, **kwargs)

def extract_model_performance(transformer_dir, target_metric):
    """
    Extracts target performance metric scores for baseline models (RF, KNN)
    and the final Transformer model from specified output files.

    Args:
        transformer_dir (str): The base directory containing transformer output folders.
        target_metric (str): The exact (case-insensitive) name of the metric
                               to extract (e.g., 'f1_score', 'f1', 'accuracy').

    Returns:
        dict: A nested dictionary containing scores. None if errors occur.
    """
    performance_data = {'Leaf': {}, 'Root': {}}
    required_files = {
        "leaf_baseline": os.path.join(transformer_dir, "phase1.1", "leaf", "transformer_baseline_comparison_Leaf.csv"),
        "root_baseline": os.path.join(transformer_dir, "phase1.1", "root", "transformer_baseline_comparison_Root.csv"),
        "leaf_transformer": os.path.join(transformer_dir, "v3_feature_attention", "leaf", "results", "transformer_class_performance_Leaf.csv"),
        "root_transformer": os.path.join(transformer_dir, "v3_feature_attention", "root", "results", "transformer_class_performance_Root.csv"),
    }
    target_metric_lower = target_metric.lower() # For comparison

    debug_print(f"--- Starting Model Performance Extraction ---")
    debug_print(f"Base Transformer Directory: {transformer_dir}")
    debug_print(f"Target metric: '{target_metric}' (case-insensitive)")

    # --- Check for essential files ---
    all_files_found = True
    debug_print("Checking for required performance files...")
    for key, fpath in required_files.items():
        debug_print(f"  Checking {key}: {fpath}")
        if not os.path.exists(fpath):
            print(f"ERROR: Required file not found: {fpath}")
            all_files_found = False
        else:
            debug_print(f"    Found.")
    if not all_files_found:
        debug_print("One or more required performance files missing. Aborting.")
        return None
    debug_print("All required performance files found.")

    # --- Helper function to process each file ---
    def process_file(file_path, tissue, is_baseline):
        debug_print(f"\nProcessing {tissue} {'Baseline' if is_baseline else 'Transformer'}: {file_path}")
        try:
            df = pd.read_csv(file_path)
            debug_print(f"  Loaded Data. Shape: {df.shape}, Columns: {df.columns.tolist()}")

            # Check required columns
            required_cols = ['Task', 'Metric', 'Score']
            if is_baseline:
                required_cols.insert(0, 'Model')
            if not all(col in df.columns for col in required_cols):
                 print(f"ERROR: Missing required columns in {file_path}. Expected: {required_cols}, Found: {df.columns.tolist()}")
                 return False # Indicate failure

            # Pre-process and type conversion
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
            if 'Metric' in df.columns:
                 df['Metric'] = df['Metric'].astype(str).str.strip() # Ensure string and strip whitespace
            else:
                 debug_print(f"  WARNING: 'Metric' column not found in {file_path}!")
                 return True # Continue processing other files, but this one failed


            # Filter for the target metric
            df_filtered = df.loc[df['Metric'].str.lower() == target_metric_lower].copy()
            debug_print(f"  Filtered for Metric='{target_metric_lower}'. Found {df_filtered.shape[0]} rows.")

            if df_filtered.empty:
                # If no rows found for target metric, print available metrics
                available_metrics = df['Metric'].unique()
                debug_print(f"  WARNING: Target metric '{target_metric}' not found in {file_path}.")
                debug_print(f"  Available metrics in this file: {available_metrics}")
            else:
                debug_print(f"  Filtered Data Head:\n{df_filtered.head()}")

            # Populate results
            for _, row in df_filtered.iterrows():
                task = row['Task']
                score = row['Score']
                model = row['Model'] if is_baseline else 'Transformer'

                if pd.isna(score):
                    debug_print(f"    Skipping {tissue} -> {task} -> {model} due to missing/non-numeric score.")
                    continue

                if task not in performance_data[tissue]:
                    performance_data[tissue][task] = {}
                performance_data[tissue][task][model] = score
                debug_print(f"    Extracted {tissue} -> {task} -> {model}: {score:.4f}") # More precision for debug
            return True # Indicate success

        except Exception as e:
            print(f"ERROR: Failed processing file {file_path}: {e}")
            debug_print(f"Exception type: {type(e).__name__}, Args: {e.args}")
            return False # Indicate failure

    # --- Process all files ---
    success = True
    if not process_file(required_files['leaf_baseline'], 'Leaf', is_baseline=True): success = False
    if not process_file(required_files['root_baseline'], 'Root', is_baseline=True): success = False
    if not process_file(required_files['leaf_transformer'], 'Leaf', is_baseline=False): success = False
    if not process_file(required_files['root_transformer'], 'Root', is_baseline=False): success = False

    if not success:
         debug_print("Errors occurred during file processing.")
         # Optionally return None or partial results depending on desired behavior
         # return None

    debug_print(f"\n--- Finished Model Performance Extraction ---")
    debug_print(f"Final extracted data structure:\n{performance_data}")
    return performance_data


# --- (print_performance_summary function remains the same) ---
def print_performance_summary(data, metric_name="F1-Score"):
    """Prints the aggregated performance data in a formatted way."""
    if not data or (not data['Leaf'] and not data['Root']):
        print("No performance data extracted or processed successfully.")
        return

    print("-" * 60)
    print(f"Predictive Model Performance Summary ({metric_name})")
    print("-" * 60)

    tissues = ['Leaf', 'Root']
    tasks = ['Genotype', 'Treatment', 'Day']
    # Models to display - adjust if names in CSV differ (e.g., 'RandomForestClassifier')
    models_to_display = ['RandomForest', 'KNN', 'Transformer']

    # Dynamically find all models actually present in the data
    all_models_found = set()
    for tissue in tissues:
        if tissue in data:
            for task in data[tissue]:
                all_models_found.update(data[tissue][task].keys())

    # Create the display order, putting known ones first
    model_order = [m for m in models_to_display if m in all_models_found]
    model_order += sorted([m for m in all_models_found if m not in models_to_display])


    for tissue in tissues:
        print(f"\n--- {tissue} Tissue ---")
        if tissue not in data or not data[tissue]:
            print(f"  No data found for {tissue} tissue.")
            continue

        tissue_data = data[tissue]
        # Dynamically adjust header width
        col_width = 15
        header = f"{'Task':<12}" + "".join([f"{model:<{col_width}}" for model in model_order])
        print(header)
        print("-" * len(header))

        for task in tasks:
            line = f"{task:<12}"
            if task in tissue_data:
                task_data = tissue_data[task]
                for model in model_order:
                    score = task_data.get(model) # Safe access
                    if pd.isna(score) or score is None:
                         score_str = "N/A"
                    elif isinstance(score, (float, np.number)):
                        # Format as percentage with 2 decimal places
                        score_str = f"{score:.2%}"
                    else:
                         score_str = str(score) # Fallback for unexpected types
                    line += f"{score_str:<{col_width}}"
            else:
                 line += "".join(["N/A".ljust(col_width) for _ in model_order])
            print(line)

    print("-" * 60)


# --- Run Extraction and Print ---
if __name__ == "__main__":
    # Pass the target metric name to the extraction function
    performance_summary = extract_model_performance(TRANSFORMER_OUTPUT_DIR, TARGET_METRIC_NAME)

    if performance_summary:
        # Pass the metric name for printing
        print_performance_summary(performance_summary, metric_name=TARGET_METRIC_NAME.replace("_", " ").title())
    else:
        print(f"\nERROR: Failed to get performance summary. Check errors and debug output above.")
        print(f"Ensure the directory '{TRANSFORMER_OUTPUT_DIR}' and the specific CSV files exist and contain the metric '{TARGET_METRIC_NAME}'.")