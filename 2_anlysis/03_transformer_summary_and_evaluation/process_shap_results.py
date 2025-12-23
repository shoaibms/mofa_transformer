import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# --- Configuration ---
# !!! SET THIS PATH to the PARENT directory containing 'shap_analysis_ggl' !!!
TRANSFORMER_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer"
SHAP_DATA_SUBDIR = "shap_analysis_ggl/importance_data"
TOP_N_FEATURES = 20 # Number of top features to list per task

# Define expected feature types (adjust if different in your files)
# These will be normalized to lowercase for comparison
SPECTRAL_TYPE_KEYWORDS = ['spectral'] # Keywords to identify spectral features
METABOLITE_TYPE_KEYWORDS = ['metabolite', 'cluster'] # Keywords for metabolites

# --- Helper Function for Debug Printing ---
def debug_print(*args, **kwargs):
    """Prints debug messages to stderr."""
    print("DEBUG:", *args, file=sys.stderr, **kwargs)

def process_shap_results(transformer_dir, shap_subdir):
    """
    Processes SHAP importance files to extract top features and aggregate
    contributions by feature type. Stores top features DataFrame per task/tissue.

    Args:
        transformer_dir (str): The base directory containing transformer outputs.
        shap_subdir (str): The subdirectory path relative to transformer_dir
                             containing the SHAP importance CSV files.

    Returns:
        dict: A nested dictionary containing SHAP summary info:
              { 'Leaf': {
                  'Genotype': {
                      'top_features': [...],
                      'contribution_pct': {...},
                      'top_features_df': pd.DataFrame(...) # Added
                  },
                  ... ,
                  'all_top_features': {feature: {task1: rank, ...}}
                },
                'Root': { ... }
              }
              Returns None if major errors occur preventing summary generation.
    """
    shap_base_path = os.path.join(transformer_dir, shap_subdir)
    # Initialize results dict structure
    results = {}
    tissues = ['Leaf', 'Root']
    tasks = ['Genotype', 'Treatment', 'Day']
    for tissue in tissues:
        results[tissue] = {'all_top_features': defaultdict(dict)}
        for task in tasks:
            results[tissue][task] = {} # Ensure task dict exists

    debug_print(f"--- Starting SHAP Results Processing ---")
    debug_print(f"Base SHAP data directory: {shap_base_path}")
    if not os.path.isdir(shap_base_path):
        print(f"ERROR: SHAP data directory not found: {shap_base_path}")
        # Return None according to docstring if base dir not found
        return None

    all_files_processed_successfully = True # Tracks if *any* file fails
    at_least_one_file_processed = False # Tracks if at least one file was processed

    for tissue in tissues:
        for task in tasks:
            filename = f"shap_importance_{tissue}_{task}.csv"
            file_path = os.path.join(shap_base_path, filename)
            debug_print(f"\nProcessing: {tissue} - {task} ({filename})")

            if not os.path.exists(file_path):
                print(f"ERROR: SHAP file not found: {file_path}")
                results[tissue][task] = {'error': 'File not found'}
                all_files_processed_successfully = False
                continue # Skip to next file

            try:
                df_shap = pd.read_csv(file_path)
                debug_print(f"  Loaded SHAP data. Shape: {df_shap.shape}, Columns: {df_shap.columns.tolist()}")

                # --- Basic Validation ---
                required_cols = ['Feature', 'MeanAbsoluteShap', 'FeatureType']
                if not all(col in df_shap.columns for col in required_cols):
                    print(f"ERROR: Missing required columns in {filename}. Expected: {required_cols}, Found: {df_shap.columns.tolist()}")
                    results[tissue][task] = {'error': 'Missing columns'}
                    all_files_processed_successfully = False
                    continue

                # Convert SHAP values to numeric, handle errors
                df_shap['MeanAbsoluteShap'] = pd.to_numeric(df_shap['MeanAbsoluteShap'], errors='coerce')
                if df_shap['MeanAbsoluteShap'].isnull().any():
                    nan_count = df_shap['MeanAbsoluteShap'].isnull().sum()
                    debug_print(f"  WARNING: Found {nan_count} non-numeric 'MeanAbsoluteShap' values in {filename}. Dropping these rows for calculation.")
                    df_shap = df_shap.dropna(subset=['MeanAbsoluteShap'])

                if df_shap.empty:
                     print(f"ERROR: No valid SHAP data remaining in {filename} after cleaning.")
                     results[tissue][task] = {'error': 'No valid data'}
                     all_files_processed_successfully = False
                     continue

                at_least_one_file_processed = True # Mark that we processed at least one file

                # --- Normalize FeatureType ---
                def normalize_type(ftype):
                    ftype_lower = str(ftype).lower()
                    if any(keyword in ftype_lower for keyword in SPECTRAL_TYPE_KEYWORDS):
                        return 'Spectral'
                    elif any(keyword in ftype_lower for keyword in METABOLITE_TYPE_KEYWORDS):
                        return 'Metabolite'
                    else:
                        return 'Other' # Or Unknown

                df_shap['FeatureTypeNorm'] = df_shap['FeatureType'].apply(normalize_type)
                type_counts = df_shap['FeatureTypeNorm'].value_counts()
                debug_print(f"  Normalized FeatureType counts:\n{type_counts}")
                if 'Other' in type_counts and type_counts['Other'] > 0:
                     debug_print(f"    Original 'Other' types found: {df_shap.loc[df_shap['FeatureTypeNorm'] == 'Other', 'FeatureType'].unique()}")


                # --- 1. Extract Top N Features ---
                df_shap_sorted = df_shap.sort_values('MeanAbsoluteShap', ascending=False).copy() # Use copy to avoid SettingWithCopyWarning
                df_shap_sorted['Rank'] = range(1, len(df_shap_sorted) + 1)
                top_features_list = df_shap_sorted.head(TOP_N_FEATURES)['Feature'].tolist()
                debug_print(f"  Top {TOP_N_FEATURES} features: {top_features_list}")

                # --- Get Top N Features DataFrame for this task/tissue ---
                df_top_n = df_shap_sorted.head(TOP_N_FEATURES)[['Feature', 'MeanAbsoluteShap', 'FeatureTypeNorm', 'Rank']].copy()
                # We don't need Tissue/Task columns in this intermediate df anymore
                # df_top_n['Tissue'] = tissue
                # df_top_n['Task'] = task
                # all_top_features_dfs.append(df_top_n)

                # --- Store ranks for generalist/specialist analysis (using existing df_shap_sorted)
                for _, row in df_shap_sorted.head(20).iterrows(): # Store top 20 ranks per task
                     results[tissue]['all_top_features'][row['Feature']][task] = row['Rank']

                # --- 2. Calculate Aggregate Contribution ---
                contribution = df_shap.groupby('FeatureTypeNorm')['MeanAbsoluteShap'].sum()
                total_shap = contribution.sum()
                contribution_pct = (contribution / total_shap * 100).round(2)
                debug_print(f"  Aggregate SHAP contribution (%):\n{contribution_pct}")

                # Store results for this task/tissue
                results[tissue][task] = {
                    'top_features': top_features_list, # Use the list generated earlier
                    'contribution_pct': contribution_pct.to_dict(), # Convert Series to dict
                    'top_features_df': df_top_n # Store the top N df for later processing
                }

            except Exception as e:
                print(f"ERROR: Failed processing file {filename}: {e}")
                debug_print(f"Exception type: {type(e).__name__}, Args: {e.args}")
                results[tissue][task] = {'error': str(e)}
                all_files_processed_successfully = False

    debug_print(f"\n--- Finished SHAP Results Processing ---")
    if not all_files_processed_successfully:
        debug_print("WARNING: Errors occurred during processing some files.")

    # --- 3. Prepare data for Generalist/Specialist Identification ---
    # (The ranks are already stored in results[tissue]['all_top_features'])
    # We will print this structure to help the user identify them.
    debug_print("\n--- Data for Generalist/Specialist Identification ---")
    debug_print("(Feature: {Task1: Rank, Task2: Rank, ...} for features in Top 20 of at least one task)")
    for tissue in tissues:
         debug_print(f"\n{tissue} Tissue Top Feature Ranks:")
         # Sort features by mean rank across tasks for easier viewing (optional)
         features_with_ranks = results[tissue]['all_top_features']
         # Calculate mean rank, handling missing tasks (assign large rank like 999)
         ranked_features = []
         for feature, task_ranks in features_with_ranks.items():
              all_ranks = [task_ranks.get(t, 999) for t in tasks] # Get rank or 999 if missing
              mean_rank = sum(all_ranks) / len(all_ranks)
              ranked_features.append({'feature': feature, 'mean_rank': mean_rank, 'ranks': task_ranks})

         # Sort by mean rank
         ranked_features.sort(key=lambda x: x['mean_rank'])

         for item in ranked_features[:15]: # Print top ~15 overall ranked features
              rank_str = ", ".join([f"{t}:{r}" for t,r in item['ranks'].items()])
              debug_print(f"  - {item['feature']} (AvgRank:{item['mean_rank']:.1f}) Ranks: {{{rank_str}}}")


    # Return summary dict containing the individual top feature DFs
    summary_to_return = results if at_least_one_file_processed else None
    return summary_to_return


def print_shap_summary(data):
    """Prints the processed SHAP summary data."""
    if not data:
        print("No SHAP data processed.")
        return

    print("-" * 70)
    print("SHAP Analysis Summary")
    print("-" * 70)

    tissues = ['Leaf', 'Root']
    tasks = ['Genotype', 'Treatment', 'Day']

    for tissue in tissues:
        print(f"\n--- {tissue} Tissue ---")
        if tissue not in data:
            print("  No data.")
            continue

        tissue_data = data[tissue]
        for task in tasks:
            print(f"  Task: {task}")
            task_data = tissue_data.get(task)
            if not task_data or 'error' in task_data:
                print(f"    Error processing: {task_data.get('error', 'Unknown error')}")
                continue

            # Print Top Features
            top_f = task_data.get('top_features', [])
            print(f"    Top {len(top_f)} Features: {', '.join(top_f) if top_f else 'None found'}")

            # Print Contributions
            contrib = task_data.get('contribution_pct', {})
            contrib_str = ", ".join([f"{ftype}: {pct:.1f}%" for ftype, pct in contrib.items()])
            print(f"    Contribution (%): {contrib_str if contrib else 'N/A'}")

    print("-" * 70)
    print("Note: Check DEBUG output (stderr) for detailed logs and data")
    print("      to help identify generalist/specialist features.")
    print("-" * 70)


# --- Run Extraction and Print ---
if __name__ == "__main__":
    # Define CSV output path
    CSV_OUTPUT_DIR = r"C:\\Users\\ms\\Desktop\\hyper\\output\\transformer\\novility_plot"
    CSV_FILENAME = "Table_S3_SHAP_Top_Features.csv"
    csv_output_path = os.path.join(CSV_OUTPUT_DIR, CSV_FILENAME)

    # --- Process SHAP results --- Process function now only returns the summary dict
    # shap_summary, top_features_df = process_shap_results(TRANSFORMER_OUTPUT_DIR, SHAP_DATA_SUBDIR)
    shap_summary = process_shap_results(TRANSFORMER_OUTPUT_DIR, SHAP_DATA_SUBDIR)

    # --- Print Console Summary --- (Keep this section)
    if shap_summary:
        print_shap_summary(shap_summary)
    else:
        print(f"\nERROR: Failed to process SHAP results sufficiently for summary. Check errors and debug output above.")
        # Check if the base directory exists, otherwise the error might be misleading
        shap_base_path = os.path.join(TRANSFORMER_OUTPUT_DIR, SHAP_DATA_SUBDIR)
        if not os.path.isdir(shap_base_path):
             print(f"Primary Error: The SHAP data directory does not exist: {shap_base_path}")
        else:
             print(f"Ensure the directory '{shap_base_path}' contains the expected 'shap_importance_*.csv' files.")
        sys.exit(1) # Exit if no summary could be generated


    # --- Generate and Save Publication-Ready CSV --- (Replaces previous CSV saving)
    tissues = ['Leaf', 'Root']
    tasks = ['Genotype', 'Treatment', 'Day']
    output_csv_lines = []
    output_csv_lines.append("Table S3 (SHAP Top Features)") # Add Title
    output_csv_lines.append("") # Add blank line

    try:
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        debug_print(f"\nPreparing publication-ready CSV: {csv_output_path}")

        for tissue in tissues:
            output_csv_lines.append(f"--- {tissue} Tissue --- ") # Add tissue header
            debug_print(f"  Processing {tissue} tissue for CSV")

            # Create base DataFrame with Rank column
            tissue_combined_df = pd.DataFrame({'Rank': range(1, TOP_N_FEATURES + 1)})

            # Merge data for each task
            for task in tasks:
                task_data = shap_summary.get(tissue, {}).get(task, {})
                df_task_top = task_data.get('top_features_df')

                if df_task_top is not None and not df_task_top.empty:
                    # Rename columns for merging
                    df_task_top = df_task_top.rename(columns={
                        'Feature': f'{task}_Feature',
                        'MeanAbsoluteShap': f'{task}_SHAP',
                        'FeatureTypeNorm': f'{task}_FeatureType'
                    })
                    # Select only needed columns + Rank for merge
                    df_task_top = df_task_top[['Rank', f'{task}_Feature', f'{task}_SHAP', f'{task}_FeatureType']]

                    # Merge with the combined tissue dataframe
                    tissue_combined_df = pd.merge(tissue_combined_df, df_task_top, on='Rank', how='left')
                    debug_print(f"    Merged data for {task}")
                else:
                    # If task data is missing, add empty columns
                    debug_print(f"    WARNING: No top features DataFrame found for {tissue}/{task}. Adding empty columns.")
                    for col_suffix in ['Feature', 'SHAP', 'FeatureType']:
                        tissue_combined_df[f'{task}_{col_suffix}'] = pd.NA

            # Define final column order for this tissue's table
            final_columns = ['Rank']
            for task in tasks:
                final_columns.extend([f'{task}_Feature', f'{task}_SHAP', f'{task}_FeatureType'])

            # Reorder columns and fill NA for any potentially missing ones
            tissue_combined_df = tissue_combined_df.reindex(columns=final_columns)
            tissue_combined_df = tissue_combined_df.fillna('') # Replace NaN/NA with empty string for CSV

            # Add the formatted table to the output lines
            output_csv_lines.append(tissue_combined_df.to_csv(index=False, lineterminator='\n'))
            output_csv_lines.append("") # Add blank line between tissues

        # Write the combined lines to the CSV file
        with open(csv_output_path, 'w', newline='') as f:
             # Join lines, ensuring only single line breaks between sections/rows
             f.write("\n".join(line.strip() for line in output_csv_lines if line.strip()))

        print(f"\nSuccessfully saved publication-ready SHAP features table to:")
        print(csv_output_path)

    except Exception as e:
        print(f"\nERROR: Failed to generate or save Publication-Ready CSV to {csv_output_path}")
        print(f"  Reason: {e}")
        debug_print(f"Exception type: {type(e).__name__}, Args: {e.args}")

    # Final success message if we got here without exiting
    print("\nScript finished.")