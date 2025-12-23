import pandas as pd
import os
import sys

# --- Configuration ---
# !!! SET THIS PATH to your MOFA output directory !!!
MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa"
# Example using mofa50 subdirectory:
# MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa\mofa50" # Uncomment if using this


# --- Helper Function for Debug Printing ---
def debug_print(*args, **kwargs):
    """Prints debug messages to stderr."""
    print("DEBUG:", *args, file=sys.stderr, **kwargs)

def get_mofa_selected_feature_counts(output_dir):
    """
    Extracts the count of selected features per view from the MOFA+
    feature selection summary file.

    Args:
        output_dir (str): The directory containing the MOFA+ output files.

    Returns:
        dict: A dictionary mapping view names (e.g., 'leaf_spectral')
              to the number of selected features.
              Returns None if the summary file or required column is missing.
    """
    results = {}
    summary_filename = "mofa_feature_selection_hybrid_summary.csv"
    summary_file_path = os.path.join(output_dir, summary_filename)
    required_column = "N_Final_Selected"
    index_column = "View" # Based on header listing

    debug_print(f"--- Starting Feature Count Extraction ---")
    debug_print(f"Target directory: {output_dir}")
    debug_print(f"Looking for summary file: {summary_file_path}")

    # --- Check for summary file ---
    if not os.path.exists(summary_file_path):
        print(f"ERROR: Summary file not found at {summary_file_path}")
        debug_print(f"File does not exist. Cannot extract counts.")
        # Provide context if file not found
        try:
            debug_print(f"Files/Folders in {output_dir}: {os.listdir(output_dir)}")
        except Exception as e:
            debug_print(f"Could not list files in {output_dir}: {e}")
        return None
    else:
        debug_print(f"Summary file found.")

    # --- Load Data ---
    try:
        debug_print(f"Attempting to load {summary_filename} with index '{index_column}'.")
        df_summary = pd.read_csv(summary_file_path, index_col=index_column)
        debug_print(f"Successfully loaded summary file. Shape: {df_summary.shape}")
        debug_print(f"Columns found: {df_summary.columns.tolist()}")
        debug_print(f"Index values (Views): {df_summary.index.tolist()}")
        debug_print(f"Head of summary data:\n{df_summary.head()}")

    except Exception as e:
        print(f"ERROR: Failed to load or process summary file {summary_file_path}: {e}")
        debug_print(f"Exception type: {type(e).__name__}")
        # Check if it failed because 'View' wasn't the first column
        try:
            debug_print("Trying again without setting index during load...")
            df_summary_alt = pd.read_csv(summary_file_path)
            debug_print(f"Loaded without index_col. Columns: {df_summary_alt.columns.tolist()}")
            if index_column in df_summary_alt.columns:
                 debug_print(f"'{index_column}' column exists. Was it not the first column?")
            else:
                 debug_print(f"'{index_column}' column still not found.")
        except Exception as e2:
            debug_print(f"Secondary load attempt also failed: {e2}")
        return None

    # --- Check for required column ---
    if required_column not in df_summary.columns:
        print(f"ERROR: Required column '{required_column}' not found in {summary_filename}.")
        debug_print(f"Available columns are: {df_summary.columns.tolist()}")
        return None
    else:
        debug_print(f"Required column '{required_column}' found.")

    # --- Extract Counts ---
    try:
        # Ensure the counts are numeric
        counts_series = pd.to_numeric(df_summary[required_column], errors='coerce')
        if counts_series.isnull().any():
            debug_print(f"WARNING: Found non-numeric values in '{required_column}'. Problematic rows:\n{df_summary[counts_series.isnull()]}")
            # Option: Fill NaNs with 0 or raise error. Let's report as is for now.
            results = counts_series.to_dict() # Will include NaNs if any
        else:
            results = counts_series.astype(int).to_dict() # Convert to int if all numeric
        debug_print(f"Extracted counts: {results}")

    except Exception as e:
        print(f"ERROR: Failed to extract or convert counts from column '{required_column}': {e}")
        debug_print(f"Exception type: {type(e).__name__}")
        return None

    debug_print(f"--- Finished Feature Count Extraction ---")
    return results


def print_feature_counts(counts):
    """Prints the extracted feature counts in a formatted way."""
    if not counts:
        print("No feature counts extracted.")
        return

    print("-" * 40)
    print("MOFA+ Selected Feature Counts per View")
    print("-" * 40)
    total_count = 0
    view_map = { # Optional: Map to slightly nicer names if needed
        'leaf_spectral': 'Leaf Spectral',
        'root_spectral': 'Root Spectral',
        'leaf_metabolite': 'Leaf Metabolite',
        'root_metabolite': 'Root Metabolite'
    }
    for view, count in counts.items():
        view_name = view_map.get(view, view) # Use mapped name or original
        if pd.isna(count):
             print(f"  - {view_name}: Error/Missing Value")
        else:
             print(f"  - {view_name}: {count} features")
             total_count += count
    print("-" * 40)
    print(f"Total selected features across views: {total_count}")
    print("-" * 40)


# --- Run Extraction and Print ---
if __name__ == "__main__":
    feature_counts = get_mofa_selected_feature_counts(MOFA_OUTPUT_DIR)

    if feature_counts:
        print_feature_counts(feature_counts)
    else:
        print(f"\nERROR: Failed to get feature counts. Check errors and debug output above.")
        print(f"Ensure the file '{os.path.join(MOFA_OUTPUT_DIR, 'mofa_feature_selection_hybrid_summary.csv')}' exists and contains the column 'N_Final_Selected' with 'View' as the first column or index.")

    # Example of accessing specific count:
    # if feature_counts:
    #     leaf_spec_count = feature_counts.get('leaf_spectral', 0)
    #     debug_print(f"\nExample access: Leaf Spectral count = {leaf_spec_count}")