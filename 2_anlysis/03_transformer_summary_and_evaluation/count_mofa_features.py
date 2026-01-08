import pandas as pd
import os

# --- Configuration ---
# !!! SET THIS PATH to your MOFA output directory !!!
MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa"
# Example using mofa50 subdirectory:
# MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa\mofa50" # Uncomment if using this

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

    # --- Check for summary file ---
    if not os.path.exists(summary_file_path):
        print(f"ERROR: Summary file not found at {summary_file_path}")
        return None

    # --- Load Data ---
    try:
        df_summary = pd.read_csv(summary_file_path, index_col=index_column)

    except Exception as e:
        print(f"ERROR: Failed to load or process summary file {summary_file_path}: {e}")
        return None

    # --- Check for required column ---
    if required_column not in df_summary.columns:
        print(f"ERROR: Required column '{required_column}' not found in {summary_filename}.")
        return None

    # --- Extract Counts ---
    try:
        # Ensure the counts are numeric
        counts_series = pd.to_numeric(df_summary[required_column], errors='coerce')
        if counts_series.isnull().any():
            # Option: Fill NaNs with 0 or raise error. Let's report as is for now.
            results = counts_series.to_dict() # Will include NaNs if any
        else:
            results = counts_series.astype(int).to_dict() # Convert to int if all numeric

    except Exception as e:
        print(f"ERROR: Failed to extract or convert counts from column '{required_column}': {e}")
        return None

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