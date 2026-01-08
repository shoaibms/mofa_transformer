import pandas as pd
import numpy as np
import os

# --- Configuration ---
# !!! SET THIS PATH to your MOFA output directory !!!
MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa"
# Example using mofa50 subdirectory:
# MOFA_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\mofa\mofa50"

def extract_mofa_summary(output_dir):
    """
    Extracts summary statistics from MOFA+ output files, using precise file
    names and headers from file_listing_mofa.txt. Includes enhanced debugging
    and revised variance/validation handling.

    Args:
        output_dir (str): The directory containing the MOFA+ output files.

    Returns:
        dict: A dictionary containing the extracted summary information.
              Returns None if essential files are missing.
    """
    results = {}
    if not os.path.isdir(output_dir):
        print(f"CRITICAL ERROR: Output directory not found or is not a directory: {output_dir}")
        return None

    required_files = {
        "variance": os.path.join(output_dir, "mofa_variance_explained_active.csv"),
        "associations": os.path.join(output_dir, "mofa_factor_metadata_associations_spearman.csv"),
        "factors": os.path.join(output_dir, "mofa_latent_factors_active.csv"),
        "weights_leaf_spec": os.path.join(output_dir, "mofa_feature_weights_leaf_spectral_active.csv"),
        "weights_root_spec": os.path.join(output_dir, "mofa_feature_weights_root_spectral_active.csv"),
        "weights_leaf_met": os.path.join(output_dir, "mofa_feature_weights_leaf_metabolite_active.csv"),
        "weights_root_met": os.path.join(output_dir, "mofa_feature_weights_root_metabolite_active.csv"),
        # Optional validation files (using names from listing)
        "permutation": os.path.join(output_dir, "mofa_permutation_test_results_HYPOTHETICAL.csv"), # Name from list
        "bootstrap": os.path.join(output_dir, "mofa_bootstrap_stability_HYPOTHETICAL.csv"),      # Name from list
    }

    # --- Check for essential files ---
    essential_files_present = True
    essential_keys = ["variance", "associations", "factors", "weights_leaf_spec",
                      "weights_root_spec", "weights_leaf_met", "weights_root_met"]
    for key in essential_keys:
        fpath = required_files[key]
        if not os.path.exists(fpath):
            print(f"Error: Essential file missing - {fpath}")
            essential_files_present = False
    if not essential_files_present:
        return None

    # --- Load Data ---
    df_variance = df_associations = df_factors = None
    df_weights = {}
    try:
        # variance: Factor(index), leaf_spectral, root_spectral, leaf_metabolite, root_metabolite
        df_variance = pd.read_csv(required_files["variance"], index_col=0)
        df_variance.index.name = 'Factor' # Set index name
        # associations: Factor, Metadata, Correlation, P_value, Note, P_value_FDR, Significant_FDR
        df_associations = pd.read_csv(required_files["associations"])
        # factors: MasterReferenceID(index), Factor1, Factor2, ...
        df_factors = pd.read_csv(required_files["factors"], index_col=0) # MasterReferenceID is index
        # weights: FeatureName(index), Factor1, Factor2, ...
        for view_key, view_name in [("weights_leaf_spec", "Leaf Spec"), ("weights_root_spec", "Root Spec"),
                                    ("weights_leaf_met", "Leaf Met"), ("weights_root_met", "Root Met")]:
            fpath = required_files[view_key]
            df_weights[view_name] = pd.read_csv(fpath, index_col=0)
            df_weights[view_name].index.name = 'Feature' # Set index name

    except Exception as e:
        print(f"CRITICAL ERROR loading essential data files: {e}")
        return None

    # --- Basic Information ---
    results["unique_biological_samples"] = 336 # From background_ex.txt
    results["mofa_model_samples"] = df_factors.shape[0]
    results["active_factors_count"] = df_variance.shape[0]
    results["initial_factors_count"] = 'N/A'

    # --- Variance Explained (Reporting RAW values from file) ---
    try:
        # Convert variance columns to numeric, coercing errors
        for col in df_variance.columns:
            df_variance[col] = pd.to_numeric(df_variance[col], errors='coerce')

        if df_variance.isnull().values.any():
             df_variance = df_variance.fillna(0)

        # Reporting RAW values/sums directly from the file
        raw_variance_sum_per_view = df_variance.sum(axis=0).round(3)
        raw_variance_total_sum = df_variance.values.sum().round(3)

        variance_warning = ("NOTE: Reporting raw variance values/sums directly from "
                            "'mofa_variance_explained_active.csv'. Units and interpretation "
                            "(e.g., R-squared, scaled variance) depend on MOFA+ generation settings.")

        results["variance_note"] = variance_warning
        results["variance_sum_per_view_RAW"] = raw_variance_sum_per_view
        results["total_variance_sum_RAW"] = raw_variance_total_sum
        results["variance_details_RAW"] = df_variance # Store raw values for lookup

    except Exception as e:
        print(f"ERROR processing variance explained: {e}")
        results["variance_note"] = "ERROR during variance processing."
        results["variance_sum_per_view_RAW"] = "Error"
        results["total_variance_sum_RAW"] = "Error"


    # --- Identify Key Factors ---
    # (Keep the previous robust logic for finding key factors and handling overlap)
    try:
        # Ensure significance column exists and handle types robustly
        significant_col = None
        significance_source = "None" # Track how significance was determined

        # Option 1: Check for 'Real_Significant_FDR' (Specific to Permutation file naming)
        if 'Real_Significant_FDR' in df_associations.columns:
             significant_col = 'Real_Significant_FDR'
             significance_source = 'Real_Significant_FDR column'
        # Option 2: Check generic 'Significant_FDR'
        elif 'Significant_FDR' in df_associations.columns:
             significant_col = 'Significant_FDR'
             significance_source = 'Significant_FDR column'
        # Option 3: Compute from 'P_value_FDR'
        elif 'P_value_FDR' in df_associations.columns:
            df_associations['P_value_FDR'] = pd.to_numeric(df_associations['P_value_FDR'], errors='coerce')
            if not df_associations['P_value_FDR'].isnull().all(): # Check if P_value_FDR exists and is useful
                df_associations['Significant_Computed'] = (df_associations['P_value_FDR'] < 0.05)
                significant_col = 'Significant_Computed'
                significance_source = 'Computed from P_value_FDR < 0.05'
            else:
                 significant_col = None # Cannot determine significance
        else:
             significant_col = None # Cannot determine significance

        if significant_col:
             # Convert the chosen significance column to boolean robustly
             if df_associations[significant_col].dtype == 'O': # Object type
                 df_associations[significant_col] = df_associations[significant_col].astype(str).str.lower().map(
                     {'true': True, '1': True, '1.0': True, # Handle various True representations
                      'false': False, '0': False, '0.0': False, # Handle various False representations
                      'nan': False, 'none': False, '': False}
                 ).fillna(False).astype(bool)
             elif pd.api.types.is_numeric_dtype(df_associations[significant_col]):
                  df_associations[significant_col] = df_associations[significant_col].fillna(0).astype(bool)
             elif pd.api.types.is_bool_dtype(df_associations[significant_col]):
                  df_associations[significant_col] = df_associations[significant_col].fillna(False)
             else: # Fallback for unexpected types
                  try:
                      df_associations[significant_col] = df_associations[significant_col].astype(bool)
                  except Exception:
                       df_associations[significant_col] = False

             significant_assoc = df_associations[df_associations[significant_col] == True].copy()
        else:
             print("WARNING: Could not determine significance from associations file. Reporting raw correlations.")
             significant_assoc = df_associations.copy() # Process all if significance unknown

        # Ensure Correlation is numeric
        significant_assoc['Correlation'] = pd.to_numeric(significant_assoc['Correlation'], errors='coerce')
        significant_assoc = significant_assoc.dropna(subset=['Correlation'])
        significant_assoc['Abs_Correlation'] = significant_assoc['Correlation'].abs()

        key_factors = {}
        assigned_factors = set()
        primary_metadata_vars = {'Genotype': 'Genotype', 'Time': 'Day', 'Treatment': 'Treatment'}
        secondary_metadata_vars = {'Batch': 'Batch'} # Add others if needed

        # (Factor assignment logic remains the same as previous version)
        for name, meta_col in primary_metadata_vars.items():
            assoc_subset = significant_assoc[significant_assoc['Metadata'] == meta_col].copy()
            if not assoc_subset.empty:
                assoc_subset.sort_values('Abs_Correlation', ascending=False, inplace=True)
                found_unassigned = False
                for idx, row in assoc_subset.iterrows():
                    factor_id = row['Factor']
                    if factor_id not in assigned_factors:
                        key_factors[name] = {
                            "FactorID": factor_id, "Rho": row['Correlation'],
                            "FDR": row.get('P_value_FDR', row.get('Real_P_value_FDR', 'N/A')) # Get FDR if exists
                        }
                        assigned_factors.add(factor_id)
                        found_unassigned = True
                        break
                if not found_unassigned:
                     best_row = assoc_subset.iloc[0]
                     factor_id = best_row['Factor']
                     key_factors[name] = {
                         "FactorID": factor_id, "Rho": best_row['Correlation'],
                         "FDR": best_row.get('P_value_FDR', best_row.get('Real_P_value_FDR', 'N/A')),
                         "Note": f"Factor also strongly associated with other primary variables."
                     }
            else:
                key_factors[name] = None

        for name, meta_col in secondary_metadata_vars.items():
             assoc_subset = significant_assoc[significant_assoc['Metadata'] == meta_col].copy()
             if not assoc_subset.empty:
                assoc_subset.sort_values('Abs_Correlation', ascending=False, inplace=True)
                best_row = assoc_subset.iloc[0]
                factor_id = best_row['Factor']
                is_primary = factor_id in assigned_factors
                note = f"Factor may also correlate with primary variables." if is_primary else ""
                key_factors[name] = {
                    "FactorID": factor_id, "Rho": best_row['Correlation'],
                    "FDR": best_row.get('P_value_FDR', best_row.get('Real_P_value_FDR', 'N/A')), "Note": note
                }
             else:
                 key_factors[name] = None

        results["key_factors"] = key_factors
    except Exception as e:
        print(f"ERROR identifying key factors: {e}")
        results["key_factors"] = {}


    # --- Variance Attributed by Key Factors (Using RAW values) ---
    variance_by_key_factors_raw = {}
    df_var_raw = results.get("variance_details_RAW")

    if df_var_raw is not None:
        key_factor_data = results.get("key_factors", {})
        for factor_type in ["Genotype", "Time", "Treatment"]:
            if key_factor_data.get(factor_type):
                factor_id = key_factor_data[factor_type]["FactorID"]
                if factor_id in df_var_raw.index:
                    variance_by_key_factors_raw[factor_type] = {}
                    # Define relevant views for reporting
                    if factor_type == "Genotype": views = ['root_metabolite', 'leaf_metabolite']
                    elif factor_type == "Time": views = ['leaf_spectral', 'root_spectral']
                    elif factor_type == "Treatment": views = ['leaf_spectral', 'leaf_metabolite']
                    else: views = []

                    for view_col_name in views:
                         if view_col_name in df_var_raw.columns:
                             raw_val = df_var_raw.loc[factor_id, view_col_name]
                             variance_by_key_factors_raw[factor_type][view_col_name] = round(raw_val, 4) # Store raw value

    results["variance_by_key_factors_RAW"] = variance_by_key_factors_raw


    # --- Top Features for Key Factors ---
    # (Logic remains the same, relies on index being correct)
    def get_top_features(factor_id, view_weights_df, n=5):
        if factor_id not in view_weights_df.columns:
             return [], []
        try:
            numeric_weights = pd.to_numeric(view_weights_df[factor_id], errors='coerce').dropna()
            if numeric_weights.empty:
                return [], []
            # Ensure index is available for feature names
            if view_weights_df.index.name is None or view_weights_df.index.empty:
                 return [], []
            top_pos = numeric_weights.nlargest(n).index.tolist()
            top_neg = numeric_weights.nsmallest(n).index.tolist()
            return top_pos, top_neg
        except Exception:
             return [], []

    top_features = {}
    factors_to_analyze = ["Genotype", "Time", "Treatment"]
    views_to_check = {
        "Genotype": ["Root Met", "Leaf Met"],
        "Time": ["Leaf Spec", "Root Spec", "Root Met", "Leaf Met"],
        "Treatment": ["Leaf Spec", "Root Spec", "Leaf Met", "Root Met"],
    }

    key_factor_data = results.get("key_factors", {})
    for factor_type in factors_to_analyze:
        if key_factor_data.get(factor_type):
            factor_id = key_factor_data[factor_type]["FactorID"]
            top_features[factor_type] = {}
            for view_name in views_to_check.get(factor_type, []):
                 if view_name in df_weights:
                     pos_feats, neg_feats = get_top_features(factor_id, df_weights[view_name], n=5)
                     top_features[factor_type][f"{view_name}_TopPositive"] = pos_feats
                     top_features[factor_type][f"{view_name}_TopNegative"] = neg_feats

    results["top_features_by_key_factor"] = top_features


    # --- Optional: Validation Statistics ---
    results["validation"] = {}
    perm_file = required_files["permutation"]
    boot_file = required_files["bootstrap"]

    # Permutation File Processing (Revised Significance Check)
    if os.path.exists(perm_file):
        try:
            df_perm = pd.read_csv(perm_file)
            # Headers: Factor, Tested_Metadata, Real_Association_Metric, Real_P_value_FDR, Real_Significant_FDR, Hypothetical_Permutation_P_Value

            num_significant = "N/A"
            total_tested = df_perm.shape[0]
            significance_col_perm = None

            if 'Real_Significant_FDR' in df_perm.columns:
                 significance_col_perm = 'Real_Significant_FDR'
                 # Convert to boolean robustly
                 if df_perm[significance_col_perm].dtype == 'O':
                     df_perm[significance_col_perm] = df_perm[significance_col_perm].astype(str).str.lower().map(
                         {'true': True, 'false': False, 'nan': False, 'none': False, '': False}
                     ).fillna(False).astype(bool)
                 else:
                      df_perm[significance_col_perm] = df_perm[significance_col_perm].fillna(0).astype(bool)
                 num_significant = df_perm[df_perm[significance_col_perm] == True].shape[0]

            elif 'Hypothetical_Permutation_P_Value' in df_perm.columns:
                 p_val_col = 'Hypothetical_Permutation_P_Value'
                 df_perm[p_val_col] = pd.to_numeric(df_perm[p_val_col], errors='coerce')
                 if not df_perm[p_val_col].isnull().all():
                      num_significant = df_perm[df_perm[p_val_col] < 0.05].shape[0]
                 else:
                      num_significant = "N/A (Bad P-values)"
            else:
                 num_significant = "N/A (Column missing)"

            results["validation"]["permutation_significant_count"] = num_significant
            results["validation"]["permutation_total_tested"] = total_tested

        except Exception as e:
            print(f"WARNING: Could not process permutation file {perm_file}: {e}")
            results["validation"]["permutation_significant_count"] = "Error"
            results["validation"]["permutation_total_tested"] = "Error"
    else:
        print(f"INFO: Optional permutation file not found: {perm_file}")
        results["validation"]["permutation_significant_count"] = "Not Found"
        results["validation"]["permutation_total_tested"] = "Not Found"


    # Bootstrap File Processing (Keep previous logic, check bins)
    if os.path.exists(boot_file):
        try:
            df_boot = pd.read_csv(boot_file)
            # Headers: Factor, View, Feature, StabilityScore, Rank_AbsWeight_OriginalRun
            results["validation"]["bootstrap_total_unique_features"] = "N/A"
            results["validation"]["bootstrap_high_confidence_pct"] = "N/A"

            if 'Feature' not in df_boot.columns:
                 print("WARNING: 'Feature' column missing in bootstrap file. Cannot calculate unique features or percentage.")
            else:
                 total_unique_features = df_boot['Feature'].nunique()
                 results["validation"]["bootstrap_total_unique_features"] = total_unique_features

                 if 'StabilityScore' in df_boot.columns:
                     df_boot['StabilityScore'] = pd.to_numeric(df_boot['StabilityScore'], errors='coerce')

                     # --- !!! User Verification Needed: Adjust bins based on StabilityScore distribution and desired 'high' threshold !!! ---
                     # Current bins assume score 0-1, high >= 0.8
                     bins = [0, 0.2, 0.5, 0.8, df_boot['StabilityScore'].max() + 0.1]
                     labels = ['very_low', 'low', 'medium', 'high']

                     df_boot['ConfidenceTier'] = pd.cut(df_boot['StabilityScore'], bins=bins, labels=labels, right=False, include_lowest=True)

                     # Calculate count of UNIQUE features in 'high' tier
                     high_confidence_unique_count = df_boot.loc[df_boot['ConfidenceTier'] == 'high', 'Feature'].nunique()

                     if total_unique_features > 0:
                         high_conf_pct = (high_confidence_unique_count / total_unique_features) * 100
                         results["validation"]["bootstrap_high_confidence_pct"] = round(high_conf_pct, 1)
                     else:
                          results["validation"]["bootstrap_high_confidence_pct"] = 0.0
                 else:
                      print("WARNING: 'StabilityScore' column missing in bootstrap file. Cannot determine confidence.")
                      results["validation"]["bootstrap_high_confidence_pct"] = "N/A (Column missing)"

        except Exception as e:
            print(f"WARNING: Could not process bootstrap file {boot_file}: {e}")
            results["validation"]["bootstrap_high_confidence_pct"] = "Error"
    else:
        print(f"INFO: Optional bootstrap file not found: {boot_file}")
        results["validation"]["bootstrap_high_confidence_pct"] = "Not Found"

    return results

# --- (print_summary function adjusted for RAW variance reporting) ---
def print_summary(results):
    """Prints the extracted summary results in a formatted way."""
    if not results:
        print("No results to print.")
        return

    print("-" * 60)
    print("MOFA+ Summary Report (Debugged v3)")
    print("-" * 60)

    print("\n--- Basic Information ---")
    print(f"Unique Biological Samples: {results.get('unique_biological_samples', 'N/A')}")
    print(f"Samples used in MOFA+ Model: {results.get('mofa_model_samples', 'N/A')}")
    print(f"Initial Factors Requested/Calculated: {results.get('initial_factors_count', 'N/A')}")
    print(f"Active Factors Identified: {results.get('active_factors_count', 'N/A')}")

    print("\n--- Variance Explained (Raw Values/Sums from File) ---")
    if results.get("variance_note"):
        print(results["variance_note"]) # Print note about interpretation
    print(f"Sum of Variance Values Across All Factors/Views: {results.get('total_variance_sum_RAW', 'N/A')}")
    print("Sum of Variance Values per View (Across Factors):")
    if isinstance(results.get("variance_sum_per_view_RAW"), pd.Series):
        for view, variance in results["variance_sum_per_view_RAW"].items():
            view_clean = view.replace('_', ' ').title()
            print(f"  - {view_clean}: {variance:.3f}") # Report raw value
    elif results.get("variance_sum_per_view_RAW") == "Error":
         print("  Error processing variance.")
    else:
        print("  N/A")
    print("(Note: Interpret these values based on MOFA+ generation settings. They may not be % R-squared.)")


    print("\n--- Key Factor Associations (FDR < 0.05 cutoff where available) ---")
    key_factors = results.get("key_factors", {})
    factor_map = {}
    displayed_factors = set() # Track factors already displayed with primary role

    # Display Primary Factors First
    for name in ['Genotype', 'Time', 'Treatment']:
        data = key_factors.get(name)
        if data:
            factor_id = data['FactorID']
            factor_map[name] = factor_id
            displayed_factors.add(factor_id)
            note = data.get('Note', '')
            fdr_val = data.get('FDR', 'N/A')
            fdr_str = f"{fdr_val:.3e}" if isinstance(fdr_val, (float, np.number)) and not pd.isna(fdr_val) else str(fdr_val) # Format FDR
            print(f"{name} Factor: {factor_id} (Rho={data['Rho']:.3f}, FDR={fdr_str}) {note}")

            # Print variance VALUE by this factor where available
            if name in results.get("variance_by_key_factors_RAW", {}):
                 print(f"  Variance Value in Key Views (from file):")
                 for view_col, raw_val in results["variance_by_key_factors_RAW"][name].items():
                      view_clean = view_col.replace('_', ' ').title()
                      print(f"    - {view_clean}: {raw_val:.4f}") # Report raw value
        else:
            print(f"{name} Factor: No significant association found or assigned (FDR < 0.05 where applicable)")

    # Display Secondary/Other Factors (like Batch)
    for name, data in key_factors.items():
         if name not in ['Genotype', 'Time', 'Treatment']: # Check if it's a secondary factor
            if data:
                factor_id = data['FactorID']
                factor_map[name] = factor_id
                note = data.get('Note', '')
                if factor_id in displayed_factors:
                     note += f" (Note: Also associated with primary variables)"
                fdr_val = data.get('FDR', 'N/A')
                fdr_str = f"{fdr_val:.3e}" if isinstance(fdr_val, (float, np.number)) and not pd.isna(fdr_val) else str(fdr_val)
                print(f"{name} Factor: {factor_id} (Rho={data['Rho']:.3f}, FDR={fdr_str}) {note}")
            else:
                 print(f"{name} Factor: No significant association found (FDR < 0.05 where applicable)")


    print("\n--- Top Features for Key Factors (Example: Top 5) ---")
    # (Keep the same printing logic for features)
    top_features = results.get("top_features_by_key_factor", {})
    for factor_type, views_data in top_features.items():
         factor_id = key_factors.get(factor_type, {}).get("FactorID", "N/A")
         if factor_id != "N/A":
             print(f"\nFactor Type: {factor_type} ({factor_id})")
             for view_key, features in views_data.items():
                 weight_type = "Positive" if "Positive" in view_key else "Negative"
                 view_name = view_key.replace("_TopPositive", "").replace("_TopNegative", "").replace('_', ' ').title()
                 if features:
                     print(f"  Top {weight_type} Weights in {view_name}:")
                     print(f"    {', '.join(map(str,features))}")


    print("\n--- Optional Validation Summary ---")
    validation = results.get("validation", {})
    perm_sig = validation.get("permutation_significant_count", "N/A")
    perm_tot = validation.get("permutation_total_tested", "N/A")
    if perm_sig == "Not Found":
         print("Permutation Test: File specified not found.")
    elif perm_sig == "Error":
         print("Permutation Test: Error processing results.")
    elif perm_sig == "N/A (Column missing)" or perm_sig == "N/A (Bad P-values)":
         print(f"Permutation Test: Could not determine significance ({perm_sig}). Total tested: {perm_tot}.")
    else:
         print(f"Permutation Test: {perm_sig} / {perm_tot} factor-metadata associations confirmed significant.")

    boot_pct = validation.get("bootstrap_high_confidence_pct", "N/A")
    boot_feat = validation.get("bootstrap_total_unique_features", "N/A")
    if boot_pct == "Not Found":
        print("Bootstrap Stability: File specified not found.")
    elif boot_pct == "Error":
        print("Bootstrap Stability: Error processing results.")
    elif boot_pct == "N/A (Column missing)":
         print("Bootstrap Stability: Cannot calculate % (StabilityScore column missing).")
    elif isinstance(boot_pct, (int, float)):
        print(f"Bootstrap Stability: {boot_pct:.1f}% of features ({boot_feat} unique) potentially showed high selection consistency (e.g., StabilityScore >= 0.8).")
    else:
         print(f"Bootstrap Stability: {boot_pct} ({boot_feat} unique features)")


    print("-" * 60)


# --- Run Extraction and Print ---
if __name__ == "__main__":
    summary_data = extract_mofa_summary(MOFA_OUTPUT_DIR)
    if summary_data:
        print_summary(summary_data)
    else:
        print(f"\nExecution failed. Check file paths and error messages in stderr output above in: {MOFA_OUTPUT_DIR}")