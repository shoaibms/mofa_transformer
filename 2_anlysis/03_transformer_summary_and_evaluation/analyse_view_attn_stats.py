import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as sp_stats  # Use scipy.stats
from statsmodels.stats.multitest import multipletests  # For FDR correction

# --- Configuration ---
# !!! SET BASE PATHS !!!
TRANSFORMER_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer"
ATTENTION_DATA_LEAF_PATH = os.path.join(TRANSFORMER_OUTPUT_DIR, "v3_feature_attention", "processed_attention_data_leaf", "processed_view_level_attention_Leaf.csv")
ATTENTION_DATA_ROOT_PATH = os.path.join(TRANSFORMER_OUTPUT_DIR, "v3_feature_attention", "processed_attention_data_root", "processed_view_level_attention_Root.csv")

# Metrics and comparisons to perform
METRICS_TO_ANALYZE = ['StdAttn_S2M', 'P95Attn_S2M'] # Add others if needed (e.g., StdAttn_M2S)
ALPHA = 0.05 # Significance level for FDR

# Define the specific comparisons needed
comparisons = [
    # 1. G1 vs G2 under stress (T1) on Day 3
    {'desc': "StdAttn_S2M: G1 vs G2 (T1, Day 3)", 'metric': 'StdAttn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3}},
    {'desc': "StdAttn_S2M: G1 vs G2 (T1, Day 3)", 'metric': 'StdAttn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3}},
    {'desc': "P95Attn_S2M: G1 vs G2 (T1, Day 3)", 'metric': 'P95Attn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3}},
    {'desc': "P95Attn_S2M: G1 vs G2 (T1, Day 3)", 'metric': 'P95Attn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3}},

    # 2. Temporal changes (Day 3 vs Day 1) within G1 under stress (T1)
    {'desc': "StdAttn_S2M: G1 Day 3 vs Day 1 (T1)", 'metric': 'StdAttn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3}, # Day 3 is group 1
     'group2_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 1}}, # Day 1 is group 2
    {'desc': "StdAttn_S2M: G1 Day 3 vs Day 1 (T1)", 'metric': 'StdAttn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 1}},
    {'desc': "P95Attn_S2M: G1 Day 3 vs Day 1 (T1)", 'metric': 'P95Attn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 1}},
    {'desc': "P95Attn_S2M: G1 Day 3 vs Day 1 (T1)", 'metric': 'P95Attn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G1', 'Treatment': 1, 'Day': 1}},

    # 3. Temporal changes (Day 3 vs Day 1) within G2 under stress (T1)
    {'desc': "StdAttn_S2M: G2 Day 3 vs Day 1 (T1)", 'metric': 'StdAttn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 1}},
    {'desc': "StdAttn_S2M: G2 Day 3 vs Day 1 (T1)", 'metric': 'StdAttn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 1}},
    {'desc': "P95Attn_S2M: G2 Day 3 vs Day 1 (T1)", 'metric': 'P95Attn_S2M', 'tissue': 'Leaf',
     'group1_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 1}},
    {'desc': "P95Attn_S2M: G2 Day 3 vs Day 1 (T1)", 'metric': 'P95Attn_S2M', 'tissue': 'Root',
     'group1_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 3},
     'group2_filter': {'Genotype': 'G2', 'Treatment': 1, 'Day': 1}},
]

# --- Helper Function for Debug Printing ---
def debug_print(*args, **kwargs):
    """Prints debug messages to stderr."""
    print("DEBUG:", *args, file=sys.stderr, **kwargs)

# --- Function to apply filters ---
def apply_filter(df, filter_dict):
    """Applies a dictionary of filters to a DataFrame."""
    mask = pd.Series(True, index=df.index)
    for col, value in filter_dict.items():
        if col not in df.columns:
            debug_print(f"  WARNING: Filter column '{col}' not found in DataFrame.")
            return pd.Series(False, index=df.index)  # Return empty if col missing
        mask &= (df[col] == value)
    return mask

# --- Main Processing ---
if __name__ == "__main__":
    debug_print("--- Starting View-Level Attention Statistics Analysis ---")

    # Load data
    try:
        debug_print(f"Loading Leaf data from: {ATTENTION_DATA_LEAF_PATH}")
        if not os.path.exists(ATTENTION_DATA_LEAF_PATH): raise FileNotFoundError
        leaf_attn = pd.read_csv(ATTENTION_DATA_LEAF_PATH)
        debug_print(f"  Leaf data loaded. Shape: {leaf_attn.shape}, Columns: {leaf_attn.columns.tolist()}")

        debug_print(f"Loading Root data from: {ATTENTION_DATA_ROOT_PATH}")
        if not os.path.exists(ATTENTION_DATA_ROOT_PATH): raise FileNotFoundError
        root_attn = pd.read_csv(ATTENTION_DATA_ROOT_PATH)
        debug_print(f"  Root data loaded. Shape: {root_attn.shape}, Columns: {root_attn.columns.tolist()}")

        # Combine for easier processing if needed, or keep separate
        data_map = {'Leaf': leaf_attn, 'Root': root_attn}

    except FileNotFoundError as e:
        print(f"ERROR: Input data file not found: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load input data: {e}")
        sys.exit(1)

    # --- Perform Comparisons ---
    results_list = []
    all_p_values = []

    print("\n--- Performing Comparisons ---")
    for comp in comparisons:
        tissue = comp['tissue']
        metric = comp['metric']
        filter1_dict = comp['group1_filter']
        filter2_dict = comp['group2_filter']
        desc = comp['desc']

        debug_print(f"\nProcessing Comparison: {desc} ({tissue})")
        df = data_map[tissue]

        # Check if metric column exists
        if metric not in df.columns:
            print(f"  ERROR: Metric column '{metric}' not found in {tissue} data. Skipping.")
            results_list.append({'desc': desc, 'tissue': tissue, 'metric': metric, 'error': f"Metric '{metric}' not found"})
            all_p_values.append(np.nan)  # Append NaN placeholder for FDR
            continue

        # Apply filters
        mask1 = apply_filter(df, filter1_dict)
        mask2 = apply_filter(df, filter2_dict)

        group1_data = df.loc[mask1, metric].dropna()
        group2_data = df.loc[mask2, metric].dropna()
        debug_print(f"  Group 1 ({filter1_dict}): n = {len(group1_data)}")
        debug_print(f"  Group 2 ({filter2_dict}): n = {len(group2_data)}")


        result_dict = comp.copy() # Start with comparison info
        result_dict['n1'] = len(group1_data)
        result_dict['n2'] = len(group2_data)

        # Check if groups have enough data for comparison
        if len(group1_data) < 2 or len(group2_data) < 2:
            print(f"  WARNING: Not enough data for comparison '{desc}' in {tissue} (n1={len(group1_data)}, n2={len(group2_data)}). Skipping test.")
            result_dict.update({'mean1': group1_data.mean() if len(group1_data)>0 else np.nan,
                                'mean2': group2_data.mean() if len(group2_data)>0 else np.nan,
                                'std1': group1_data.std() if len(group1_data)>0 else np.nan,
                                'std2': group2_data.std() if len(group2_data)>0 else np.nan,
                                'p_value': np.nan, 'error': 'Insufficient data'})
            all_p_values.append(np.nan)
        else:
            # Calculate stats
            mean1, mean2 = group1_data.mean(), group2_data.mean()
            std1, std2 = group1_data.std(), group2_data.std()
            result_dict['mean1'] = mean1
            result_dict['mean2'] = mean2
            result_dict['std1'] = std1
            result_dict['std2'] = std2

            # Calculate percentage difference (relative to group 2)
            if mean2 != 0 and not pd.isna(mean2):
                result_dict['pct_diff'] = ((mean1 - mean2) / abs(mean2)) * 100
            else:
                result_dict['pct_diff'] = np.nan
                debug_print("  Warning: Cannot calculate % difference because group 2 mean is zero or NaN.")

            # Perform Mann-Whitney U test (non-parametric)
            try:
                stat, p_value = sp_stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                result_dict['p_value'] = p_value
                debug_print(f"  Mann-Whitney U test: p-value = {p_value:.4e}")
                all_p_values.append(p_value)
            except ValueError as ve: # Can happen if all values are identical
                print(f"  WARNING: Mann-Whitney U test failed for '{desc}' (possibly identical groups?): {ve}")
                result_dict['p_value'] = np.nan
                result_dict['error'] = 'Mann-Whitney U failed'
                all_p_values.append(np.nan)


        results_list.append(result_dict)

    # --- Apply FDR Correction ---
    debug_print("\n--- Applying FDR Correction ---")
    valid_p_values = [p for p in all_p_values if not pd.isna(p)]
    if valid_p_values:
        reject, pvals_fdr, _, _ = multipletests(valid_p_values, alpha=ALPHA, method='fdr_bh')
        debug_print(f"  Applied Benjamini-Hochberg FDR correction to {len(valid_p_values)} p-values.")

        # Add FDR results back to the main results list
        fdr_iter = iter(pvals_fdr)
        reject_iter = iter(reject)
        for i, p in enumerate(all_p_values):
            if not pd.isna(p):
                results_list[i]['p_value_fdr'] = next(fdr_iter)
                results_list[i]['significant_fdr'] = next(reject_iter)
            else: # Keep NaNs for tests that weren't performed
                results_list[i]['p_value_fdr'] = np.nan
                results_list[i]['significant_fdr'] = False # Mark as not significant if test failed
    else:
        debug_print("  No valid p-values found to perform FDR correction.")
        for result in results_list:
            result['p_value_fdr'] = np.nan
            result['significant_fdr'] = False

    # --- Print Formatted Results ---
    print("\n" + "="*80)
    print("View-Level Attention Statistics Comparison Results")
    print(f"(Significance based on FDR < {ALPHA})")
    print("="*80)

    # Group results for printing
    grouped_results = {}
    for r in results_list:
        key = (r['tissue'], r['desc']) # Group by tissue and description
        if key not in grouped_results: grouped_results[key] = []
        grouped_results[key].append(r)

    for (tissue, desc), results in grouped_results.items():
        print(f"\nComparison: {desc} ({tissue})")
        if not results: continue

        # Assume only one metric per description/tissue combo in our setup
        r = results[0]
        if 'error' in r:
            print(f"  Status: {r['error']}")
            continue

        g1_label = list(r['group1_filter'].items())
        g2_label = list(r['group2_filter'].items())
        print(f"  Group 1 ({g1_label}): Mean={r['mean1']:.4f} +/- {r['std1']:.4f} (n={r['n1']})")
        print(f"  Group 2 ({g2_label}): Mean={r['mean2']:.4f} +/- {r['std2']:.4f} (n={r['n2']})")
        if not pd.isna(r['pct_diff']):
            print(f"  % Difference (G1 vs G2): {r['pct_diff']:.1f}%")
        print(f"  P-value (Mann-Whitney U): {r['p_value']:.3e}" if not pd.isna(r['p_value']) else "  P-value (Mann-Whitney U): N/A")
        print(f"  FDR Adjusted P-value:   {r['p_value_fdr']:.3e}" if not pd.isna(r['p_value_fdr']) else "  FDR Adjusted P-value:   N/A")
        print(f"  Significant (FDR < {ALPHA}): {r['significant_fdr']}")

    print("\n" + "="*80)