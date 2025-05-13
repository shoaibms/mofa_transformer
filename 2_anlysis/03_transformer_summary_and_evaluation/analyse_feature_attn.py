import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# --- Configuration ---
# !!! SET BASE PATHS !!!
TRANSFORMER_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer"
ATTENTION_SUBDIR = os.path.join("v3_feature_attention") # Subdir containing processed_attention_data_*
OUTPUT_DIR_NOVILITY = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot" # Output dir for summary CSVs
OUTPUT_FILENAME_COND_ATTN = "S(Z) (Top Cond. Attn Pairs).csv" # Filename for top pairs CSV

# Parameters
TOP_N_HIGH_ATTN_EXAMPLES = 2       # Number of top attention pairs to show as EXAMPLES in summary printout
TOP_N_CONDITIONAL_PAIRS = 30       # Number of top attention pairs to save per condition in CSV
TOP_N_FOR_G1_G2_COMP = 100         # Compare G1/G2 strength on these top overall pairs
TOP_N_FOR_TEMPORAL = 500           # Analyze trends within this many top pairs (from trends file)
EARLY_DEV_DAY = 2                  # Define "early" development as showing strength by this day
MIN_ATTN_DIFF_TEMPORAL = 0.01      # Minimum attention increase needed to consider "earlier development"

# --- Helper Function for Debug Printing ---
def debug_print(*args, **kwargs):
    """Prints debug messages to stderr."""
    print("DEBUG:", *args, file=sys.stderr, **kwargs)

def analyze_conditional_attention(transformer_dir, attention_subdir):
    """
    Analyzes conditional feature-level attention files to extract specific
    examples, quantitative comparisons, and top N lists per condition.

    Args:
        transformer_dir (str): The base directory containing transformer outputs.
        attention_subdir (str): Subdirectory path containing processed attention data.

    Returns:
        dict: Nested dictionary containing extracted results. None on error.
    """
    base_path = os.path.join(transformer_dir, attention_subdir)
    results = {'Leaf': {}, 'Root': {}}
    required_dirs = {
        'leaf': os.path.join(base_path, "processed_attention_data_leaf"),
        'root': os.path.join(base_path, "processed_attention_data_root")
    }

    debug_print("--- Starting Conditional Attention Analysis ---")
    debug_print(f"Base Attention Data Dir: {base_path}")

    # Check directories
    for tissue, dir_path in required_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"ERROR: Required directory not found: {dir_path}")
            return None

    all_results = {} # Store results per tissue

    for tissue in ['Leaf', 'Root']:
        debug_print(f"\n--- Processing Tissue: {tissue} ---")
        tissue_results = {}
        data_dir = required_dirs[tissue.lower()]

        # --- Load Data ---
        try:
            cond_file = os.path.join(data_dir, f"processed_mean_attention_conditional_{tissue}.csv")
            trends_file = os.path.join(data_dir, f"processed_attention_trends_top_500_{tissue}.csv")
            top_overall_file = os.path.join(data_dir, f"processed_top_500_pairs_overall_{tissue}.csv") # Assumes S2M

            debug_print(f"  Loading conditional data: {os.path.basename(cond_file)}")
            if not os.path.exists(cond_file): raise FileNotFoundError(cond_file)
            df_cond = pd.read_csv(cond_file)
            # Ensure numeric type for attention
            df_cond['Mean_Attention_S2M_Group_AvgHeads'] = pd.to_numeric(df_cond['Mean_Attention_S2M_Group_AvgHeads'], errors='coerce')
            df_cond.dropna(subset=['Mean_Attention_S2M_Group_AvgHeads'], inplace=True)

            debug_print(f"  Loading trends data: {os.path.basename(trends_file)}")
            if not os.path.exists(trends_file): raise FileNotFoundError(trends_file)
            df_trends = pd.read_csv(trends_file)
            df_trends['Mean_Attention_S2M_Group_AvgHeads'] = pd.to_numeric(df_trends['Mean_Attention_S2M_Group_AvgHeads'], errors='coerce')
            df_trends.dropna(subset=['Mean_Attention_S2M_Group_AvgHeads'], inplace=True)


            debug_print(f"  Loading top overall pairs: {os.path.basename(top_overall_file)}")
            if not os.path.exists(top_overall_file): raise FileNotFoundError(top_overall_file)
            df_top_overall = pd.read_csv(top_overall_file)
            # Select top N pairs from this list for G1/G2 comparison
            top_overall_pairs_set = set(zip(df_top_overall['Spectral_Feature'].head(TOP_N_FOR_G1_G2_COMP),
                                            df_top_overall['Metabolite_Feature'].head(TOP_N_FOR_G1_G2_COMP)))
            debug_print(f"    Selected Top {len(top_overall_pairs_set)} unique overall pairs for G1/G2 comparison.")

        except FileNotFoundError as e:
            print(f"ERROR: Required data file not found: {e.filename}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load or process data files for {tissue}: {e}")
            return None


        # --- a) Top S2M attention links under key conditions (T1 Day 3 Overall, G1, G2) ---
        debug_print(f"\n  a) Finding Top {TOP_N_CONDITIONAL_PAIRS} S2M Links under Key Conditions (T1, Day 3)...")
        df_t1_d3 = df_cond[(df_cond['Treatment'] == 1) & (df_cond['Day'] == 3)].copy()
        df_g1_t1d3 = df_cond[(df_cond['Genotype'] == 'G1') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == 3)].copy()
        df_g2_t1d3 = df_cond[(df_cond['Genotype'] == 'G2') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == 3)].copy()

        # Helper to extract top N unique pairs
        def get_top_n_pairs(df, n):
            sorted_df = df.sort_values('Mean_Attention_S2M_Group_AvgHeads', ascending=False)
            top_pairs_list = []
            unique_pairs_seen = set()
            for _, row in sorted_df.iterrows():
                 pair = (row['Spectral_Feature'], row['Metabolite_Feature'])
                 # For overall T1D3, ensure uniqueness across G1/G2 if they appear separately
                 # For G1/G2 specific lists, uniqueness check isn't strictly needed but doesn't hurt
                 if pair not in unique_pairs_seen:
                     top_pairs_list.append({
                         'spectral': row['Spectral_Feature'],
                         'metabolite': row['Metabolite_Feature'],
                         'mean_attn': row['Mean_Attention_S2M_Group_AvgHeads'],
                         # 'condition': f"{row.get('Genotype', 'Overall')}, T1, D3" # Example for potential inclusion
                     })
                     unique_pairs_seen.add(pair)
                 if len(top_pairs_list) >= n:
                     break
            return top_pairs_list

        # Get Top N for each condition
        top_pairs_t1d3_overall = get_top_n_pairs(df_t1_d3, TOP_N_CONDITIONAL_PAIRS)
        top_pairs_g1_t1d3 = get_top_n_pairs(df_g1_t1d3, TOP_N_CONDITIONAL_PAIRS)
        top_pairs_g2_t1d3 = get_top_n_pairs(df_g2_t1d3, TOP_N_CONDITIONAL_PAIRS)

        tissue_results['top_pairs_t1d3_overall'] = top_pairs_t1d3_overall
        tissue_results['top_pairs_g1_t1d3'] = top_pairs_g1_t1d3
        tissue_results['top_pairs_g2_t1d3'] = top_pairs_g2_t1d3
        debug_print(f"    Found Top {len(top_pairs_t1d3_overall)} Overall T1D3 pairs.")
        debug_print(f"    Found Top {len(top_pairs_g1_t1d3)} G1 T1D3 pairs.")
        debug_print(f"    Found Top {len(top_pairs_g2_t1d3)} G2 T1D3 pairs.")

        # Keep top N examples for summary printout (using the overall list)
        tissue_results['top_stress_pair_examples'] = top_pairs_t1d3_overall[:TOP_N_HIGH_ATTN_EXAMPLES]


        # --- b) Hub features (Proxy: Features from top overall T1D3 links) ---
        debug_print(f"\n  b) Identifying Proxy Hub Features (from top {TOP_N_HIGH_ATTN_EXAMPLES} overall T1D3 links)...")
        hub_features = set()
        # Use the example list generated above for consistency with printout
        for pair_info in tissue_results.get('top_stress_pair_examples', []):
            hub_features.add(pair_info['spectral'])
            hub_features.add(pair_info['metabolite'])
        tissue_results['proxy_hubs'] = list(hub_features)
        debug_print(f"    Proxy hubs: {list(hub_features)}")


        # --- c) Quantitative comparison (G1 vs G2 under stress, Day 3, Top N overall pairs) ---
        debug_print(f"\n  c) Comparing G1 vs G2 Mean Attention on Top {TOP_N_FOR_G1_G2_COMP} Overall Pairs (T1, Day 3)...")
        df_g1_t1d3 = df_cond[(df_cond['Genotype'] == 'G1') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == 3)].copy()
        df_g2_t1d3 = df_cond[(df_cond['Genotype'] == 'G2') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == 3)].copy()

        # Create lookup dictionaries for faster access
        g1_attn_map = {(row['Spectral_Feature'], row['Metabolite_Feature']): row['Mean_Attention_S2M_Group_AvgHeads']
                       for _, row in df_g1_t1d3.iterrows()}
        g2_attn_map = {(row['Spectral_Feature'], row['Metabolite_Feature']): row['Mean_Attention_S2M_Group_AvgHeads']
                       for _, row in df_g2_t1d3.iterrows()}

        g1_means = []
        g2_means = []
        pairs_found_count = 0
        for pair in top_overall_pairs_set:
            g1_val = g1_attn_map.get(pair)
            g2_val = g2_attn_map.get(pair)
            # Only include pairs where we have data for both G1 and G2 under T1D3
            if g1_val is not None and g2_val is not None:
                g1_means.append(g1_val)
                g2_means.append(g2_val)
                pairs_found_count += 1

        if pairs_found_count > 0:
            avg_g1_attn = np.mean(g1_means)
            avg_g2_attn = np.mean(g2_means)
            fold_change = avg_g1_attn / avg_g2_attn if avg_g2_attn != 0 else np.inf
            pct_diff = ((avg_g1_attn - avg_g2_attn) / avg_g2_attn * 100) if avg_g2_attn != 0 else np.inf

            tissue_results['g1_g2_comparison'] = {
                'n_pairs_compared': pairs_found_count,
                'avg_g1_attn': avg_g1_attn,
                'avg_g2_attn': avg_g2_attn,
                'fold_change_g1_vs_g2': fold_change,
                'pct_diff_g1_vs_g2': pct_diff
            }
            debug_print(f"    Compared {pairs_found_count} pairs found in both G1/G2 under T1D3.")
            debug_print(f"    Avg Attn G1: {avg_g1_attn:.4f}, Avg Attn G2: {avg_g2_attn:.4f}")
            debug_print(f"    Fold Change (G1/G2): {fold_change:.2f}, Pct Diff: {pct_diff:.1f}%")
        else:
             tissue_results['g1_g2_comparison'] = {'error': f"No overlapping pairs found between Top {TOP_N_FOR_G1_G2_COMP} overall and T1D3 data."}
             debug_print(f"    ERROR: No common pairs found for comparison.")


        # --- d) Earlier attention development in G1 ---
        debug_print(f"\n  d) Searching for Earlier Attention Development in G1 (T1)...")
        early_dev_examples = []
        # Pivot the trends table for easier comparison
        # Filter for T1 first
        df_trends_t1 = df_trends[df_trends['Treatment'] == 1].copy()
        if not df_trends_t1.empty:
             try:
                # Use multi-index for clarity
                trends_pivot = df_trends_t1.set_index(['Spectral_Feature', 'Metabolite_Feature', 'Genotype', 'Day'])['Mean_Attention_S2M_Group_AvgHeads'].unstack(level=['Genotype', 'Day'])
                # Select columns for G1 and G2 at different days
                g1_d1_col = ('G1', 1.0)
                g1_d2_col = ('G1', 2.0)
                g1_d3_col = ('G1', 3.0)
                g2_d1_col = ('G2', 1.0)
                g2_d2_col = ('G2', 2.0)
                g2_d3_col = ('G2', 3.0)

                # Check if all necessary columns exist after pivoting
                required_pivot_cols = [g1_d1_col, g1_d2_col, g1_d3_col, g2_d1_col, g2_d2_col, g2_d3_col]
                if all(col in trends_pivot.columns for col in required_pivot_cols):
                    # Find pairs where G1 attention on Day 2 is significantly higher than Day 1 AND higher than G2 on Day 2
                    # Condition 1: G1 Day 2 substantially higher than G1 Day 1
                    cond1 = (trends_pivot[g1_d2_col] - trends_pivot[g1_d1_col]) > MIN_ATTN_DIFF_TEMPORAL
                    # Condition 2: G1 Day 2 is higher than G2 Day 2
                    cond2 = trends_pivot[g1_d2_col] > trends_pivot[g2_d2_col]
                    # Condition 3: G2 attention hasn't peaked by Day 2 (e.g., G2 Day 3 > G2 Day 2 or G2 Day 2 similar to Day 1)
                    cond3 = (trends_pivot[g2_d3_col] > trends_pivot[g2_d2_col]) | (trends_pivot[g2_d2_col] - trends_pivot[g2_d1_col] < MIN_ATTN_DIFF_TEMPORAL / 2) # G2 not strongly increasing by D2

                    potential_pairs = trends_pivot[cond1 & cond2 & cond3]
                    debug_print(f"    Found {len(potential_pairs)} potential pairs showing earlier G1 development based on criteria.")

                    # Get top examples based on G1 Day 2 attention
                    sorted_pairs = potential_pairs.sort_values(g1_d2_col, ascending=False)

                    for idx, row in sorted_pairs.head(TOP_N_HIGH_ATTN_EXAMPLES).iterrows():
                        pair_data = {
                            'spectral': idx[0], 'metabolite': idx[1],
                            'G1_attn': [row.get(g1_d1_col, np.nan), row.get(g1_d2_col, np.nan), row.get(g1_d3_col, np.nan)],
                            'G2_attn': [row.get(g2_d1_col, np.nan), row.get(g2_d2_col, np.nan), row.get(g2_d3_col, np.nan)],
                        }
                        early_dev_examples.append(pair_data)
                        debug_print(f"      Example found: {pair_data['spectral']} <-> {pair_data['metabolite']}")

                else:
                    debug_print(f"    WARNING: Could not find all required Day/Genotype columns after pivoting trends data. Columns: {trends_pivot.columns}")

             except Exception as e:
                print(f"  WARNING: Error pivoting or analyzing trends data for {tissue}: {e}")
                debug_print(f"    Trends T1 data head:\n{df_trends_t1.head()}")


        tissue_results['early_dev_examples'] = early_dev_examples

        all_results[tissue] = tissue_results # Store results for this tissue

    debug_print("\n--- Finished Conditional Attention Analysis ---")
    return all_results


def print_conditional_summary(data):
    """Prints the processed conditional attention summary data."""
    if not data:
        print("No conditional attention data processed.")
        return

    print("\n" + "="*80)
    print("Conditional Feature-Level Attention Analysis Summary")
    print("="*80)

    for tissue in ['Leaf', 'Root']:
        print(f"\n--- {tissue} Tissue ---")
        if tissue not in data:
            print("  No data processed for this tissue.")
            continue
        tissue_results = data[tissue]

        # a) Top Stress Pairs Examples
        print(f"\na) Top {TOP_N_HIGH_ATTN_EXAMPLES} Example S2M Attention Pairs under Stress (Overall T1, Day 3):")
        print(f"   (Full list of Top {TOP_N_CONDITIONAL_PAIRS} pairs saved to CSV)")
        top_stress_examples = tissue_results.get('top_stress_pair_examples', []) # Use the example list
        if top_stress_examples:
            for i, pair in enumerate(top_stress_examples):
                print(f"  {i+1}. {pair['spectral']} <-> {pair['metabolite']} (Mean Attn: {pair['mean_attn']:.4f})")
        else:
            print("  No top stress pairs found or error occurred.")

        # b) Proxy Hubs
        print("\nb) Proxy Hub Features (from Top Stress Pairs):")
        hubs = tissue_results.get('proxy_hubs', [])
        if hubs:
            print(f"  {', '.join(hubs)}")
        else:
            print("  No proxy hubs identified.")

        # c) G1 vs G2 Comparison
        print(f"\nc) G1 vs G2 Comparison (Avg Attn on Top {TOP_N_FOR_G1_G2_COMP} Overall Pairs, T1, Day 3):")
        comp = tissue_results.get('g1_g2_comparison', {})
        if 'error' in comp:
             print(f"  Error: {comp['error']}")
        elif 'avg_g1_attn' in comp:
             print(f"  Compared {comp['n_pairs_compared']} pairs:")
             print(f"    Avg G1 S2M Attention: {comp['avg_g1_attn']:.4f}")
             print(f"    Avg G2 S2M Attention: {comp['avg_g2_attn']:.4f}")
             print(f"    % Diff (G1 vs G2):   {comp['pct_diff_g1_vs_g2']:.1f}%")
        else:
            print("  Comparison data not available.")

        # d) Early Development Examples
        print("\nd) Examples of Earlier Attention Development in G1 (T1):")
        early_dev = tissue_results.get('early_dev_examples', [])
        if early_dev:
            for i, pair in enumerate(early_dev):
                print(f"  {i+1}. {pair['spectral']} <-> {pair['metabolite']}")
                print(f"     G1 Attn (D1,D2,D3): {pair['G1_attn'][0]:.4f}, {pair['G1_attn'][1]:.4f}, {pair['G1_attn'][2]:.4f}")
                print(f"     G2 Attn (D1,D2,D3): {pair['G2_attn'][0]:.4f}, {pair['G2_attn'][1]:.4f}, {pair['G2_attn'][2]:.4f}")
        else:
            print("  No clear examples found based on criteria.")

    print("\n" + "="*80)


# --- Function to Save Top Conditional Pairs to CSV ---
def save_top_conditional_pairs_csv(analysis_results, output_dir, filename, top_n):
    """
    Saves the top N conditional attention pairs for specified conditions
    to a CSV file.

    Args:
        analysis_results (dict): The nested dictionary from analyze_conditional_attention.
        output_dir (str): Directory to save the CSV file.
        filename (str): Name for the output CSV file.
        top_n (int): The number of top pairs to include per condition.
    """
    all_top_pairs = []
    output_path = os.path.join(output_dir, filename)

    debug_print(f"\n--- Saving Top {top_n} Conditional Pairs to CSV ---")
    debug_print(f"  Target Path: {output_path}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directory '{output_dir}': {e}")
        return

    for tissue in ['Leaf', 'Root']:
        if tissue not in analysis_results:
            debug_print(f"  Skipping {tissue}: No data found in analysis results.")
            continue
        tissue_data = analysis_results[tissue]
        debug_print(f"  Processing {tissue} data...")

        conditions = {
            'Overall_T1_D3': tissue_data.get('top_pairs_t1d3_overall', []),
            'G1_T1_D3': tissue_data.get('top_pairs_g1_t1d3', []),
            'G2_T1_D3': tissue_data.get('top_pairs_g2_t1d3', [])
        }

        for condition_name, pairs_list in conditions.items():
            if not pairs_list:
                 debug_print(f"    No pairs found for condition: {condition_name}")
                 continue

            debug_print(f"    Adding Top {min(top_n, len(pairs_list))} pairs for condition: {condition_name}")
            # The lists should already contain top N, but slice just in case
            for rank, pair_info in enumerate(pairs_list[:top_n]):
                all_top_pairs.append({
                    'Tissue': tissue,
                    'Condition': condition_name,
                    'Rank': rank + 1,
                    'Spectral_Feature': pair_info.get('spectral', 'N/A'),
                    'Metabolite_Feature': pair_info.get('metabolite', 'N/A'),
                    'Mean_Attention': pair_info.get('mean_attn', np.nan)
                })

    if not all_top_pairs:
        print("WARNING: No top conditional pairs found across all tissues/conditions to save.")
        debug_print("  No data appended to the CSV list.")
        return

    df_output = pd.DataFrame(all_top_pairs)
    df_output.sort_values(by=['Tissue', 'Condition', 'Rank'], inplace=True) # Sort for consistent output

    try:
        df_output.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\nSuccessfully saved top {top_n} conditional attention pairs to: {output_path}")
        debug_print(f"--- Finished Saving Top Conditional Pairs CSV ---")
    except Exception as e:
        print(f"\nERROR: Failed to save top conditional pairs CSV: {e}")
        print(f"  Attempted path: {output_path}")
        debug_print(f"  Error details: {e}")


# --- Run Analysis and Print ---
if __name__ == "__main__":
    analysis_results = analyze_conditional_attention(TRANSFORMER_OUTPUT_DIR, ATTENTION_SUBDIR)

    if analysis_results:
        print_conditional_summary(analysis_results)
        # Save the detailed top N pairs to CSV
        save_top_conditional_pairs_csv(
            analysis_results,
            OUTPUT_DIR_NOVILITY,
            OUTPUT_FILENAME_COND_ATTN,
            TOP_N_CONDITIONAL_PAIRS
        )
    else:
        print(f"\nERROR: Failed to perform conditional attention analysis. Check errors and debug output.")
        print(f"Ensure the directory structure under '{TRANSFORMER_OUTPUT_DIR}' is correct and contains the required 'processed_attention_*.csv' files.")