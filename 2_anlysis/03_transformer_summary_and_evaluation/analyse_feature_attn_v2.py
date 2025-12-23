import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# --- Configuration (hard-coded paths by design) ---
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

        # --- Nested helper for pair extraction ---
        def get_top_n_pairs(df, n):
            sorted_df = df.sort_values('Mean_Attention_S2M_Group_AvgHeads', ascending=False)
            top_pairs_list = []
            unique_pairs_seen = set()
            for _, row in sorted_df.iterrows():
                pair = (row['Spectral_Feature'], row['Metabolite_Feature'])
                if pair not in unique_pairs_seen:
                    top_pairs_list.append({
                        'spectral': row['Spectral_Feature'],
                        'metabolite': row['Metabolite_Feature'],
                        'mean_attn': row['Mean_Attention_S2M_Group_AvgHeads'],
                    })
                    unique_pairs_seen.add(pair)
                if len(top_pairs_list) >= n:
                    break
            return top_pairs_list

        tissue_results = {'daily': {}}

        # --- Loop through each day to perform day-specific analysis ---
        for day in [1, 2, 3]:
            day_results = {}
            debug_print(f"\n----- Analyzing Day {day} for {tissue} -----")

            # --- a) Top S2M attention links under key conditions (T1, specific day) ---
            debug_print(f"\n  a) Finding Top {TOP_N_CONDITIONAL_PAIRS} S2M Links (T1, Day {day})...")
            df_t1_d = df_cond[(df_cond['Treatment'] == 1) & (df_cond['Day'] == day)].copy()
            df_g1_t1d = df_cond[(df_cond['Genotype'] == 'G1') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == day)].copy()
            df_g2_t1d = df_cond[(df_cond['Genotype'] == 'G2') & (df_cond['Treatment'] == 1) & (df_cond['Day'] == day)].copy()

            top_pairs_t1d_overall = get_top_n_pairs(df_t1_d, TOP_N_CONDITIONAL_PAIRS)
            top_pairs_g1_t1d = get_top_n_pairs(df_g1_t1d, TOP_N_CONDITIONAL_PAIRS)
            top_pairs_g2_t1d = get_top_n_pairs(df_g2_t1d, TOP_N_CONDITIONAL_PAIRS)

            day_results['top_pairs_t1d_overall'] = top_pairs_t1d_overall
            day_results['top_pairs_g1_t1d'] = top_pairs_g1_t1d
            day_results['top_pairs_g2_t1d'] = top_pairs_g2_t1d
            debug_print(f"    Found Top {len(top_pairs_t1d_overall)} Overall T1D{day} pairs.")
            debug_print(f"    Found Top {len(top_pairs_g1_t1d)} G1 T1D{day} pairs.")
            debug_print(f"    Found Top {len(top_pairs_g2_t1d)} G2 T1D{day} pairs.")

            day_results['top_stress_pair_examples'] = top_pairs_t1d_overall[:TOP_N_HIGH_ATTN_EXAMPLES]

            # --- b) Hub features (Proxy: Features from top links on this day) ---
            debug_print(f"\n  b) Identifying Proxy Hub Features (from top T1D{day} links)...")
            hub_features = set()
            for pair_info in day_results.get('top_stress_pair_examples', []):
                hub_features.add(pair_info['spectral'])
                hub_features.add(pair_info['metabolite'])
            day_results['proxy_hubs'] = list(hub_features)
            debug_print(f"    Proxy hubs: {list(hub_features)}")

            # --- c) Quantitative comparison (G1 vs G2 under stress, this day) ---
            debug_print(f"\n  c) Comparing G1 vs G2 Mean Attention on Top {TOP_N_FOR_G1_G2_COMP} Overall Pairs (T1, Day {day})...")
            g1_attn_map = {(row['Spectral_Feature'], row['Metabolite_Feature']): row['Mean_Attention_S2M_Group_AvgHeads']
                           for _, row in df_g1_t1d.iterrows()}
            g2_attn_map = {(row['Spectral_Feature'], row['Metabolite_Feature']): row['Mean_Attention_S2M_Group_AvgHeads']
                           for _, row in df_g2_t1d.iterrows()}

            g1_means = []
            g2_means = []
            pairs_found_count = 0
            for pair in top_overall_pairs_set:
                g1_val = g1_attn_map.get(pair)
                g2_val = g2_attn_map.get(pair)
                if g1_val is not None and g2_val is not None:
                    g1_means.append(g1_val)
                    g2_means.append(g2_val)
                    pairs_found_count += 1

            if pairs_found_count > 0:
                avg_g1_attn = np.mean(g1_means)
                avg_g2_attn = np.mean(g2_means)
                fold_change = avg_g1_attn / avg_g2_attn if avg_g2_attn != 0 else np.inf
                pct_diff = ((avg_g1_attn - avg_g2_attn) / avg_g2_attn * 100) if avg_g2_attn != 0 else np.inf

                # Compute paired statistics (same feature-pairs in both groups)
                wilcoxon_p = np.nan
                mwu_p = np.nan
                
                if len(g1_means) >= 10:  # Wilcoxon needs reasonable n
                    try:
                        # Primary: Paired Wilcoxon signed-rank (matched by pair identity)
                        _, wilcoxon_p = wilcoxon(g1_means, g2_means, 
                                                  alternative='two-sided', 
                                                  zero_method='pratt')
                    except Exception as e:
                        debug_print(f"    Wilcoxon failed: {e}")
                    
                    try:
                        # Sensitivity: Unpaired Mann-Whitney U
                        _, mwu_p = mannwhitneyu(g1_means, g2_means, 
                                                 alternative='two-sided')
                    except Exception as e:
                        debug_print(f"    MWU failed: {e}")

                day_results['g1_g2_comparison'] = {
                    'n_pairs_compared': pairs_found_count,
                    'avg_g1_attn': avg_g1_attn,
                    'avg_g2_attn': avg_g2_attn,
                    'fold_change_g1_vs_g2': fold_change,
                    'pct_diff_g1_vs_g2': pct_diff,
                    'wilcoxon_p': wilcoxon_p,  # Primary (paired)
                    'mwu_p': mwu_p             # Sensitivity (unpaired)
                }
                
                debug_print(f"    Compared {pairs_found_count} pairs found in both G1/G2 under T1D{day}.")
                debug_print(f"    Avg Attn G1: {avg_g1_attn:.4f}, Avg Attn G2: {avg_g2_attn:.4f}")
                debug_print(f"    Fold Change (G1/G2): {fold_change:.2f}, Pct Diff: {pct_diff:.1f}%")
                debug_print(f"    Wilcoxon p: {wilcoxon_p:.2e}, MWU p: {mwu_p:.2e}")
            else:
                day_results['g1_g2_comparison'] = {'error': f"No overlapping pairs found between Top {TOP_N_FOR_G1_G2_COMP} overall and T1D{day} data."}
                debug_print(f"    ERROR: No common pairs found for T1D{day} comparison.")

            tissue_results['daily'][day] = day_results

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
        daily_data = tissue_results.get('daily', {})

        # --- Loop through each day's results ---
        for day in sorted(daily_data.keys()):
            day_results = daily_data[day]
            print(f"\n----------------- Day {day} Analysis -----------------")

            # a) Top Stress Pairs Examples
            print(f"\na) Top {TOP_N_HIGH_ATTN_EXAMPLES} Example S2M Attention Pairs under Stress (Overall T1, Day {day}):")
            print(f"   (Full list of Top {TOP_N_CONDITIONAL_PAIRS} pairs saved to CSV)")
            top_stress_examples = day_results.get('top_stress_pair_examples', [])
            if top_stress_examples:
                for i, pair in enumerate(top_stress_examples):
                    print(f"  {i+1}. {pair['spectral']} <-> {pair['metabolite']} (Mean Attn: {pair['mean_attn']:.4f})")
            else:
                print("  No top stress pairs found or error occurred for this day.")

            # b) Proxy Hubs
            print(f"\nb) Proxy Hub Features (from Top Stress Pairs on Day {day}):")
            hubs = day_results.get('proxy_hubs', [])
            if hubs:
                print(f"  {', '.join(hubs)}")
            else:
                print("  No proxy hubs identified for this day.")

            # c) G1 vs G2 Comparison
            print(f"\nc) G1 vs G2 Comparison (Avg Attn on Top {TOP_N_FOR_G1_G2_COMP} Overall Pairs, T1, Day {day}):")
            comp = day_results.get('g1_g2_comparison', {})
            if 'error' in comp:
                print(f"  Error: {comp['error']}")
            elif 'avg_g1_attn' in comp:
                print(f"  Compared {comp['n_pairs_compared']} pairs:")
                print(f"    Avg G1 S2M Attention: {comp['avg_g1_attn']:.4f}")
                print(f"    Avg G2 S2M Attention: {comp['avg_g2_attn']:.4f}")
                print(f"    % Diff (G1 vs G2):   {comp['pct_diff_g1_vs_g2']:.1f}%")
            else:
                print("  Comparison data not available for this day.")

        # --- Print temporal analysis once per tissue ---
        print("\n----------------- Temporal Analysis -----------------")
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

        daily_data = tissue_data.get('daily', {})
        if not daily_data:
            debug_print(f"    No daily data found for {tissue}.")
            continue

        for day, day_results in daily_data.items():
            conditions = {
                f'Overall_T1_D{day}': day_results.get('top_pairs_t1d_overall', []),
                f'G1_T1_D{day}': day_results.get('top_pairs_g1_t1d', []),
                f'G2_T1_D{day}': day_results.get('top_pairs_g2_t1d', [])
            }

            for condition_name, pairs_list in conditions.items():
                if not pairs_list:
                    debug_print(f"    No pairs found for condition: {condition_name}")
                    continue

                debug_print(f"    Adding Top {min(top_n, len(pairs_list))} pairs for condition: {condition_name}")
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


def compile_coord_strength_statistics(all_results, output_dir):
    """
    Compile coordination strength statistics from analysis results and apply FDR.
    
    Uses paired Wilcoxon as primary test (feature-pairs are matched units).
    MWU provided as sensitivity check.
    
    Note: Top-100 pairs share features, so these tests are supportive rather 
    than fully independent-sample inference. This is acknowledged in Methods.
    """
    print("\n" + "="*70)
    print("Coordination Strength (Top-100 Pairs) - Statistical Summary")
    print("="*70)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    stats_results = []
    
    for tissue in ['Leaf', 'Root']:
        if tissue not in all_results:
            continue
        tissue_data = all_results[tissue]
        daily_data = tissue_data.get('daily', {})
        
        for day in sorted(daily_data.keys()):
            day_results = daily_data[day]
            comp = day_results.get('g1_g2_comparison', {})
            
            if 'error' in comp or 'avg_g1_attn' not in comp:
                continue
            
            stats_results.append({
                'Tissue': tissue,
                'Day': day,
                'Condition': 'T1',
                'N_Pairs': comp.get('n_pairs_compared', 0),
                'G1_Mean_Attn': comp.get('avg_g1_attn', np.nan),
                'G2_Mean_Attn': comp.get('avg_g2_attn', np.nan),
                'Fold_Change': comp.get('fold_change_g1_vs_g2', np.nan),
                'Pct_Diff': comp.get('pct_diff_g1_vs_g2', np.nan),
                'Wilcoxon_P': comp.get('wilcoxon_p', np.nan),
                'MWU_P': comp.get('mwu_p', np.nan)
            })
    
    if not stats_results:
        print("  No results to compile.")
        return None
    
    stats_df = pd.DataFrame(stats_results)
    
    # Apply FDR correction to primary (Wilcoxon) p-values
    valid_wilcoxon = stats_df['Wilcoxon_P'].dropna()
    if len(valid_wilcoxon) > 0:
        reject, pvals_fdr, _, _ = multipletests(
            valid_wilcoxon.values, alpha=0.05, method='fdr_bh'
        )
        stats_df.loc[valid_wilcoxon.index, 'Wilcoxon_FDR'] = pvals_fdr
        stats_df.loc[valid_wilcoxon.index, 'Significant_FDR'] = reject
    else:
        stats_df['Wilcoxon_FDR'] = np.nan
        stats_df['Significant_FDR'] = False
    
    # Also FDR-correct MWU for comparison
    valid_mwu = stats_df['MWU_P'].dropna()
    if len(valid_mwu) > 0:
        _, mwu_fdr, _, _ = multipletests(valid_mwu.values, alpha=0.05, method='fdr_bh')
        stats_df.loc[valid_mwu.index, 'MWU_FDR'] = mwu_fdr
    else:
        stats_df['MWU_FDR'] = np.nan
    
    # Reorder columns for clarity
    col_order = ['Tissue', 'Day', 'Condition', 'N_Pairs', 
                 'G1_Mean_Attn', 'G2_Mean_Attn', 'Fold_Change', 'Pct_Diff',
                 'Wilcoxon_P', 'Wilcoxon_FDR', 'Significant_FDR',
                 'MWU_P', 'MWU_FDR']
    stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
    
    # Sort by tissue then day
    stats_df = stats_df.sort_values(['Tissue', 'Day'])
    
    # Save to CSV
    outfile = os.path.join(output_dir, "coord_strength_top100_stats.csv")
    stats_df.to_csv(outfile, index=False, float_format='%.6e')
    print(f"\nSaved to: {outfile}")
    
    # Print formatted summary
    print("\nResults (Primary: Paired Wilcoxon | Sensitivity: MWU):")
    print("-"*70)
    for _, row in stats_df.iterrows():
        sig = "SIG" if row.get('Significant_FDR', False) else ""
        print(f"  {row['Tissue']:5} Day {row['Day']}: "
              f"Fold={row['Fold_Change']:5.2f}x  "
              f"Wilcoxon p={row['Wilcoxon_P']:.2e}  "
              f"FDR={row.get('Wilcoxon_FDR', np.nan):.2e}  {sig}")
    print("-"*70)
    
    # Identify the key result for Abstract
    leaf_d3 = stats_df[(stats_df['Tissue'] == 'Leaf') & (stats_df['Day'] == 3)]
    if not leaf_d3.empty:
        row = leaf_d3.iloc[0]
        print(f"\n>>> KEY RESULT FOR ABSTRACT (Leaf, Peak Stress Day 3):")
        print(f"    Fold-change: {row['Fold_Change']:.2f}x")
        print(f"    Wilcoxon FDR: {row.get('Wilcoxon_FDR', np.nan):.2e}")
        if row.get('Significant_FDR', False):
            print(f"    STATUS: SIGNIFICANT (FDR < 0.05)")
        else:
            print(f"    STATUS: Not significant at FDR < 0.05")
    
    return stats_df


# --- Run Analysis and Print ---
if __name__ == "__main__":
    analysis_results = analyze_conditional_attention(TRANSFORMER_OUTPUT_DIR, ATTENTION_SUBDIR)

    if analysis_results:
        # Compile and save coordination strength statistics with FDR
        coord_stats_df = compile_coord_strength_statistics(analysis_results, OUTPUT_DIR_NOVILITY)
        
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