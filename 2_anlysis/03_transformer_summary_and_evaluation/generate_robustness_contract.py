"""
generate_robustness_contract.py
-------------------------------
1. Identifies the Genotype-associated Factor dynamically.
2. Performs a Sensitivity Sweep (Top 1%, 2.5%, 5%, 10%) to prove robustness.
3. Generates the 'Official' overlap list for Figure 6.
4. Saves everything to 'robustness_contract.json'.
"""

import pandas as pd
import numpy as np
import os
import json
import re

# =============================================================================
# PATHS - CONSISTENT WITH PROJECT STRUCTURE
# =============================================================================
BASE_DIR = r"C:/Users/ms/Desktop/hyper"
MOFA_DIR = os.path.join(BASE_DIR, "output", "mofa")
SHAP_DIR = os.path.join(BASE_DIR, "output", "transformer", "shap_analysis_ggl", "importance_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "robustness")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
PRIMARY_THRESHOLD_PCT = 0.05  # The threshold used for Figure 6 plotting (5%)
SWEEP_THRESHOLDS = [0.01, 0.025, 0.05, 0.10]  # For robustness table
SWEET_SPOT_RANGE = (546, 635)

def get_wavelength(feature_name):
    """Extracts integer wavelength from 'W_546'."""
    if not str(feature_name).startswith('W_'):
        return None
    match = re.search(r'W_(\d+)', str(feature_name))
    return int(match.group(1)) if match else None

def clean_feature_name(name):
    """Remove tissue/modality suffixes from feature names."""
    for suffix in ['_leaf_spectral', '_root_spectral', '_leaf_metabolite', '_root_metabolite']:
        name = str(name).replace(suffix, '')
    return name

def main():
    print("=== RUNNING ROBUSTNESS AUDIT ===")
    
    # 1. DYNAMIC FACTOR SELECTION
    assoc_path = os.path.join(MOFA_DIR, "mofa_factor_metadata_associations_spearman.csv")
    assoc_df = pd.read_csv(assoc_path)
    
    # Filter for Genotype (column is 'Correlation' not 'R')
    geno_rows = assoc_df[assoc_df['Metadata'] == 'Genotype'].copy()
    geno_rows['abs_R'] = geno_rows['Correlation'].abs()
    best_row = geno_rows.loc[geno_rows['abs_R'].idxmax()]
    best_factor = best_row['Factor']
    factor_idx = int(best_factor.replace('Factor', ''))
    
    print(f"✔ Factor Selection: {best_factor} (R = {best_row['Correlation']:.4f})")

    # 2. LOAD DATA
    # Load MOFA Weights (lowercase 'leaf', index_col=0)
    weights_path = os.path.join(MOFA_DIR, "mofa_feature_weights_leaf_spectral_active.csv")
    mofa_df = pd.read_csv(weights_path, index_col=0)
    mofa_df = mofa_df.reset_index()
    mofa_df.columns = ['Feature'] + list(mofa_df.columns[1:])
    mofa_df['Feature'] = mofa_df['Feature'].apply(clean_feature_name)
    mofa_df['abs_weight'] = mofa_df[best_factor].abs()
    
    # Load SHAP Importance (column is 'MeanAbsoluteShap')
    shap_path = os.path.join(SHAP_DIR, "shap_importance_Leaf_Genotype.csv")
    shap_df = pd.read_csv(shap_path)

    # 3. SENSITIVITY SWEEP
    sweep_results = []
    official_overlap_list = []
    
    print("\n✔ Sensitivity Sweep (546-635nm Check):")
    for thresh in SWEEP_THRESHOLDS:
        # Top N based on percentage
        n_mofa = max(1, int(len(mofa_df) * thresh))
        n_shap = max(1, int(len(shap_df) * thresh))
        
        top_mofa = set(mofa_df.nlargest(n_mofa, 'abs_weight')['Feature'])
        top_shap = set(shap_df.nlargest(n_shap, 'MeanAbsoluteShap')['Feature'])
        
        overlap = list(top_mofa.intersection(top_shap))
        jaccard = len(overlap) / len(top_mofa.union(top_shap)) if overlap else 0
        
        # Check Spectral Range
        wavelengths = [get_wavelength(f) for f in overlap if get_wavelength(f)]
        if wavelengths:
            in_range = [w for w in wavelengths if SWEET_SPOT_RANGE[0] <= w <= SWEET_SPOT_RANGE[1]]
            pct_in_range = (len(in_range) / len(wavelengths)) * 100
            min_w, max_w = min(wavelengths), max(wavelengths)
        else:
            pct_in_range = 0
            min_w, max_w = 0, 0
            
        print(f"  - Top {thresh*100:.1f}%: Jaccard={jaccard:.3f} | Range=[{min_w}-{max_w}nm] | In-Band={pct_in_range:.1f}%")
        
        sweep_results.append({
            "threshold_pct": thresh,
            "jaccard": round(jaccard, 4),
            "range_min": int(min_w),
            "range_max": int(max_w),
            "percent_in_band": round(pct_in_range, 1)
        })
        
        # Save the primary threshold list for Figure 6
        if thresh == PRIMARY_THRESHOLD_PCT:
            official_overlap_list = sorted(overlap)

    # 4. GENERATE CONTRACT JSON
    contract = {
        "metadata": {
            "generated_by": "generate_robustness_contract.py",
            "timestamp": pd.Timestamp.now().isoformat()
        },
        "selected_factor": {
            "name": best_factor,
            "index": factor_idx,
            "correlation": round(float(best_row['Correlation']), 4)
        },
        "primary_overlap": {
            "threshold_used": PRIMARY_THRESHOLD_PCT,
            "feature_list": official_overlap_list,
            "count": len(official_overlap_list)
        },
        "robustness_sweep": sweep_results
    }
    
    # Save
    contract_path = os.path.join(OUTPUT_DIR, "robustness_contract.json")
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=4)
        
    print(f"\n✔ Contract Saved: {contract_path}")
    print("  (Figure 6 will now read this file to ensure consistency)")

if __name__ == "__main__":
    main()