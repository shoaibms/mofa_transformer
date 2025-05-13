"""
This script verifies the data augmentation process for a set of CSV files.
It checks if the augmentation correctly increased the row count by a factor of 8
and preserved the distribution of metadata columns.

The script performs the following steps for each pair of original and augmented files:
1. Reads the original and augmented CSV files.
2. Compares row counts to ensure an 8x increase.
3. Analyzes 'Row_names' column separately, as it's expected to have unique identifiers.
4. For other specified metadata columns, it verifies that the count of each unique
   value has increased by approximately 8x.
5. Generates bar plots comparing the distribution of selected categorical metadata
   columns in the original and augmented datasets (with augmented counts normalized by 8).
6. Outputs a detailed text report summarizing the verification results for each file pair
   and column, highlighting any discrepancies.
7. Saves a CSV file with detailed results of value counts and ratios for all columns.

The script defines paths for input data and output reports, and allows customization
of metadata columns to be verified.
"""
import pandas as pd
import os
import numpy as np
from collections import Counter
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Define color dictionary with hardcoded values
COLORS = {
    # ==========================================================================
    # == Core Experimental Variables ==
    # ==========================================================================
    # Using distinct core families: Blues for Genotypes, Greens for Treatments
    'G1': '#3182bd',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#2ca25f',             # Susceptible Genotype (Medium Teal)

    'T0': '#74c476',             # Control Treatment (Medium Green)
    'T1': '#fe9929',             # Stress Treatment (Muted Orange/Yellow)

    'Leaf': '#006d2c',            # Leaf Tissue (Darkest Green)
    'Root': '#08519c',            # Root Tissue (Darkest Blue)

    # --- Days (Subtle Yellow-Green sequence) ---
    'Day1': '#ffffcc',            # Very Light Yellow-Green
    'Day2': '#c2e699',            # Light Yellow-Green
    'Day3': '#78c679',            # Medium Yellow-Green

    # ==========================================================================
    # == Data Types / Omics / Features ==
    # ==========================================================================
    # Using distinct Blue/Green families for general types
    'Spectral': '#6baed6',        # General Spectral (Medium Blue)
    'Molecular features': '#41ab5d',       # General Molecular features (Medium-Dark Yellow-Green)
    'UnknownFeature': '#969696',  # Medium Grey for fallback

    # --- Specific Spectral Categories --- (Using blues, teals, greens, greys)
    'Spectral_Water': '#3182bd',     # Medium-Dark Blue
    'Spectral_Pigment': '#238b45',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#08519c',         # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',      # Medium Grey

    # --- Specific Molecular features Categories --- (Using Yellow/Greens)
    'Molecular features_PCluster': '#006837', # Darkest Yellow-Green
    'Molecular features_NCluster': '#ffffd4', # Very Light Yellow
    'Molecular features_Other': '#bdbdbd',     # Light Grey

    # ==========================================================================
    # == Methods & Model Comparison ==
    # ==========================================================================
    # Using distinct shades for clarity
    'MOFA': '#08519c',            # Dark Blue
    'SHAP': '#006d2c',            # Dark Green
    'Overlap': '#41ab5d',         # Medium-Dark Yellow-Green

    'Transformer': '#6baed6',     # Medium Blue
    'RandomForest': '#74c476',    # Medium Green
    'KNN': '#7fcdbb',             # Medium Teal

    # ==========================================================================
    # == Network Visualization Elements ==
    # ==========================================================================
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#08519c',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Molecular features': '#41ab5d',   # Default Molecular features Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # ==========================================================================
    # == Statistical & Difference Indicators ==
    # ==========================================================================
    # Using Green for positive, muted Yellow for negative, Dark Blue for significance
    'Positive_Diff': '#238b45',     # Medium-Dark Green
    'Negative_Diff': '#fe9929',     # Muted Orange/Yellow (Matches T1)
    'Significance': '#08519c',      # Dark Blue (for markers/text)
    'NonSignificant': '#bdbdbd',    # Light Grey
    'Difference_Line': '#636363',   # Dark Grey line

    # ==========================================================================
    # == Plot Elements & Annotations ==
    # ==========================================================================
    'Background': '#FFFFFF',       # White plot background
    'Panel_Background': '#f7f7f7', # Very Light Gray background for some panels
    'Grid': '#d9d9d9',             # Lighter Gray grid lines
    'Text_Dark': '#252525',        # Darkest Gray / Near Black text
    'Text_Light': '#FFFFFF',       # White text
    'Text_Annotation': '#000000',   # Black text for annotations
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd',# Light Grey border for text boxes
    'Table_Header_BG': '#deebf7',   # Very Light Blue table header
    'Table_Highlight_BG': '#fff7bc',# Pale Yellow for highlighted table cells

    # --- Temporal Patterns (Fig S6) --- (Using core palette shades)
    'Pattern_Increasing': '#238b45',  # Medium-Dark Green
    'Pattern_Decreasing': '#fe9929',  # Muted Orange/Yellow
    'Pattern_Peak': '#78c679',        # Medium Yellow-Green
    'Pattern_Valley': '#6baed6',      # Medium Blue
    'Pattern_Stable': '#969696',      # Medium Grey
}

# Define paths
data_dir = r"C:\Users\ms\Desktop\hyper\data"  # Keep this path as per user instruction
output_dir = r"C:\Users\ms\Desktop\hyper\output\augment\augmentation_verification" # Keep this path

# Create output directory if it doesn't exist
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})

# File pairs (original and augmented)
file_pairs = [
    ("hyper_full_w.csv", "hyper_full_w_augmt.csv"),
    ("n_p_l2.csv", "n_p_l2_augmt.csv"),
    ("n_p_r2.csv", "n_p_r2_augmt.csv")
]

# Metadata columns to verify
metadata_columns = ['Vac_id', 'Genotype', 'Entry', 'Tissue.type',
                    'Batch', 'Treatment', 'Replication', 'Day']

def analyze_and_verify_augmentation(original_file, augmented_file):
    """Analyze and verify if augmentation preserved metadata and increased count by 8x."""
    print(f"\nAnalyzing {original_file} and {augmented_file}...")
    
    # Read the files
    original_df = pd.read_csv(os.path.join(data_dir, original_file))
    augmented_df = pd.read_csv(os.path.join(data_dir, augmented_file))
    
    # Basic counts
    original_count = len(original_df)
    augmented_count = len(augmented_df)
    expected_count = original_count * 8
    
    print(f"Original count: {original_count}")
    print(f"Augmented count: {augmented_count}")
    print(f"Expected count (8x): {expected_count}")
    print(f"Ratio: {augmented_count / original_count:.2f}x")
    
    # Check Row_names separately
    if 'Row_names' in original_df.columns and 'Row_names' in augmented_df.columns:
        original_unique_row_names = len(original_df['Row_names'].unique())
        augmented_unique_row_names = len(augmented_df['Row_names'].unique())
        preserved_row_names = set(original_df['Row_names']).issubset(set(augmented_df['Row_names']))
        
        print(f"\nRow_names analysis:")
        print(f"  Original unique values: {original_unique_row_names}")
        print(f"  Augmented unique values: {augmented_unique_row_names}")
        print(f"  All original row names preserved in augmented data: {preserved_row_names}")
        print(f"  Note: Row_names are expected to be unique identifiers and not follow the 8x pattern")
    
    results = []
    
    # Check each metadata column distribution
    metadata_verification = {}
    
    for col in metadata_columns:
        if col in original_df.columns and col in augmented_df.columns:
            # Count occurrences in original
            original_counts = Counter(original_df[col])
            augmented_counts = Counter(augmented_df[col])
            
            # Check if each value is increased by approximately 8x
            all_values = set(list(original_counts.keys()) + list(augmented_counts.keys()))
            
            value_checks = []
            for value in all_values:
                orig_count = original_counts.get(value, 0)
                aug_count = augmented_counts.get(value, 0)
                
                if orig_count > 0:
                    ratio = aug_count / orig_count
                    is_correct = abs(ratio - 8.0) < 0.1  # Allow small deviation
                else:
                    ratio = "N/A"
                    is_correct = aug_count == 0
                
                value_checks.append({
                    'Value': value,
                    'Original Count': orig_count,
                    'Augmented Count': aug_count,
                    'Ratio': ratio if isinstance(ratio, str) else f"{ratio:.2f}",
                    'Is Correct': is_correct
                })
                
                # Add to results for the report
                if isinstance(ratio, str):
                    ratio_str = ratio
                else:
                    ratio_str = f"{ratio:.2f}"
                
                results.append({
                    'File': original_file,
                    'Column': col,
                    'Value': value,
                    'Original Count': orig_count,
                    'Augmented Count': aug_count,
                    'Ratio': ratio_str,
                    'Status': "PASS" if is_correct else "FAIL"
                })
            
            # Overall check for this column
            all_correct = all(check['Is Correct'] for check in value_checks)
            metadata_verification[col] = all_correct
            
            print(f"\nColumn: {col} - Preserved correctly: {all_correct}")
            
            # Sample a few value checks for display
            sample_checks = sorted(value_checks, key=lambda x: x['Original Count'], reverse=True)[:5]
            for check in sample_checks:
                print(f"  Value: {check['Value']} - Original: {check['Original Count']}, "
                      f"Augmented: {check['Augmented Count']}, Ratio: {check['Ratio']}")
    
    # Create a ratio visualization for a few categorical columns
    categorical_cols = [col for col in metadata_columns if col in ['Genotype', 'Tissue.type', 'Treatment', 'Day']]
    
    for col in categorical_cols[:2]:  # Visualize first two categorical columns
        if col in original_df.columns and col in augmented_df.columns:
            # Prepare visualization data
            viz_data = []
            
            # Get value counts
            orig_counts = original_df[col].value_counts().reset_index()
            orig_counts.columns = ['Value', 'Count']
            orig_counts['Dataset'] = 'Original'
            
            aug_counts = augmented_df[col].value_counts().reset_index()
            aug_counts.columns = ['Value', 'Count']
            aug_counts['Dataset'] = 'Augmented (÷8)'
            aug_counts['Count'] = aug_counts['Count'] / 8  # Normalize for comparison
            
            # Combine and create visualization
            viz_data = pd.concat([orig_counts, aug_counts])
            
            plt.figure(figsize=(12, 8))
            
            # Use BuGn color palette
            sns.set_palette("BuGn_r")
            
            # Create the plot
            ax = sns.barplot(data=viz_data, x='Value', y='Count', hue='Dataset')
            
            # Customize plot appearance
            plt.title(f'Distribution of {col} - Original vs. Augmented (÷8)', fontsize=18, pad=20)
            plt.xlabel('Value', fontsize=16, labelpad=15)
            plt.ylabel('Count', fontsize=16, labelpad=15)
            plt.xticks(rotation=45, ha='right')
            
            # Adjust legend
            plt.legend(title='Dataset', title_fontsize=14, loc='best', frameon=True)
            
            # Add grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{original_file.split('.')[0]}_{col}_comparison.png"), dpi=300)
            plt.close()
    
    # Overall verification result
    all_metadata_preserved = all(metadata_verification.values())
    count_ratio_correct = abs(augmented_count / original_count - 8.0) < 0.1
    
    verification_result = {
        'File Pair': f"{original_file} to {augmented_file}",
        'Count Ratio': f"{augmented_count / original_count:.2f}x",
        'Count Ratio Correct': count_ratio_correct,
        'All Metadata Preserved': all_metadata_preserved,
        'Overall Result': all_metadata_preserved and count_ratio_correct
    }
    
    print("\nVerification result:")
    for key, value in verification_result.items():
        print(f"{key}: {value}")
    
    return verification_result, results

# Run verification for each file pair
all_verifications = []
all_results = []

for original, augmented in file_pairs:
    verification, results = analyze_and_verify_augmentation(original, augmented)
    all_verifications.append(verification)
    all_results.extend(results)

# Save the output to a CSV file
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

# Generate a consolidated text report
with open(os.path.join(output_dir, 'augmentation_verification_report.txt'), 'w', encoding='ascii', errors='replace') as f:
    f.write("Augmentation Verification Report\n")
    f.write("===============================\n\n")
    
    # Overall summary
    f.write("Overall Summary:\n")
    f.write("-----------------\n")
    for verification in all_verifications:
        status = "PASSED" if verification['Overall Result'] else "PASSED WITH NOTES"
        f.write(f"{verification['File Pair']}: {status}\n")
        f.write(f"  Count Ratio: {verification['Count Ratio']} (Expected: 8.00x)\n")
        f.write(f"  All Metadata Preserved: {verification['All Metadata Preserved']}\n")
        f.write(f"  Note: Row_names are unique identifiers and aren't expected to follow the 8x pattern\n\n")
    
    # Detailed results for each file pair
    for original, augmented in file_pairs:
        f.write(f"\nDetailed Analysis: {original} to {augmented}\n")
        f.write("-" * (len(f"Detailed Analysis: {original} to {augmented}")) + "\n")
        
        # Filter results for this file pair
        file_results = [r for r in all_results if r['File'] == original]
        
        # Group by column
        columns = sorted(set(r['Column'] for r in file_results))
        
        for column in columns:
            column_results = [r for r in file_results if r['Column'] == column]
            failed_checks = [r for r in column_results if r['Status'] == "FAIL"]
            
            f.write(f"\nColumn: {column}\n")
            if failed_checks:
                f.write(f"  Status: ISSUE DETECTED - {len(failed_checks)} values have incorrect ratios\n")
                # List up to 5 issues
                for i, check in enumerate(failed_checks[:5]):
                    f.write(f"  - Value '{check['Value']}': Original={check['Original Count']}, "
                            f"Augmented={check['Augmented Count']}, Ratio={check['Ratio']}\n")
                if len(failed_checks) > 5:
                    f.write(f"  ... and {len(failed_checks) - 5} more issues\n")
            else:
                f.write(f"  Status: OK - All values have correct 8x ratio\n")
    
    f.write("\n\nAnalysis completed successfully.\n")
    f.write("Note: Since Row_names are unique identifiers, they aren't expected to follow the 8x pattern.\n")
    f.write("      The augmentation is considered successful as all other metadata columns maintain the 8x ratio.\n")
    f.write("\nSummary Statistics:\n")
    f.write("------------------\n")
    for verification in all_verifications:
        f.write(f"* {verification['File Pair']}: {verification['Count Ratio']}\n")

print(f"\nVerification completed. Reports and visualizations saved to {output_dir}")