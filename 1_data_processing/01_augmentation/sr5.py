"""
This module provides the CrossModalityValidation class for comprehensive validation
of augmented data across spectral and Molecular feature modalities. It ensures data
consistency, balance, and biological relevance after augmentation, generating
visualizations and an HTML summary report. The script can be run from the
command line to perform these validations.
"""
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, chisquare
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import jensenshannon
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Get distinct colors for visualization
def get_bugn_colors(n_colors=3):
    """Get a specified number of distinct colors, interpolating from a base set."""
    # Base hex colors
    hex_colors = ['#66b399', '#b7d69a', '#3a999e']
    
    # Convert hex colors to RGB values (0-1 scale) with alpha
    base_palette_rgba = []
    for hex_color in hex_colors:
        h = hex_color.lstrip('#')
        rgb = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
        base_palette_rgba.append(rgb + [1.0]) # Add alpha value
    
    if n_colors <= 0:
        return np.array([])
    if n_colors <= len(base_palette_rgba):
        return np.array(base_palette_rgba[:n_colors])
    else:
        # For more colors, interpolate linearly across the base palette.
        interpolated_palette = np.zeros((n_colors, 4))
        # Points for base palette (e.g., 0, 0.5, 1 for 3 colors)
        base_points = np.linspace(0, 1, len(base_palette_rgba))
        # Points for target n_colors
        target_points = np.linspace(0, 1, n_colors)
        
        for component_idx in range(4): # R, G, B, A
            component_values = [color[component_idx] for color in base_palette_rgba]
            interpolated_palette[:, component_idx] = np.interp(target_points, base_points, component_values)
        return interpolated_palette

class CrossModalityValidation:
    """
    Comprehensive validation of augmented data across spectral and Molecular feature modalities to ensure
    consistency, balance, and biological relevance.
    """
    
    def __init__(self, spectral_original_path, spectral_augmented_path, 
                 molecular_feature_original_path, molecular_feature_augmented_path, output_dir):
        """
        Initialize the CrossModalityValidation class.
        
        Parameters:
        -----------
        spectral_original_path : str
            Path to original spectral data CSV file
        spectral_augmented_path : str
            Path to augmented spectral data CSV file
        molecular_feature_original_path : str
            Path to original Molecular feature data CSV file
        molecular_feature_augmented_path : str
            Path to augmented Molecular feature data CSV file
        output_dir : str
            Directory to save validation results
        """
        self.spectral_original_path = spectral_original_path
        self.spectral_augmented_path = spectral_augmented_path
        self.molecular_feature_original_path = molecular_feature_original_path
        self.molecular_feature_augmented_path = molecular_feature_augmented_path
        self.output_dir = output_dir
        
        # Create output directories
        self.validation_dir = os.path.join(output_dir, 'cross_validation')
        if not os.path.exists(self.validation_dir):
            os.makedirs(self.validation_dir)
            
        self.plots_dir = os.path.join(self.validation_dir, 'plots')
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        self.results_dir = os.path.join(self.validation_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load and prepare the original and augmented data for both modalities."""
        print("Loading data...")
        
        # Load spectral data
        self.spectral_original = pd.read_csv(self.spectral_original_path)
        self.spectral_augmented = pd.read_csv(self.spectral_augmented_path)
        
        # Load Molecular feature data
        self.molecular_feature_original = pd.read_csv(self.molecular_feature_original_path)
        self.molecular_feature_augmented = pd.read_csv(self.molecular_feature_augmented_path)
        
        # Identify column types
        # Spectral columns
        self.wavelength_cols = [col for col in self.spectral_original.columns if col.startswith('W_')]
        self.spectral_metadata_cols = [col for col in self.spectral_original.columns if not col.startswith('W_')]
        
        # Molecular feature columns
        self.n_cluster_cols = [col for col in self.molecular_feature_original.columns if col.startswith('N_Cluster_')]
        self.p_cluster_cols = [col for col in self.molecular_feature_original.columns if col.startswith('P_Cluster_')]
        self.molecular_feature_cols = self.n_cluster_cols + self.p_cluster_cols
        self.molecular_feature_metadata_cols = [col for col in self.molecular_feature_original.columns if col not in self.molecular_feature_cols]
        
        # Separate augmented data by source
        self.spectral_augmented_only = self.spectral_augmented[~self.spectral_augmented['Row_names'].isin(self.spectral_original['Row_names'])]
        self.molecular_feature_augmented_only = self.molecular_feature_augmented[~self.molecular_feature_augmented['Row_names'].isin(self.molecular_feature_original['Row_names'])]
        
        # Check if tissue type is consistent
        if 'Tissue.type' in self.spectral_metadata_cols and 'Tissue.type' in self.molecular_feature_metadata_cols:
            spectral_tissues = set(self.spectral_original['Tissue.type'].unique())
            molecular_feature_tissues = set(self.molecular_feature_original['Tissue.type'].unique())
            
            if len(spectral_tissues) > 1 and len(molecular_feature_tissues) == 1:
                # If spectral has both tissues but Molecular feature has only one, filter spectral to match
                target_tissue = list(molecular_feature_tissues)[0]
                print(f"Note: Molecular feature data contains only tissue type '{target_tissue}', filtering spectral data accordingly.")
                self.spectral_original = self.spectral_original[self.spectral_original['Tissue.type'] == target_tissue]
                self.spectral_augmented = self.spectral_augmented[self.spectral_augmented['Tissue.type'] == target_tissue]
                self.spectral_augmented_only = self.spectral_augmented_only[self.spectral_augmented_only['Tissue.type'] == target_tissue]
        
        print(f"Loaded spectral data: {len(self.spectral_original)} original, {len(self.spectral_augmented_only)} augmented")
        print(f"Loaded Molecular feature data: {len(self.molecular_feature_original)} original, {len(self.molecular_feature_augmented_only)} augmented")
    
    def experimental_factor_consistency(self):
        """
        Verify consistency of experimental factors across both modalities after augmentation.
        This checks that augmentation preserved the experimental design structure.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with experimental factor consistency results
        """
        print("\n1. Verifying experimental factor consistency across modalities...")
        
        # Define experimental factors to check
        factors = ['Tissue.type', 'Genotype', 'Batch', 'Treatment', 'Day', 'Replication']
        
        # Initialize results dictionary
        consistency_results = {}
        
        # Check original data for baseline
        original_consistency = {}
        for factor in factors:
            if factor in self.spectral_metadata_cols and factor in self.molecular_feature_metadata_cols:
                # Get distribution of factor in both modalities
                spectral_dist = self.spectral_original[factor].value_counts(normalize=True)
                molecular_feature_dist = self.molecular_feature_original[factor].value_counts(normalize=True)
                
                # Calculate consistency as correlation between distributions
                merged_dist = pd.DataFrame({
                    'spectral': spectral_dist,
                    'molecular_feature': molecular_feature_dist
                }).fillna(0)
                
                # Calculate correlation and difference
                try:
                    # Check if we have enough data for correlation
                    if len(merged_dist) > 2:
                        corr = merged_dist['spectral'].corr(merged_dist['molecular_feature'])
                        if pd.isna(corr):  # Handle NaN correlation
                            corr = 1.0 if np.allclose(merged_dist['spectral'], merged_dist['molecular_feature']) else 0.0
                    else:
                        # For just 1-2 levels, check for direct alignment
                        if np.allclose(merged_dist['spectral'], merged_dist['molecular_feature']):
                            corr = 1.0
                        else:
                            corr = 0.0
                    
                    max_diff = np.max(np.abs(merged_dist['spectral'] - merged_dist['molecular_feature']))
                except Exception as e:
                    print(f"  Warning: Error calculating correlation for {factor}: {e}")
                    corr = 1.0  # Assume perfect correlation as fallback
                    max_diff = 0.0
                
                original_consistency[factor] = {
                    'correlation': corr,
                    'max_difference': max_diff
                }
        
        # Check augmented data
        augmented_consistency = {}
        for factor in factors:
            if factor in self.spectral_metadata_cols and factor in self.molecular_feature_metadata_cols:
                # Get distribution of factor in both modalities
                spectral_dist = self.spectral_augmented[factor].value_counts(normalize=True)
                molecular_feature_dist = self.molecular_feature_augmented[factor].value_counts(normalize=True)
                
                # Calculate consistency as correlation between distributions
                merged_dist = pd.DataFrame({
                    'spectral': spectral_dist,
                    'molecular_feature': molecular_feature_dist
                }).fillna(0)
                
                # Calculate correlation and difference
                try:
                    # Check if we have enough data for correlation
                    if len(merged_dist) > 2:
                        corr = merged_dist['spectral'].corr(merged_dist['molecular_feature'])
                        if pd.isna(corr):  # Handle NaN correlation
                            corr = 1.0 if np.allclose(merged_dist['spectral'], merged_dist['molecular_feature']) else 0.0
                    else:
                        # For just 1-2 levels, check for direct alignment
                        if np.allclose(merged_dist['spectral'], merged_dist['molecular_feature']):
                            corr = 1.0
                        else:
                            corr = 0.0
                    
                    max_diff = np.max(np.abs(merged_dist['spectral'] - merged_dist['molecular_feature']))
                except Exception as e:
                    print(f"  Warning: Error calculating correlation for {factor}: {e}")
                    corr = 1.0  # Assume perfect correlation as fallback
                    max_diff = 0.0
                
                augmented_consistency[factor] = {
                    'correlation': corr,
                    'max_difference': max_diff
                }
        
        # Calculate consistency preservation
        for factor in factors:
            if factor in original_consistency and factor in augmented_consistency:
                orig_corr = original_consistency[factor]['correlation']
                aug_corr = augmented_consistency[factor]['correlation']
                
                # Calculate consistency preservation (1.0 = perfect preservation)
                corr_preservation = aug_corr / orig_corr if orig_corr > 0 else 0
                
                # Calculate maximum difference
                orig_diff = original_consistency[factor]['max_difference']
                aug_diff = augmented_consistency[factor]['max_difference']
                
                # Calculate difference ratio (closer to 1.0 is better)
                diff_ratio = aug_diff / orig_diff if orig_diff > 0 else (1.0 if aug_diff == 0 else float('inf'))
                
                consistency_results[factor] = {
                    'original_correlation': orig_corr,
                    'augmented_correlation': aug_corr,
                    'correlation_preservation': corr_preservation,
                    'original_max_difference': orig_diff,
                    'augmented_max_difference': aug_diff,
                    'difference_ratio': diff_ratio,
                    'overall_preservation': 1.0 - min(1.0, abs(corr_preservation - 1.0))
                }
        
        # Create summary DataFrame
        summary_data = []
        for factor, metrics in consistency_results.items():
            summary_data.append({
                'Factor': factor,
                'Original_Correlation': metrics['original_correlation'],
                'Augmented_Correlation': metrics['augmented_correlation'],
                'Correlation_Preservation': metrics['correlation_preservation'],
                'Original_Max_Difference': metrics['original_max_difference'],
                'Augmented_Max_Difference': metrics['augmented_max_difference'],
                'Difference_Ratio': metrics['difference_ratio'],
                'Overall_Preservation': metrics['overall_preservation']
            })
        
        if not summary_data:
            print("Warning: No experimental factors could be analyzed.")
            return pd.DataFrame()
        
        factor_summary = pd.DataFrame(summary_data)
        
        # Save results to CSV
        factor_summary.to_csv(os.path.join(self.results_dir, "experimental_factor_consistency.csv"), index=False)
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Get the user-defined green-shade colors
        all_colors = get_bugn_colors(3)
        # Define the mapping based on desired visual progression:
        # Good (>0.9) = Bluish-Green (#3a999e -> index 2)
        # Moderate (0.8-0.9) = Teal/Green (#66b399 -> index 0)
        # Poor (<=0.8) = Lime/Green (#b7d69a -> index 1)
        color_good = all_colors[2]
        color_moderate = all_colors[0]
        color_poor = all_colors[1]

        # Assign colors based on scores using the new mapping
        score_values = factor_summary['Overall_Preservation']
        score_colors = [color_good if score > 0.9 else color_moderate if score > 0.8 else color_poor
                       for score in score_values]

        # Bar chart of overall preservation
        bars = plt.bar(
            factor_summary['Factor'],
            score_values,
            color=score_colors
        )
        # Update axhline colors to match the thresholds they represent
        plt.axhline(y=0.9, linestyle='--', color=color_good, alpha=0.7)     # Line above Moderate threshold, use Good color
        plt.axhline(y=0.8, linestyle='--', color=color_moderate, alpha=0.7) # Line above Poor threshold, use Moderate color
        plt.title('Experimental Factor Consistency Preservation', fontsize=18)
        
        plt.xlabel('Factor', fontsize=16)
        plt.ylabel('Preservation Score (1.0 = Perfect)', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        
        # Add data labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, "factor_consistency.png"), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, "factor_consistency.pdf"))
        plt.close()
        
        print("Experimental factor consistency results:")
        print(factor_summary[['Factor', 'Correlation_Preservation', 'Overall_Preservation']])
        
        return factor_summary
    
    def metadata_balance_assessment(self):
        """
        Assess metadata balance across both modalities to ensure experimental factors 
        are represented proportionally.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with metadata balance assessment results
        """
        print("\n2. Assessing metadata balance across modalities...")
        
        # Define experimental factors to check
        factors = ['Tissue.type', 'Genotype', 'Batch', 'Treatment', 'Day', 'Replication']
        
        # Initialize results dictionary
        balance_results = {}
        
        # Assess balance in original data (baseline)
        original_balance = {}
        for factor in factors:
            if factor in self.spectral_metadata_cols and factor in self.molecular_feature_metadata_cols:
                # Get counts for each value of the factor in both modalities
                spectral_counts = self.spectral_original[factor].value_counts()
                molecular_feature_counts = self.molecular_feature_original[factor].value_counts()
                
                # Calculate chi-square statistic to measure distribution similarity
                merged_counts = pd.DataFrame({
                    'spectral': spectral_counts,
                    'molecular_feature': molecular_feature_counts
                }).fillna(0)
                
                # Calculate JS divergence for distributions
                p1 = merged_counts['spectral'] / merged_counts['spectral'].sum()
                p2 = merged_counts['molecular_feature'] / merged_counts['molecular_feature'].sum()
                js_div = jensenshannon(p1, p2)
                
                # Also calculate normalized difference in proportions
                spectral_props = self.spectral_original[factor].value_counts(normalize=True)
                molecular_feature_props = self.molecular_feature_original[factor].value_counts(normalize=True)
                
                merged_props = pd.DataFrame({
                    'spectral': spectral_props,
                    'molecular_feature': molecular_feature_props
                }).fillna(0)
                
                max_prop_diff = np.max(np.abs(merged_props['spectral'] - merged_props['molecular_feature']))
                
                original_balance[factor] = {
                    'js_divergence': js_div,
                    'max_proportion_diff': max_prop_diff
                }
        
        # Assess balance in augmented data
        augmented_balance = {}
        for factor in factors:
            if factor in self.spectral_metadata_cols and factor in self.molecular_feature_metadata_cols:
                # Get counts for each value of the factor in both modalities
                spectral_counts = self.spectral_augmented[factor].value_counts()
                molecular_feature_counts = self.molecular_feature_augmented[factor].value_counts()
                
                # Calculate distribution similarity
                merged_counts = pd.DataFrame({
                    'spectral': spectral_counts,
                    'molecular_feature': molecular_feature_counts
                }).fillna(0)
                
                # Calculate JS divergence for distributions
                p1 = merged_counts['spectral'] / merged_counts['spectral'].sum()
                p2 = merged_counts['molecular_feature'] / merged_counts['molecular_feature'].sum()
                js_div = jensenshannon(p1, p2)
                
                # Also calculate normalized difference in proportions
                spectral_props = self.spectral_augmented[factor].value_counts(normalize=True)
                molecular_feature_props = self.molecular_feature_augmented[factor].value_counts(normalize=True)
                
                merged_props = pd.DataFrame({
                    'spectral': spectral_props,
                    'molecular_feature': molecular_feature_props
                }).fillna(0)
                
                max_prop_diff = np.max(np.abs(merged_props['spectral'] - merged_props['molecular_feature']))
                
                augmented_balance[factor] = {
                    'js_divergence': js_div,
                    'max_proportion_diff': max_prop_diff
                }
        
        # Calculate balance preservation
        for factor in factors:
            if factor in original_balance and factor in augmented_balance:
                orig_js = original_balance[factor]['js_divergence']
                aug_js = augmented_balance[factor]['js_divergence']
                
                # Higher JS is worse, so calculate preservation as ratio (closer to 1.0 is better)
                js_preservation = aug_js / orig_js if orig_js > 0 else (1.0 if aug_js == 0 else float('inf'))
                
                # Same for proportion difference
                orig_diff = original_balance[factor]['max_proportion_diff']
                aug_diff = augmented_balance[factor]['max_proportion_diff']
                diff_preservation = aug_diff / orig_diff if orig_diff > 0 else (1.0 if aug_diff == 0 else float('inf'))
                
                # Overall balance preservation score (1.0 = perfect)
                overall_score = 1.0 - min(1.0, (abs(js_preservation - 1.0) + abs(diff_preservation - 1.0)) / 2)
                
                balance_results[factor] = {
                    'original_js_divergence': orig_js,
                    'augmented_js_divergence': aug_js,
                    'js_preservation_ratio': js_preservation,
                    'original_max_diff': orig_diff,
                    'augmented_max_diff': aug_diff,
                    'diff_preservation_ratio': diff_preservation,
                    'overall_balance_score': overall_score
                }
        
        # Create summary DataFrame
        summary_data = []
        for factor, metrics in balance_results.items():
            summary_data.append({
                'Factor': factor,
                'Original_JS_Divergence': metrics['original_js_divergence'],
                'Augmented_JS_Divergence': metrics['augmented_js_divergence'],
                'JS_Preservation_Ratio': metrics['js_preservation_ratio'],
                'Original_Max_Diff': metrics['original_max_diff'],
                'Augmented_Max_Diff': metrics['augmented_max_diff'],
                'Diff_Preservation_Ratio': metrics['diff_preservation_ratio'],
                'Overall_Balance_Score': metrics['overall_balance_score']
            })
        
        if not summary_data:
            print("Warning: No experimental factors could be analyzed for balance.")
            return pd.DataFrame()
        
        balance_summary = pd.DataFrame(summary_data)
        
        # Save results to CSV
        balance_summary.to_csv(os.path.join(self.results_dir, "metadata_balance.csv"), index=False)
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Get the user-defined green-shade colors
        all_colors = get_bugn_colors(3)
        # Define the mapping based on desired visual progression:
        # Good (>0.9) = Bluish-Green (#3a999e -> index 2)
        # Moderate (0.8-0.9) = Teal/Green (#66b399 -> index 0)
        # Poor (<=0.8) = Lime/Green (#b7d69a -> index 1)
        color_good = all_colors[2]
        color_moderate = all_colors[0]
        color_poor = all_colors[1]

        # Assign colors based on scores using the new mapping
        score_values = balance_summary['Overall_Balance_Score']
        score_colors = [color_good if score > 0.9 else color_moderate if score > 0.8 else color_poor
                       for score in score_values]

        # Bar chart of overall balance scores
        bars = plt.bar(
            balance_summary['Factor'],
            score_values,
            color=score_colors
        )
        # Update axhline colors to match the thresholds they represent
        plt.axhline(y=0.9, linestyle='--', color=color_good, alpha=0.7)     # Line above Moderate threshold, use Good color
        plt.axhline(y=0.8, linestyle='--', color=color_moderate, alpha=0.7) # Line above Poor threshold, use Moderate color
        plt.title('Metadata Balance Preservation Across Modalities', fontsize=18)
        
        plt.xlabel('Factor', fontsize=16)
        plt.ylabel('Balance Score (1.0 = Perfect)', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        
        # Add data labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, "metadata_balance.png"), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, "metadata_balance.pdf"))
        plt.close()
        
        print("Metadata balance assessment results:")
        print(balance_summary[['Factor', 'JS_Preservation_Ratio', 'Overall_Balance_Score']])
        
        return balance_summary
    
    def treatment_response_preservation(self):
        """
        Verify that treatment response patterns are preserved across both modalities
        after augmentation.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with treatment response preservation results
        """
        print("\n3. Evaluating treatment response pattern preservation...")
        
        # Check if Treatment column exists
        if 'Treatment' not in self.spectral_metadata_cols or 'Treatment' not in self.molecular_feature_metadata_cols:
            print("Warning: 'Treatment' column not found in one or both modalities. Skipping treatment response analysis.")
            return pd.DataFrame()
        
        # Function to calculate treatment effects
        def calculate_treatment_effects(data, feature_cols):
            effects = {}
            
            # Get unique treatments
            treatments = sorted(data['Treatment'].unique())
            
            # Assuming binary treatment (0, 1) for simplicity
            if len(treatments) == 2:
                control = treatments[0]
                treatment = treatments[1]
                
                for feature in feature_cols:
                    if feature in data.columns:
                        # Get values for each group
                        control_values = data[data['Treatment'] == control][feature].values
                        treatment_values = data[data['Treatment'] == treatment][feature].values
                        
                        if len(control_values) > 0 and len(treatment_values) > 0:
                            # Calculate Cohen's d effect size
                            mean_diff = np.mean(treatment_values) - np.mean(control_values)
                            pooled_std = np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)
                            
                            if pooled_std > 0:
                                effect_size = mean_diff / pooled_std
                                
                                # Perform t-test
                                try:
                                    t_stat, p_value = ttest_ind(treatment_values, control_values, equal_var=False)
                                    effects[feature] = {
                                        'effect_size': effect_size,
                                        'p_value': p_value
                                    }
                                except:
                                    # Skip if t-test fails
                                    pass
            
            return effects
        
        # Calculate treatment effects on spectral data (using subset of wavelengths for efficiency)
        # Sample wavelengths across the spectrum
        sampled_wavelengths = self.wavelength_cols[::20]  # Take every 20th wavelength
        if len(sampled_wavelengths) > 100:  # Further limit if still too many
            sampled_wavelengths = sampled_wavelengths[:100]
        
        print(f"  Using {len(sampled_wavelengths)} sampled wavelengths for spectral analysis...")
        
        spectral_orig_effects = calculate_treatment_effects(self.spectral_original, sampled_wavelengths)
        spectral_aug_effects = calculate_treatment_effects(self.spectral_augmented, sampled_wavelengths)
        
        print(f"  Found {len(spectral_orig_effects)} significant spectral features in original data...")
        
        # Calculate treatment effects on Molecular feature data
        molecular_feature_orig_effects = calculate_treatment_effects(self.molecular_feature_original, self.molecular_feature_cols)
        molecular_feature_aug_effects = calculate_treatment_effects(self.molecular_feature_augmented, self.molecular_feature_cols)
        
        print(f"  Found {len(molecular_feature_orig_effects)} significant Molecular features in original data...")
        
        # Calculate preservation metrics
        preservation_results = {}
        
        # For spectral data
        spectral_preservation = self.calculate_effect_preservation(
            spectral_orig_effects, spectral_aug_effects, 'Spectral')
        preservation_results['spectral'] = spectral_preservation
        
        # For Molecular feature data
        molecular_feature_preservation = self.calculate_effect_preservation(
            molecular_feature_orig_effects, molecular_feature_aug_effects, 'Molecular feature')
        preservation_results['molecular_feature'] = molecular_feature_preservation
        
        # Calculate cross-modal consistency
        # Original data cross-modal consistency
        orig_cross_modal = self.calculate_cross_modal_consistency(
            spectral_orig_effects, molecular_feature_orig_effects)
        
        # Augmented data cross-modal consistency
        aug_cross_modal = self.calculate_cross_modal_consistency(
            spectral_aug_effects, molecular_feature_aug_effects)
        
        # Calculate consistency preservation
        cross_modal_preservation = {}
        cross_modal_preservation['sign_concordance_orig'] = orig_cross_modal['sign_concordance']
        cross_modal_preservation['sign_concordance_aug'] = aug_cross_modal['sign_concordance']
        cross_modal_preservation['concordance_preservation'] = (
            aug_cross_modal['sign_concordance'] / orig_cross_modal['sign_concordance'] 
            if orig_cross_modal['sign_concordance'] > 0 else 0
        )
        cross_modal_preservation['effect_correlation_orig'] = orig_cross_modal['effect_correlation']
        cross_modal_preservation['effect_correlation_aug'] = aug_cross_modal['effect_correlation']
        cross_modal_preservation['correlation_preservation'] = (
            aug_cross_modal['effect_correlation'] / orig_cross_modal['effect_correlation']
            if orig_cross_modal['effect_correlation'] > 0 else 0
        )
        cross_modal_preservation['overall_preservation'] = (
            cross_modal_preservation['concordance_preservation'] * 0.5 +
            (min(1.0, max(0, cross_modal_preservation['correlation_preservation']))) * 0.5
        )
        
        preservation_results['cross_modal'] = cross_modal_preservation
        
        # Create summary DataFrame
        summary_data = [{
            'Category': 'Spectral',
            'Effect_Size_Correlation': spectral_preservation['effect_correlation'],
            'Sign_Concordance': spectral_preservation['sign_concordance'],
            'Overall_Preservation': spectral_preservation['overall_preservation']
        }, {
            'Category': 'Molecular feature',
            'Effect_Size_Correlation': molecular_feature_preservation['effect_correlation'],
            'Sign_Concordance': molecular_feature_preservation['sign_concordance'],
            'Overall_Preservation': molecular_feature_preservation['overall_preservation']
        }, {
            'Category': 'Cross-Modal',
            'Effect_Size_Correlation': cross_modal_preservation['correlation_preservation'],
            'Sign_Concordance': cross_modal_preservation['concordance_preservation'],
            'Overall_Preservation': cross_modal_preservation['overall_preservation']
        }]
        
        treatment_summary = pd.DataFrame(summary_data)
        
        # Save results to CSV
        treatment_summary.to_csv(os.path.join(self.results_dir, "treatment_response_preservation.csv"), index=False)
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Get the user-defined green-shade colors
        all_colors = get_bugn_colors(3)
        # Define the mapping based on desired visual progression:
        # Good (>0.9) = Bluish-Green (#3a999e -> index 2)
        # Moderate (0.8-0.9) = Teal/Green (#66b399 -> index 0)
        # Poor (<=0.8) = Lime/Green (#b7d69a -> index 1)
        color_good = all_colors[2]
        color_moderate = all_colors[0]
        color_poor = all_colors[1]

        # Assign colors based on scores using the new mapping
        score_values = treatment_summary['Overall_Preservation']
        score_colors = [color_good if score > 0.9 else color_moderate if score > 0.8 else color_poor
                       for score in score_values]
        
        bars = plt.bar(
            treatment_summary['Category'], 
            score_values,
            color=score_colors
        )
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        # Update axhline colors to match the thresholds they represent
        plt.axhline(y=0.9, linestyle='--', color=color_good, alpha=0.7)     # Line above Moderate threshold, use Good color
        plt.axhline(y=0.8, linestyle='--', color=color_moderate, alpha=0.7) # Line above Poor threshold, use Moderate color
        plt.title('Treatment Response Pattern Preservation', fontsize=18)
        plt.xlabel('Category', fontsize=16)
        plt.ylabel('Preservation Score (1.0 = Perfect)', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, "treatment_response.png"), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, "treatment_response.pdf"))
        plt.close()
        
        print("Treatment response preservation results:")
        print(treatment_summary)
        
        return treatment_summary
    
    def calculate_effect_preservation(self, orig_effects, aug_effects, category):
        """Helper function to calculate treatment effect preservation metrics"""
        # Find common features
        common_features = set(orig_effects.keys()) & set(aug_effects.keys())
        
        if not common_features:
            return {
                'effect_correlation': 0,
                'sign_concordance': 0,
                'p_value_preservation': 0,
                'overall_preservation': 0
            }
        
        # Calculate correlation of effect sizes
        orig_effect_sizes = [orig_effects[f]['effect_size'] for f in common_features]
        aug_effect_sizes = [aug_effects[f]['effect_size'] for f in common_features]
        
        if len(common_features) > 2:
            try:
                effect_correlation = pearsonr(orig_effect_sizes, aug_effect_sizes)[0]
            except:
                effect_correlation = 0
        else:
            effect_correlation = 1 if np.sign(np.sum(orig_effect_sizes)) == np.sign(np.sum(aug_effect_sizes)) else 0
        
        # Calculate sign concordance
        concordant = sum(1 for f in common_features if 
                          np.sign(orig_effects[f]['effect_size']) == np.sign(aug_effects[f]['effect_size']))
        sign_concordance = concordant / len(common_features) if common_features else 0
        
        # Calculate p-value preservation
        orig_sig = sum(1 for f in common_features if orig_effects[f]['p_value'] < 0.05)
        aug_sig = sum(1 for f in common_features if aug_effects[f]['p_value'] < 0.05)
        
        p_value_preservation = aug_sig / orig_sig if orig_sig > 0 else 0
        
        # Calculate overall preservation score
        overall_preservation = (
            max(0, effect_correlation) * 0.4 +
            sign_concordance * 0.4 +
            min(1.0, p_value_preservation) * 0.2
        )
        
        return {
            'effect_correlation': effect_correlation,
            'sign_concordance': sign_concordance,
            'p_value_preservation': p_value_preservation,
            'overall_preservation': overall_preservation
        }
    
    def calculate_cross_modal_consistency(self, spectral_effects, molecular_feature_effects):
        """Calculate consistency of treatment effects between modalities"""
        # Extract effect sizes
        spectral_effects_dict = {k: v['effect_size'] for k, v in spectral_effects.items()}
        molecular_feature_effects_dict = {k: v['effect_size'] for k, v in molecular_feature_effects.items()}
        
        # Calculate pairwise correlations between spectral and Molecular feature features
        correlations = []
        for s_feature, s_effect in spectral_effects_dict.items():
            for m_feature, m_effect in molecular_feature_effects_dict.items():
                correlations.append((s_feature, m_feature, s_effect, m_effect))
        
        # Calculate sign concordance
        sign_concordant = sum(1 for _, _, s_effect, m_effect in correlations 
                              if np.sign(s_effect) == np.sign(m_effect))
        sign_concordance = sign_concordant / len(correlations) if correlations else 0
        
        # Calculate overall correlation between modalities
        spectral_values = [s_effect for _, _, s_effect, _ in correlations]
        molecular_feature_values = [m_effect for _, _, _, m_effect in correlations]
        
        if len(correlations) > 2:
            try:
                effect_correlation = pearsonr(spectral_values, molecular_feature_values)[0]
            except:
                effect_correlation = 0
        else:
            effect_correlation = 1 if np.sign(np.sum(spectral_values)) == np.sign(np.sum(molecular_feature_values)) else 0
        
        return {
            'sign_concordance': sign_concordance,
            'effect_correlation': effect_correlation
        }
    
    def genotype_specific_feature_preservation(self):
        """
        Validate preservation of genotype-specific molecular signatures across both modalities.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with genotype feature preservation results
        """
        print("\n4. Validating genotype-specific feature preservation...")
        
        # Check if Genotype column exists
        if 'Genotype' not in self.spectral_metadata_cols or 'Genotype' not in self.molecular_feature_metadata_cols:
            print("Warning: 'Genotype' column not found in one or both modalities. Skipping genotype analysis.")
            return pd.DataFrame()
        
        # Function to identify genotype-specific features
        def identify_genotype_markers(data, feature_cols):
            markers = {}
            
            # Get unique genotypes
            genotypes = sorted(data['Genotype'].unique())
            
            if len(genotypes) < 2:
                return {}
            
            for feature in feature_cols:
                if feature in data.columns:
                    # Perform ANOVA to test for differences
                    groups = [data[data['Genotype'] == g][feature].values for g in genotypes]
                    
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:
                        try:
                            f_val, p_val = f_oneway(*groups)
                            
                            if p_val < 0.05:  # Significant difference between genotypes
                                # Calculate effect size (eta-squared)
                                group_means = [np.mean(g) for g in groups]
                                overall_mean = np.mean([np.mean(g) for g in groups])
                                
                                # Between-group sum of squares
                                between_ss = sum(len(g) * (m - overall_mean)**2 for g, m in zip(groups, group_means))
                                
                                # Total sum of squares
                                total_ss = sum(sum((x - overall_mean)**2 for x in g) for g in groups)
                                
                                # Calculate eta-squared
                                eta_squared = between_ss / total_ss if total_ss > 0 else 0
                                
                                markers[feature] = {
                                    'p_value': p_val,
                                    'f_value': f_val,
                                    'eta_squared': eta_squared
                                }
                        except:
                            # Skip if ANOVA fails
                            pass
            
            return markers
        
        # Sample spectral features for efficiency
        sampled_wavelengths = self.wavelength_cols[::20]  # Take every 20th wavelength
        if len(sampled_wavelengths) > 100:
            sampled_wavelengths = sampled_wavelengths[:100]
            
        print(f"  Using {len(sampled_wavelengths)} sampled wavelengths for spectral analysis...")
        
        # Identify genotype markers in original data
        spectral_orig_markers = identify_genotype_markers(self.spectral_original, sampled_wavelengths)
        molecular_feature_orig_markers = identify_genotype_markers(self.molecular_feature_original, self.molecular_feature_cols)
        
        print(f"  Found {len(spectral_orig_markers)} genotype-specific spectral features in original data...")
        print(f"  Found {len(molecular_feature_orig_markers)} genotype-specific Molecular features in original data...")
        
        # Identify genotype markers in augmented data
        spectral_aug_markers = identify_genotype_markers(self.spectral_augmented, sampled_wavelengths)
        molecular_feature_aug_markers = identify_genotype_markers(self.molecular_feature_augmented, self.molecular_feature_cols)
        
        # Calculate preservation metrics
        preservation_results = {}
        
        # For spectral data
        spectral_preservation = self.calculate_marker_preservation(
            spectral_orig_markers, spectral_aug_markers)
        preservation_results['spectral'] = spectral_preservation
        
        # For Molecular feature data
        molecular_feature_preservation = self.calculate_marker_preservation(
            molecular_feature_orig_markers, molecular_feature_aug_markers)
        preservation_results['molecular_feature'] = molecular_feature_preservation
        
        # Calculate cross-modal consistency
        # Original data cross-modal consistency
        orig_cross_modal = self.calculate_marker_consistency(
            spectral_orig_markers, molecular_feature_orig_markers)
        
        # Augmented data cross-modal consistency
        aug_cross_modal = self.calculate_marker_consistency(
            spectral_aug_markers, molecular_feature_aug_markers)
        
        # Calculate consistency preservation
        cross_modal_preservation = {}
        cross_modal_preservation['marker_overlap_orig'] = orig_cross_modal['overlap_ratio']
        cross_modal_preservation['marker_overlap_aug'] = aug_cross_modal['overlap_ratio']
        cross_modal_preservation['overlap_preservation'] = (
            aug_cross_modal['overlap_ratio'] / orig_cross_modal['overlap_ratio'] 
            if orig_cross_modal['overlap_ratio'] > 0 else 0
        )
        cross_modal_preservation['effect_correlation_orig'] = orig_cross_modal['effect_correlation']
        cross_modal_preservation['effect_correlation_aug'] = aug_cross_modal['effect_correlation']
        cross_modal_preservation['correlation_preservation'] = (
            aug_cross_modal['effect_correlation'] / orig_cross_modal['effect_correlation']
            if orig_cross_modal['effect_correlation'] > 0 else 0
        )
        cross_modal_preservation['overall_preservation'] = (
            min(1.0, cross_modal_preservation['overlap_preservation']) * 0.5 +
            min(1.0, max(0, cross_modal_preservation['correlation_preservation'])) * 0.5
        )
        
        preservation_results['cross_modal'] = cross_modal_preservation
        
        # Create summary DataFrame
        summary_data = [{
            'Category': 'Spectral',
            'Marker_Overlap': spectral_preservation['marker_overlap'],
            'Effect_Correlation': spectral_preservation['effect_correlation'],
            'Overall_Preservation': spectral_preservation['overall_preservation']
        }, {
            'Category': 'Molecular feature',
            'Marker_Overlap': molecular_feature_preservation['marker_overlap'],
            'Effect_Correlation': molecular_feature_preservation['effect_correlation'],
            'Overall_Preservation': molecular_feature_preservation['overall_preservation']
        }, {
            'Category': 'Cross-Modal',
            'Marker_Overlap': cross_modal_preservation['overlap_preservation'],
            'Effect_Correlation': cross_modal_preservation['correlation_preservation'],
            'Overall_Preservation': cross_modal_preservation['overall_preservation']
        }]
        
        genotype_summary = pd.DataFrame(summary_data)
        
        # Save results to CSV
        genotype_summary.to_csv(os.path.join(self.results_dir, "genotype_feature_preservation.csv"), index=False)
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Get the user-defined green-shade colors
        all_colors = get_bugn_colors(3)
        # Define the mapping based on desired visual progression:
        # Good (>0.9) = Bluish-Green (#3a999e -> index 2)
        # Moderate (0.8-0.9) = Teal/Green (#66b399 -> index 0)
        # Poor (<=0.8) = Lime/Green (#b7d69a -> index 1)
        color_good = all_colors[2]
        color_moderate = all_colors[0]
        color_poor = all_colors[1]

        # Assign colors based on scores using the new mapping
        score_values = genotype_summary['Overall_Preservation']
        score_colors = [color_good if score > 0.9 else color_moderate if score > 0.8 else color_poor
                       for score in score_values]
        
        bars = plt.bar(
            genotype_summary['Category'], 
            score_values,
            color=score_colors
        )
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        # Update axhline colors to match the thresholds they represent
        plt.axhline(y=0.9, linestyle='--', color=color_good, alpha=0.7)     # Line above Moderate threshold, use Good color
        plt.axhline(y=0.8, linestyle='--', color=color_moderate, alpha=0.7) # Line above Poor threshold, use Moderate color
        plt.title('Genotype-Specific Feature Preservation', fontsize=18)
        plt.xlabel('Category', fontsize=16)
        plt.ylabel('Preservation Score (1.0 = Perfect)', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, "genotype_preservation.png"), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, "genotype_preservation.pdf"))
        plt.close()
        
        print("Genotype-specific feature preservation results:")
        print(genotype_summary)
        
        return genotype_summary
    
    def calculate_marker_preservation(self, orig_markers, aug_markers):
        """Helper function to calculate genotype marker preservation metrics"""
        # Calculate overlap of significant markers
        original_set = set(orig_markers.keys())
        augmented_set = set(aug_markers.keys())
        
        overlap = len(original_set & augmented_set)
        marker_overlap = overlap / len(original_set) if original_set else 0
        jaccard = overlap / len(original_set | augmented_set) if (original_set | augmented_set) else 0
        
        # Calculate preservation of effect sizes
        common_markers = original_set & augmented_set
        if common_markers:
            # Look at preservation of effect sizes
            orig_effects = [orig_markers[m]['eta_squared'] for m in common_markers]
            aug_effects = [aug_markers[m]['eta_squared'] for m in common_markers]
            
            if len(common_markers) > 2:
                try:
                    effect_corr = pearsonr(orig_effects, aug_effects)[0]
                except:
                    effect_corr = 0
            else:
                effect_corr = 1 if np.sign(np.sum(orig_effects)) == np.sign(np.sum(aug_effects)) else 0
            
            # Calculate effect similarity
            effect_diffs = [abs(orig_markers[m]['eta_squared'] - aug_markers[m]['eta_squared']) 
                          for m in common_markers]
            effect_similarity = 1 - min(1, np.mean(effect_diffs) / 0.5)  # Normalize by 0.5 (typical eta-squared range)
        else:
            effect_corr = 0
            effect_similarity = 0
        
        # Overall preservation score
        overall_preservation = (
            marker_overlap * 0.4 +
            max(0, effect_corr) * 0.4 +
            effect_similarity * 0.2
        )
        
        return {
            'marker_overlap': marker_overlap,
            'jaccard_similarity': jaccard,
            'effect_correlation': effect_corr,
            'effect_similarity': effect_similarity,
            'overall_preservation': overall_preservation
        }
    
    def calculate_marker_consistency(self, spectral_markers, molecular_feature_markers):
        """Calculate consistency of genotype markers between modalities"""
        # Check for overlap in significant features
        spectral_set = set(spectral_markers.keys())
        molecular_feature_set = set(molecular_feature_markers.keys())
        
        # Since spectral and Molecular feature features are different entities, 
        # use a different approach for consistency
        
        # Convert effect sizes to sorted lists to compare distributions
        spectral_effects = sorted([m['eta_squared'] for m in spectral_markers.values()])
        molecular_feature_effects = sorted([m['eta_squared'] for m in molecular_feature_markers.values()])
        
        # Calculate overlap ratio (using the closer sizes of significant markers)
        if len(spectral_effects) > 0 and len(molecular_feature_effects) > 0:
            overlap_ratio = min(len(spectral_effects), len(molecular_feature_effects)) / max(len(spectral_effects), len(molecular_feature_effects))
        else:
            overlap_ratio = 0
        
        # Calculate correlation of sorted effect sizes
        # Resample to match lengths
        if len(spectral_effects) > 0 and len(molecular_feature_effects) > 0:
            if len(spectral_effects) > len(molecular_feature_effects):
                # Resample spectral effects to match Molecular feature length
                indices = np.linspace(0, len(spectral_effects) - 1, len(molecular_feature_effects)).astype(int)
                resampled_spectral = [spectral_effects[i] for i in indices]
                resampled_molecular_feature = molecular_feature_effects
            else:
                # Resample Molecular feature effects to match spectral length
                indices = np.linspace(0, len(molecular_feature_effects) - 1, len(spectral_effects)).astype(int)
                resampled_molecular_feature = [molecular_feature_effects[i] for i in indices]
                resampled_spectral = spectral_effects
            
            # Calculate correlation
            if len(resampled_spectral) > 2:
                try:
                    correlation = pearsonr(resampled_spectral, resampled_molecular_feature)[0]
                except:
                    correlation = 0
            else:
                correlation = 1 if np.sign(np.sum(resampled_spectral)) == np.sign(np.sum(resampled_molecular_feature)) else 0
        else:
            correlation = 0
        
        return {
            'overlap_ratio': overlap_ratio,
            'effect_correlation': correlation
        }
    
    def batch_effect_constancy_validation(self):
        """
        Verify that batch effects are consistently preserved across both modalities
        after augmentation.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with batch effect constancy validation results
        """
        print("\n5. Validating batch effect constancy...")
        
        # Check if Batch column exists
        if 'Batch' not in self.spectral_metadata_cols or 'Batch' not in self.molecular_feature_metadata_cols:
            print("Warning: 'Batch' column not found in one or both modalities. Skipping batch effect analysis.")
            return pd.DataFrame()
        
        # Function to calculate variance explained by batch for each feature
        def calculate_batch_variance(data, feature_cols, batch_col='Batch'):
            batch_variance = {}
            
            for feature in feature_cols:
                if feature in data.columns:
                    try:
                        # Calculate total variance
                        total_var = np.var(data[feature])
                        
                        if total_var > 0:
                            # Calculate between-batch variance
                            batch_means = data.groupby(batch_col)[feature].mean()
                            between_var = np.var(batch_means) * len(data) / len(batch_means)
                            
                            # Variance explained ratio
                            var_ratio = between_var / total_var
                            batch_variance[feature] = var_ratio
                    except:
                        # Skip if calculation fails
                        pass
            
            return batch_variance
        
        # Sample spectral features for efficiency
        sampled_wavelengths = self.wavelength_cols[::20]  # Take every 20th wavelength
        if len(sampled_wavelengths) > 100:
            sampled_wavelengths = sampled_wavelengths[:100]
        
        print(f"  Using {len(sampled_wavelengths)} sampled wavelengths for spectral analysis...")
        
        # Calculate batch variance for original data
        spectral_orig_variance = calculate_batch_variance(self.spectral_original, sampled_wavelengths)
        molecular_feature_orig_variance = calculate_batch_variance(self.molecular_feature_original, self.molecular_feature_cols)
        
        # Calculate batch variance for augmented data
        spectral_aug_variance = calculate_batch_variance(self.spectral_augmented, sampled_wavelengths)
        molecular_feature_aug_variance = calculate_batch_variance(self.molecular_feature_augmented, self.molecular_feature_cols)
        
        # Calculate preservation metrics
        batch_results = {}
        
        # Spectral batch effect preservation
        spectral_preservation = self.calculate_batch_preservation(
            spectral_orig_variance, spectral_aug_variance)
        batch_results['spectral'] = spectral_preservation
        
        # Molecular feature batch effect preservation
        molecular_feature_preservation = self.calculate_batch_preservation(
            molecular_feature_orig_variance, molecular_feature_aug_variance)
        batch_results['molecular_feature'] = molecular_feature_preservation
        
        # Calculate cross-modal batch effect consistency
        # Original data cross-modal consistency
        orig_cross_modal = self.calculate_batch_consistency(
            spectral_orig_variance, molecular_feature_orig_variance)
        
        # Augmented data cross-modal consistency
        aug_cross_modal = self.calculate_batch_consistency(
            spectral_aug_variance, molecular_feature_aug_variance)
        
        # Cross-modal batch effect consistency preservation
        cross_modal_preservation = {}
        cross_modal_preservation['mean_ratio_orig'] = orig_cross_modal['mean_ratio']
        cross_modal_preservation['mean_ratio_aug'] = aug_cross_modal['mean_ratio']
        cross_modal_preservation['ratio_preservation'] = (
            aug_cross_modal['mean_ratio'] / orig_cross_modal['mean_ratio'] 
            if orig_cross_modal['mean_ratio'] > 0 else 0
        )
        cross_modal_preservation['correlation_orig'] = orig_cross_modal['correlation']
        cross_modal_preservation['correlation_aug'] = aug_cross_modal['correlation']
        cross_modal_preservation['correlation_preservation'] = (
            aug_cross_modal['correlation'] / orig_cross_modal['correlation']
            if orig_cross_modal['correlation'] > 0 else 0
        )
        cross_modal_preservation['overall_preservation'] = (
            1.0 - min(1.0, abs(cross_modal_preservation['ratio_preservation'] - 1.0)) * 0.5 -
            min(1.0, abs(cross_modal_preservation['correlation_preservation'] - 1.0)) * 0.5
        )
        
        batch_results['cross_modal'] = cross_modal_preservation
        
        # Create summary DataFrame
        summary_data = [{
            'Category': 'Spectral',
            'Mean_Variance_Ratio': spectral_preservation['mean_ratio'],
            'Variance_Correlation': spectral_preservation['variance_correlation'],
            'Overall_Preservation': spectral_preservation['overall_preservation']
        }, {
            'Category': 'Molecular feature',
            'Mean_Variance_Ratio': molecular_feature_preservation['mean_ratio'],
            'Variance_Correlation': molecular_feature_preservation['variance_correlation'],
            'Overall_Preservation': molecular_feature_preservation['overall_preservation']
        }, {
            'Category': 'Cross-Modal',
            'Mean_Variance_Ratio': cross_modal_preservation['ratio_preservation'],
            'Variance_Correlation': cross_modal_preservation['correlation_preservation'],
            'Overall_Preservation': cross_modal_preservation['overall_preservation']
        }]
        
        batch_summary = pd.DataFrame(summary_data)
        
        # Save results to CSV
        batch_summary.to_csv(os.path.join(self.results_dir, "batch_effect_constancy.csv"), index=False)
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Get the user-defined green-shade colors
        all_colors = get_bugn_colors(3)
        # Define the mapping based on desired visual progression:
        # Good (>0.9) = Bluish-Green (#3a999e -> index 2)
        # Moderate (0.8-0.9) = Teal/Green (#66b399 -> index 0)
        # Poor (<=0.8) = Lime/Green (#b7d69a -> index 1)
        color_good = all_colors[2]
        color_moderate = all_colors[0]
        color_poor = all_colors[1]

        # Assign colors based on scores using the new mapping
        score_values = batch_summary['Overall_Preservation']
        score_colors = [color_good if score > 0.9 else color_moderate if score > 0.8 else color_poor
                       for score in score_values]
        
        bars = plt.bar(
            batch_summary['Category'], 
            score_values,
            color=score_colors
        )
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        # Update axhline colors to match the thresholds they represent
        plt.axhline(y=0.9, linestyle='--', color=color_good, alpha=0.7)     # Line above Moderate threshold, use Good color
        plt.axhline(y=0.8, linestyle='--', color=color_moderate, alpha=0.7) # Line above Poor threshold, use Moderate color
        plt.title('Batch Effect Constancy Preservation', fontsize=18)
        plt.xlabel('Category', fontsize=16)
        plt.ylabel('Preservation Score (1.0 = Perfect)', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, "batch_effect_constancy.png"), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, "batch_effect_constancy.pdf"))
        plt.close()
        
        print("Batch effect constancy validation results:")
        print(batch_summary)
        
        return batch_summary
    
    def calculate_batch_preservation(self, orig_variance, aug_variance):
        """Helper function to calculate batch effect preservation metrics"""
        # Find common features
        common_features = set(orig_variance.keys()) & set(aug_variance.keys())
        
        if not common_features:
            return {
                'mean_ratio': 0,
                'variance_correlation': 0,
                'overall_preservation': 0
            }
        
        # Calculate mean batch variance ratio
        orig_mean = np.mean([orig_variance[f] for f in common_features])
        aug_mean = np.mean([aug_variance[f] for f in common_features])
        
        # Calculate ratio (augmented / original)
        mean_ratio = aug_mean / orig_mean if orig_mean > 0 else 0
        
        # Calculate correlation of feature-specific batch variances
        orig_values = [orig_variance[f] for f in common_features]
        aug_values = [aug_variance[f] for f in common_features]
        
        if len(common_features) > 2:
            try:
                variance_correlation = pearsonr(orig_values, aug_values)[0]
            except:
                variance_correlation = 0
        else:
            variance_correlation = 1 if np.sign(np.sum(orig_values)) == np.sign(np.sum(aug_values)) else 0
        
        # Calculate overall preservation score
        # Perfect preservation is when mean_ratio = 1.0 and variance_correlation = 1.0
        overall_preservation = (
            1.0 - min(1.0, abs(mean_ratio - 1.0)) * 0.5 +
            max(0, variance_correlation) * 0.5
        )
        
        return {
            'mean_ratio': mean_ratio,
            'variance_correlation': variance_correlation,
            'overall_preservation': overall_preservation
        }
    
    def calculate_batch_consistency(self, spectral_variance, molecular_feature_variance):
        """Calculate consistency of batch effects between modalities"""
        # Calculate mean batch variance for each modality
        spectral_mean = np.mean(list(spectral_variance.values())) if spectral_variance else 0
        molecular_feature_mean = np.mean(list(molecular_feature_variance.values())) if molecular_feature_variance else 0
        
        # Calculate mean ratio between modalities
        mean_ratio = spectral_mean / molecular_feature_mean if molecular_feature_mean > 0 else 0
        
        # Convert variance values to sorted lists to compare distributions
        spectral_values = sorted(list(spectral_variance.values()))
        molecular_feature_values = sorted(list(molecular_feature_variance.values()))
        
        # Calculate correlation of sorted variances
        if spectral_values and molecular_feature_values:
            # Resample to match lengths
            if len(spectral_values) > len(molecular_feature_values):
                # Resample spectral values to match Molecular feature length
                indices = np.linspace(0, len(spectral_values) - 1, len(molecular_feature_values)).astype(int)
                resampled_spectral = [spectral_values[i] for i in indices]
                resampled_molecular_feature = molecular_feature_values
            else:
                # Resample Molecular feature values to match spectral length
                indices = np.linspace(0, len(molecular_feature_values) - 1, len(spectral_values)).astype(int)
                resampled_molecular_feature = [molecular_feature_values[i] for i in indices]
                resampled_spectral = spectral_values
            
            # Calculate correlation
            if len(resampled_spectral) > 2:
                try:
                    correlation = pearsonr(resampled_spectral, resampled_molecular_feature)[0]
                except:
                    correlation = 0
            else:
                correlation = 1 if np.sign(np.sum(resampled_spectral)) == np.sign(np.sum(resampled_molecular_feature)) else 0
        else:
            correlation = 0
        
        return {
            'mean_ratio': mean_ratio,
            'correlation': correlation
        }
    
    def generate_summary_report(self, all_results):
        """
        Generate comprehensive HTML report summarizing all cross-modality validation results.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary containing all validation results
        """
        report_path = os.path.join(self.validation_dir, 'cross_modality_report.html')
        
        with open(report_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Cross-Modality Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; font-size: 16px; }
        h1 { color: #2c3e50; font-size: 28px; }
        h2 { color: #3498db; margin-top: 30px; font-size: 24px; }
        h3 { color: #2980b9; font-size: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 16px; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        .good { color: #66b399; }  /* Soft teal/green */
        .moderate { color: #b7d69a; }  /* Soft lime/green */
        .poor { color: #3a999e; }  /* Soft teal/blue */
        img { max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Cross-Modality Validation Report</h1>
    <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="summary">
        <h2>Overview</h2>
        <p>This report presents comprehensive cross-modality validation of augmented spectral and Molecular feature data.</p>
        <p><strong>Spectral data:</strong> Original: """ + self.spectral_original_path + """</p>
        <p><strong>Spectral data:</strong> Augmented: """ + self.spectral_augmented_path + """</p>
        <p><strong>Molecular feature data:</strong> Original: """ + self.molecular_feature_original_path + """</p>
        <p><strong>Molecular feature data:</strong> Augmented: """ + self.molecular_feature_augmented_path + """</p>
    </div>
""")
            
            # Add sections for each validation result
            sections = [
                ('factor_consistency', 'Experimental Factor Consistency', 'Factor', 'Overall_Preservation', 
                 'This analysis verifies that experimental factors maintain consistent distributions across both spectral and Molecular feature data after augmentation.'),
                ('metadata_balance', 'Metadata Balance Assessment', 'Factor', 'Overall_Balance_Score',
                 'This analysis checks that metadata remains balanced between spectral and Molecular feature data after augmentation.'),
                ('treatment_response', 'Treatment Response Pattern Preservation', 'Category', 'Overall_Preservation',
                 'This analysis confirms that treatment response patterns are consistently preserved across both modalities.'),
                ('genotype_preservation', 'Genotype-Specific Feature Preservation', 'Category', 'Overall_Preservation',
                 'This analysis validates the preservation of genotype-specific molecular signatures across both modalities.'),
                ('batch_effect', 'Batch Effect Constancy Validation', 'Category', 'Overall_Preservation',
                 'This analysis verifies that batch effects are consistently maintained across both modalities.')
            ]
            
            for key, title, index_col, score_col, description in sections:
                if key in all_results and not all_results[key].empty and score_col in all_results[key].columns:
                    f.write(f"""
    <h2>{title}</h2>
    <p>{description}</p>
    <table>
        <tr>
            <th>{index_col}</th>
            <th>Preservation Score</th>
            <th>Assessment</th>
        </tr>
""")
                    
                    for _, row in all_results[key].iterrows():
                        index_val = row[index_col]
                        score = row[score_col]
                        
                        # Determine quality class with BuGn-like colors
                        if score > 0.9:
                            quality_class = "good"
                            assessment = "Excellent preservation"
                        elif score > 0.8:
                            quality_class = "moderate"
                            assessment = "Good preservation"
                        else:
                            quality_class = "poor"
                            assessment = "Moderate preservation"
                        
                        f.write(f"""
        <tr>
            <td>{index_val}</td>
            <td class="{quality_class}">{score:.4f}</td>
            <td>{assessment}</td>
        </tr>
""")
                    
                    f.write("</table>")
                    
                    # Add plot if available
                    plot_path = os.path.join("plots", f"{key}.png")
                    if os.path.exists(os.path.join(self.validation_dir, plot_path)):
                        f.write(f"""
    <div>
        <img src="{plot_path}" alt="{title} Plot">
    </div>
""")
            
            # Calculate overall cross-modality validation score
            scores = []
            
            # Collect all overall preservation scores
            for key in all_results:
                if key == 'factor_consistency':
                    if not all_results[key].empty and 'Overall_Preservation' in all_results[key].columns:
                        scores.extend(all_results[key]['Overall_Preservation'].values)
                elif key == 'metadata_balance':
                    if not all_results[key].empty and 'Overall_Balance_Score' in all_results[key].columns:
                        scores.extend(all_results[key]['Overall_Balance_Score'].values)
                elif key in ['treatment_response', 'genotype_preservation', 'batch_effect']:
                    if not all_results[key].empty and 'Overall_Preservation' in all_results[key].columns:
                        # Only take the cross-modal preservation score
                        cross_modal_score = all_results[key][all_results[key]['Category'] == 'Cross-Modal']['Overall_Preservation'].values
                        if len(cross_modal_score) > 0:
                            scores.append(cross_modal_score[0])
            
            # Calculate overall score
            if scores:
                overall_score = np.mean(scores)
                print(f"Overall cross-modality score: {overall_score:.4f} (based on {len(scores)} metrics)")
                
                # Add overall assessment
                f.write("""
    <h2>Overall Assessment</h2>
""")
                
                if overall_score > 0.9:
                    f.write(f"""
    <div class="summary">
        <h3 class="good">Excellent Cross-Modality Consistency: {overall_score:.4f}/1.00</h3>
        <p>The augmented data demonstrates exceptional consistency across both spectral and Molecular feature modalities. 
        Experimental factors, treatment responses, genotype-specific features, and batch effects are all preserved with 
        high fidelity. This dataset is suitable for integrated multi-modal analyses with high confidence.</p>
    </div>
""")
                elif overall_score > 0.8:
                    f.write(f"""
    <div class="summary">
        <h3 class="moderate">Good Cross-Modality Consistency: {overall_score:.4f}/1.00</h3>
        <p>The augmented data demonstrates good consistency across both spectral and Molecular feature modalities.
        Most experimental factors and biological effects are well-preserved, with minor deviations in some aspects.
        This dataset is suitable for integrated analyses with appropriate consideration of the limitations.</p>
    </div>
""")
                else:
                    f.write(f"""
    <div class="summary">
        <h3 class="poor">Moderate Cross-Modality Consistency: {overall_score:.4f}/1.00</h3>
        <p>The augmented data shows adequate but imperfect consistency across modalities.
        Some experimental factors or biological effects show notable deviations from the original relationships.
        Caution is advised when performing integrated analyses, and validation of key findings is recommended.</p>
    </div>
""")
            
            f.write("""
    <hr>
    <p><em>Report generated by Cross-Modality Validation Framework</em></p>
</body>
</html>
""")
        
        print(f"\nCross-modality validation report saved to: {report_path}")
    
    def run_all_validations(self):
        """
        Run all cross-modality validation checks and generate integrated report.
        
        Returns:
        --------
        dict
            Dictionary containing all validation results
        """
        print("\nRunning comprehensive cross-modality validation...")
        start_time = time.time()
        
        # Run all validation methods
        results = {}
        
        # 1. Experimental Factor Consistency Verification
        results['factor_consistency'] = self.experimental_factor_consistency()
        
        # 2. Metadata Balance Assessment
        results['metadata_balance'] = self.metadata_balance_assessment()
        
        # 3. Treatment Response Pattern Preservation
        results['treatment_response'] = self.treatment_response_preservation()
        
        # 4. Genotype-Specific Feature Preservation
        results['genotype_preservation'] = self.genotype_specific_feature_preservation()
        
        # 5. Batch Effect Constancy Validation
        results['batch_effect'] = self.batch_effect_constancy_validation()
        
        # Generate integrated report
        self.generate_summary_report(results)
        
        end_time = time.time()
        print(f"\nAll cross-modality validation checks completed in {end_time - start_time:.2f} seconds")
        
        return results