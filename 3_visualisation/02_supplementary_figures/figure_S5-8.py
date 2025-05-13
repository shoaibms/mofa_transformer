"""
Data Visualization Suite for Augmented Plant Science Datasets.

This script provides a comprehensive suite for generating publication-quality
visualizations to analyze and compare original multi-modal (spectral and metabolite)
plant science data against their augmented counterparts. It includes functionalities
for visualizing:
- Spectral signatures and metabolite profiles.
- The effect of data augmentation on experimental factor preservation.
- Comparative analysis of different augmentation methods.
- Integrated dashboards summarizing overall augmentation quality.

The suite is designed to be robust, handling various data characteristics
and providing detailed statistical comparisons where appropriate.
It leverages common Python libraries such as pandas, numpy, matplotlib,
seaborn, and scikit-learn.
"""
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance, mannwhitneyu
from scipy.spatial.distance import jensenshannon
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

# Insert COLORS dictionary here
COLORS = {
    # ==========================================================================
    # == Core Experimental Variables ==
    # ==========================================================================
    'G1': '#00FA9A',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#48D1CC',             # Susceptible Genotype (Medium Teal)
    'T0': '#4682B4',            # Control Treatment (Medium Green)
    'T1': '#BDB76B',             # Stress Treatment (Muted Orange/Yellow)
    'Leaf': '#00FF00',            # Leaf Tissue (Darkest Green)
    'Root': '#40E0D0',            # Root Tissue (Darkest Blue)
    'Day1': '#ffffcc',            # Very Light Yellow-Green
    'Day2': '#9CBA79',            # Light Yellow-Green
    'Day3': '#3e7d5a',            # Medium Yellow-Green

    # ==========================================================================
    # == Data Types / Omics / Features ==
    # ==========================================================================
    'Spectral': '#ECDA79',        # General Spectral (Medium Blue)
    'Metabolite': '#84ab92',       # General Metabolite (Medium-Dark Yellow-Green)
    'UnknownFeature': '#B0E0E6',  # Medium Grey for fallback
    'Spectral_Water': '#6DCAFA',     # Medium-Dark Blue
    'Spectral_Pigment': '#00FA9A',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#00BFFF',         # Darkest Blue
    'Spectral_Other': '#969696',      # Medium Grey
    'Metabolite_PCluster': '#3DB3BF', # Darkest Yellow-Green
    'Metabolite_NCluster': '#ffffd4', # Very Light Yellow
    'Metabolite_Other': '#bdbdbd',     # Light Grey

    # ==========================================================================
    # == Methods & Model Comparison ==
    # ==========================================================================
    'MOFA': '#FFEBCD',            # Dark Blue
    'SHAP': '#F0E68C',            # Dark Green
    'Overlap': '#AFEEEE',         # Medium-Dark Yellow-Green
    'Transformer': '#fae3a2',     # Medium Blue
    'RandomForest': '#40E0D0',    # Medium Green
    'KNN': '#729c87',             # Medium Teal

    # ==========================================================================
    # == Network Visualization Elements ==
    # ==========================================================================
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#EEE8AA',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Metabolite': '#FFC4A1',   # Default Metabolite Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # ==========================================================================
    # == Statistical & Difference Indicators ==
    # ==========================================================================
    'Positive_Diff': '#66CDAA',     # Medium-Dark Green
    'Negative_Diff': '#fe9929',     # Muted Orange/Yellow
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

# Font settings from colour.py
FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],  # Fallback list
    'main_title': 22,
    'panel_label': 19,  # Note: This script doesn't explicitly add A, B, C labels, but we can use for panel titles
    'panel_title': 17,
    'axis_label': 17,
    'tick_label': 16,
    'legend_title': 19,
    'legend_text': 16,
    'annotation': 15,
    'caption': 15,
    'table_header': 15,
    'table_cell': 15,
}


class VisualizationSuite:
    """
    Comprehensive visualization suite for multi-modal plant data analysis,
    providing publication-quality visualizations optimized for high-performance hardware.
    """
    def __init__(self, spectral_original_path, spectral_augmented_path,
                 metabolite_original_path, metabolite_augmented_path, output_dir):
        self.spectral_original_path = spectral_original_path
        self.spectral_augmented_path = spectral_augmented_path
        self.metabolite_original_path = metabolite_original_path
        self.metabolite_augmented_path = metabolite_augmented_path
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load COLORS and FONTS
        self.COLORS = COLORS
        self.FONTS = FONTS_SANS

        self.load_data()
        self.setup_plotting_style()

    def load_data(self):
        """Load data efficiently, minimizing memory usage."""
        print("Loading data...")
        self.spectral_original = pd.read_csv(self.spectral_original_path)
        self.spectral_augmented = pd.read_csv(self.spectral_augmented_path)
        self.metabolite_original = pd.read_csv(self.metabolite_original_path)
        self.metabolite_augmented = pd.read_csv(self.metabolite_augmented_path)

        # Define feature and metadata columns
        self.wavelength_cols = [col for col in self.spectral_original.columns if col.startswith('W_')]
        self.spectral_metadata_cols = [col for col in self.spectral_original.columns if not col.startswith('W_')]

        # For metabolite data, identify proper column prefixes
        if any(col.startswith('M_') for col in self.metabolite_original.columns):
            self.metabolite_cols = [col for col in self.metabolite_original.columns if col.startswith('M_')]
            self.metabolite_metadata_cols = [col for col in self.metabolite_original.columns if not col.startswith('M_')]
        elif any(col.startswith('N_Cluster_') or col.startswith('P_Cluster_') for col in self.metabolite_original.columns):
            self.metabolite_cols = [col for col in self.metabolite_original.columns
                                   if col.startswith('N_Cluster_') or col.startswith('P_Cluster_')]
            self.metabolite_metadata_cols = [col for col in self.metabolite_original.columns
                                           if not (col.startswith('N_Cluster_') or col.startswith('P_Cluster_'))]
        else:
            # Fallback: assume non-metadata columns are metabolite features
            # Guess metadata cols based on spectral metadata or limited unique values
            potential_meta = [col for col in self.metabolite_original.columns
                             if col in self.spectral_metadata_cols or self.metabolite_original[col].nunique() < 20]
            self.metabolite_metadata_cols = potential_meta
            self.metabolite_cols = [col for col in self.metabolite_original.columns
                                  if col not in self.metabolite_metadata_cols]

        self.wavelengths = np.array([float(col.split('_')[1]) for col in self.wavelength_cols])

        # Identify common metadata
        self.common_metadata = list(set(self.spectral_metadata_cols).intersection(
                                   set(self.metabolite_metadata_cols)))
        if 'Row_names' in self.common_metadata:
            self.common_metadata.remove('Row_names') # Often just an ID

        # Filter augmented-only data
        self.spectral_augmented_only = self.spectral_augmented[~self.spectral_augmented['Row_names'].isin(self.spectral_original['Row_names'])]
        self.metabolite_augmented_only = self.metabolite_augmented[~self.metabolite_augmented['Row_names'].isin(self.metabolite_original['Row_names'])]

        # Extract augmentation methods
        self.spectral_methods = []
        if 'Row_names' in self.spectral_augmented_only.columns:
             self.spectral_methods = list(sorted({row.split('_')[-1] for row in self.spectral_augmented_only['Row_names'] if '_' in row}))
        self.metabolite_methods = []
        if 'Row_names' in self.metabolite_augmented_only.columns:
            self.metabolite_methods = list(sorted({row.split('_')[-1] for row in self.metabolite_augmented_only['Row_names'] if '_' in row}))

        print(f"Data loaded successfully:")
        print(f"  - Spectral: {len(self.spectral_original)} original, {len(self.spectral_augmented_only)} augmented")
        print(f"  - Metabolite: {len(self.metabolite_original)} original, {len(self.metabolite_augmented_only)} augmented")
        print(f"  - Wavelength features: {len(self.wavelength_cols)}")
        print(f"  - Metabolite features: {len(self.metabolite_cols)}")
        print(f"  - Common metadata: {len(self.common_metadata)}")
        print(f"  - Augmentation methods: Spectral ({len(self.spectral_methods)}: {self.spectral_methods}), Metabolite ({len(self.metabolite_methods)}: {self.metabolite_methods})")

    def setup_plotting_style(self):
        """Set up consistent plotting styles using COLORS and FONTS."""
        plt.rcParams['font.family'] = self.FONTS['family']
        plt.rcParams['font.sans-serif'] = self.FONTS['sans_serif']
        plt.rcParams['font.size'] = self.FONTS['tick_label'] # Base size
        plt.rcParams['axes.labelcolor'] = self.COLORS['Text_Dark']
        plt.rcParams['xtick.color'] = self.COLORS['Text_Dark']
        plt.rcParams['ytick.color'] = self.COLORS['Text_Dark']
        plt.rcParams['axes.titlecolor'] = self.COLORS['Text_Dark']
        plt.rcParams['figure.facecolor'] = self.COLORS['Background']
        plt.rcParams['axes.facecolor'] = self.COLORS['Panel_Background']
        plt.rcParams['grid.color'] = self.COLORS['Grid']

        sns.set_style('whitegrid', {
            'axes.edgecolor': self.COLORS['Text_Dark'],
            'grid.color': self.COLORS['Grid'],
            'axes.facecolor': self.COLORS['Panel_Background'],
            'figure.facecolor': self.COLORS['Background']
            })

        # Define specific color uses
        self.plot_colors = {
            'original': self.COLORS.get('G1', '#3182bd'),  # Using G1 (bluish) for original
            'augmented': self.COLORS.get('T1', '#fe9929'), # Using T1 (orange/yellow) for augmented
            'spectral': self.COLORS.get('Spectral', '#6baed6'),
            'molecular_features': self.COLORS.get('Molecular features', '#41ab5d')
        }
        self.correlation_cmap = sns.diverging_palette(
            220, 10, as_cmap=True
        ) # Keep standard diverging map for correlations

        # Colormap for multiple levels/methods (using tab10 as a fallback if not enough distinct COLORS)
        self.discrete_cmap = plt.cm.tab10 # Default

    def factor_specific_augmentation_effect(self):
        """
        Creates comprehensive visualization showing how experimental factors are preserved
        during data augmentation across both spectral and Molecular features modalities.

        This visualization helps assess whether the augmentation process maintains the
        biological signal associated with key experimental factors.
        """
        print("Generating factor-specific augmentation effect visualization...")
        start_time = time.time()

        # Identify key experimental factors common to both spectral and Molecular features data
        potential_factors = ['Treatment', 'Genotype', 'Batch', 'Day', 'Tissue.type', 'Replication']
        common_factors = [f for f in potential_factors if f in self.spectral_original.columns
                         and f in self.molecular_features_original.columns
                         and f in self.common_metadata] # Ensure it's truly common

        if not common_factors:
            print("Warning: No common predefined experimental factors found.")
            print("Checking for other potential categorical columns...")
            # Look for any categorical columns that might be experimental factors
            spectral_cat_cols = [col for col in self.spectral_metadata_cols
                                if self.spectral_original[col].dtype == 'object'
                                or (self.spectral_original[col].nunique() < 10 and self.spectral_original[col].nunique() > 1)] # Exclude single-value columns
            molecular_features_cat_cols = [col for col in self.molecular_features_metadata_cols
                                  if self.molecular_features_original[col].dtype == 'object'
                                  or (self.molecular_features_original[col].nunique() < 10 and self.molecular_features_original[col].nunique() > 1)]
            common_factors = list(set(spectral_cat_cols) & set(molecular_features_cat_cols))

            if not common_factors:
                print("No common categorical factors found. Cannot generate factor-specific visualization.")
                return

        print(f"Found {len(common_factors)} common experimental factors: {', '.join(common_factors)}")

        # Create a multi-panel figure for factor-specific effects
        n_factors = len(common_factors)
        fig = plt.figure(figsize=(18, 7 * n_factors)) # Increased size
        gs = gridspec.GridSpec(n_factors, 2, figure=fig, hspace=0.5, wspace=0.3) # Increased spacing

        # For each factor, create spectral and Molecular features visualizations
        for i, factor in enumerate(common_factors):
            print(f"Processing factor: {factor}")

            # Create subplots for this factor (spectral and Molecular features)
            ax_spectral = fig.add_subplot(gs[i, 0])
            ax_molecular_features = fig.add_subplot(gs[i, 1])

            # Generate factor-specific plots
            self._plot_spectral_factor_effect(factor, ax_spectral)
            self._plot_molecular_features_factor_effect(factor, ax_molecular_features)

            # Add a row title for this factor - Adjusted position
            # Calculate y position based on GridSpec
            row_top = 1.0 - (i / n_factors) * (1 - gs.hspace / (7 * n_factors / 18)) # Approximate
            fig.text(0.5, row_top - 0.02 / n_factors , f"Factor: {factor}",
                     ha='center', va='bottom',
                     fontsize=self.FONTS['panel_label'], fontweight='bold', color=self.COLORS['Text_Dark'])

        # Overall title and layout adjustment
        plt.suptitle("Factor-Specific Augmentation Effect Analysis",
                     fontsize=self.FONTS['main_title'], y=1.01, color=self.COLORS['Text_Dark'])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjusted rect for better spacing

        # Save the figure
        output_path = os.path.join(self.output_dir, 'factor_specific_effect.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        # Also save as PDF for publication quality
        pdf_path = os.path.join(self.output_dir, 'factor_specific_effect.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())

        end_time = time.time()
        print(f"Factor-specific visualization completed in {end_time - start_time:.2f} seconds")
        print(f"Saved to {output_path}")

    def _plot_spectral_factor_effect(self, factor, ax):
        """
        Create visualization for how a specific factor is preserved in spectral data.
        """
        try:
            factor_levels = sorted(self.spectral_original[factor].unique())
        except TypeError: # Handle mixed types if they occur
             factor_levels = sorted([str(f) for f in self.spectral_original[factor].unique()])
             self.spectral_original[factor] = self.spectral_original[factor].astype(str)
             self.spectral_augmented[factor] = self.spectral_augmented[factor].astype(str)


        if len(factor_levels) < 2:
            ax.text(0.5, 0.5, f"Only one level for factor '{factor}'",
                    ha='center', va='center', fontsize=self.FONTS['annotation'], color=self.COLORS['Text_Dark'])
            ax.set_title(f"Spectral Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            return

        # Subsample wavelengths for clearer visualization
        sampled_indices = np.linspace(0, len(self.wavelength_cols)-1, 100, dtype=int)
        wavelength_subset = [self.wavelength_cols[i] for i in sampled_indices]
        wavelength_values = self.wavelengths[sampled_indices]

        # Create a colormap for factor levels
        level_cmap = plt.cm.get_cmap('tab10', len(factor_levels)) # Use tab10 for distinct colors
        level_colors = [level_cmap(i) for i in range(len(factor_levels))]

        # Plot original data with solid lines
        for i, level in enumerate(factor_levels):
            level_data = self.spectral_original[self.spectral_original[factor] == level]
            if len(level_data) > 0:
                mean_spectrum = level_data[wavelength_subset].mean().values
                std_spectrum = level_data[wavelength_subset].std().values
                lower = mean_spectrum - std_spectrum
                upper = mean_spectrum + std_spectrum
                ax.plot(wavelength_values, mean_spectrum, '-', color=level_colors[i],
                        linewidth=2, label=f"Orig: {level} (n={len(level_data)})")
                ax.fill_between(wavelength_values, lower, upper, color=level_colors[i], alpha=0.2)

        # Plot augmented data with dashed lines
        augmented_only = self.spectral_augmented[~self.spectral_augmented['Row_names'].isin(self.spectral_original['Row_names'])]
        if factor in augmented_only.columns:
            augmented_only[factor] = augmented_only[factor].astype(self.spectral_original[factor].dtype) # Ensure same type


            for i, level in enumerate(factor_levels):
                level_data_aug = augmented_only[augmented_only[factor] == level]
                if len(level_data_aug) > 0:
                    mean_spectrum = level_data_aug[wavelength_subset].mean().values
                    std_spectrum = level_data_aug[wavelength_subset].std().values
                    lower = mean_spectrum - std_spectrum
                    upper = mean_spectrum + std_spectrum
                    ax.plot(wavelength_values, mean_spectrum, '--', color=level_colors[i],
                            linewidth=2, label=f"Aug: {level} (n={len(level_data_aug)})")
                    ax.fill_between(wavelength_values, lower, upper, color=level_colors[i], alpha=0.1)

        # Calculate divergence metrics (if augmented data exists)
        divergence_metrics = {}
        if not augmented_only.empty:
            def calculate_level_divergence(level, wl):
                orig_data = self.spectral_original[self.spectral_original[factor] == level][wl].values
                aug_data = augmented_only[augmented_only[factor] == level][wl].values

                if len(orig_data) < 2 or len(aug_data) < 2: return None
                try:
                    bins = min(20, max(5, int(min(len(orig_data), len(aug_data)) / 5)))
                    hist_range = (min(np.min(orig_data), np.min(aug_data)),
                                 max(np.max(orig_data), np.max(aug_data)))
                    orig_hist, _ = np.histogram(orig_data, bins=bins, range=hist_range, density=True)
                    aug_hist, _ = np.histogram(aug_data, bins=bins, range=hist_range, density=True)
                    orig_hist = np.maximum(orig_hist, 1e-10) / np.sum(np.maximum(orig_hist, 1e-10))
                    aug_hist = np.maximum(aug_hist, 1e-10) / np.sum(np.maximum(aug_hist, 1e-10))
                    return jensenshannon(orig_hist, aug_hist)
                except Exception as e:
                    # print(f"Warning: JS divergence failed for level {level}, wl {wl}: {e}")
                    return None

            for level in factor_levels:
                 if level in augmented_only[factor].unique():
                    divergence_wavelengths = np.random.choice(self.wavelength_cols, min(20, len(self.wavelength_cols)), replace=False)
                    results = Parallel(n_jobs=-1)(
                        delayed(calculate_level_divergence)(level, wl) for wl in divergence_wavelengths
                    )
                    valid_results = [r for r in results if r is not None and not np.isnan(r)]
                    if valid_results:
                        divergence_metrics[level] = np.mean(valid_results)

            # Add divergence metrics as text annotation
            if divergence_metrics:
                divergence_text = "JS Divergence (Orig vs Aug):\n" + "\n".join([f"{level}: {value:.3f}" for level, value in divergence_metrics.items()])
                ax.text(0.02, 0.98, divergence_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        fontsize=self.FONTS['annotation'] - 2, # Smaller font for annotation
                        color=self.COLORS['Text_Annotation'],
                        bbox={'facecolor': self.COLORS['Annotation_Box_BG'],
                              'alpha': 0.7, 'pad': 3,
                              'edgecolor': self.COLORS['Annotation_Box_Edge']})

        # Styling
        ax.set_title(f"Spectral Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel("Wavelength (nm)", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.set_ylabel("Reflectance", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.grid(True, alpha=0.3, color=self.COLORS['Grid'])
        ax.legend(loc='lower right', fontsize=self.FONTS['legend_text'] - 4, ncol=2) # Smaller legend
        ax.tick_params(axis='both', which='major', labelsize=self.FONTS['tick_label']-2)

    def _plot_molecular_features_factor_effect(self, factor, ax):
        """
        Create visualization for how a specific factor is preserved in Molecular features data.
        """
        try:
            factor_levels = sorted(self.molecular_features_original[factor].unique())
        except TypeError: # Handle mixed types
             factor_levels = sorted([str(f) for f in self.molecular_features_original[factor].unique()])
             self.molecular_features_original[factor] = self.molecular_features_original[factor].astype(str)
             self.molecular_features_augmented[factor] = self.molecular_features_augmented[factor].astype(str)

        if len(factor_levels) < 2:
            ax.text(0.5, 0.5, f"Only one level for factor '{factor}'",
                    ha='center', va='center', fontsize=self.FONTS['annotation'], color=self.COLORS['Text_Dark'])
            ax.set_title(f"Metabolite Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            return

        # Identify top differentiating metabolites for this factor
        if len(self.metabolite_cols) > 20:
            top_metabolites = self._identify_top_metabolites(factor, 20)
        else:
            top_metabolites = self.metabolite_cols

        if not top_metabolites:
             ax.text(0.5, 0.5, f"No metabolite data for factor '{factor}'",
                    ha='center', va='center', fontsize=self.FONTS['annotation'], color=self.COLORS['Text_Dark'])
             ax.set_title(f"Metabolite Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
             return

        # Calculate mean values for heatmap
        heatmap_data = []
        for level in factor_levels:
            orig_level_data = self.metabolite_original[self.metabolite_original[factor] == level]
            if not orig_level_data.empty:
                for met in top_metabolites:
                    if met in orig_level_data.columns:
                        heatmap_data.append({
                            'Factor_Level': f"{level} (O)",
                            'Metabolite': met.split('_')[-1] if '_' in met else met,
                            'Mean_Abundance': orig_level_data[met].mean()
                        })

        augmented_only = self.metabolite_augmented[~self.metabolite_augmented['Row_names'].isin(self.metabolite_original['Row_names'])]
        if factor in augmented_only.columns:
             augmented_only[factor] = augmented_only[factor].astype(self.metabolite_original[factor].dtype) # Ensure same type

             for level in factor_levels:
                aug_level_data = augmented_only[augmented_only[factor] == level]
                if not aug_level_data.empty:
                    for met in top_metabolites:
                         if met in aug_level_data.columns:
                             heatmap_data.append({
                                'Factor_Level': f"{level} (A)",
                                'Metabolite': met.split('_')[-1] if '_' in met else met,
                                'Mean_Abundance': aug_level_data[met].mean()
                            })

        # Convert to DataFrame and pivot for heatmap
        if not heatmap_data:
            ax.text(0.5, 0.5, "No data for heatmap",
                    ha='center', va='center', fontsize=self.FONTS['annotation'], color=self.COLORS['Text_Dark'])
            ax.set_title(f"Metabolite Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            return

        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_data = heatmap_df.pivot(index='Factor_Level', columns='Metabolite', values='Mean_Abundance')
        pivot_data = pivot_data.dropna(axis=1, how='all').dropna(axis=0, how='all') # Drop empty cols/rows

        if pivot_data.empty:
            ax.text(0.5, 0.5, "Insufficient data for heatmap",
                    ha='center', va='center', fontsize=self.FONTS['annotation'], color=self.COLORS['Text_Dark'])
        else:
            # Plot heatmap
            sns.heatmap(pivot_data, cmap='viridis', ax=ax,
                        cbar_kws={'label': 'Mean Abundance'},
                        linewidths=0.5, linecolor=self.COLORS['Grid'])
            ax.figure.axes[-1].yaxis.label.set_size(self.FONTS['axis_label']) # Cbar label size
            ax.tick_params(axis='x', labelsize=self.FONTS['tick_label'] - 6) # Smaller ticks if many metabolites
            ax.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 4, rotation=0)

            # Calculate preservation metrics
            preservation_metrics = {}
            if not augmented_only.empty:
                for level in factor_levels:
                    if f"{level} (O)" in pivot_data.index and f"{level} (A)" in pivot_data.index:
                        orig_values = pivot_data.loc[f"{level} (O)"].dropna().values
                        aug_values = pivot_data.loc[f"{level} (A)"].dropna().values

                        if len(orig_values) < 2 or len(aug_values) < 2: continue

                        # Align values based on common metabolites if necessary (though pivot should handle this)
                        common_mets = pivot_data.columns[~pivot_data.loc[[f"{level} (O)", f"{level} (A)"]].isnull().any()]
                        orig_values_aligned = pivot_data.loc[f"{level} (O)", common_mets].values
                        aug_values_aligned = pivot_data.loc[f"{level} (A)", common_mets].values

                        if len(orig_values_aligned) < 2: continue

                        # Calculate Spearman correlation (robust)
                        corr, _ = spearmanr(orig_values_aligned, aug_values_aligned)

                        # Calculate JS divergence
                        try:
                            combined_range = (min(np.min(orig_values_aligned), np.min(aug_values_aligned)),
                                            max(np.max(orig_values_aligned), np.max(aug_values_aligned)))
                            bins = min(10, max(3, len(orig_values_aligned) // 2)) # Fewer bins
                            orig_hist, _ = np.histogram(orig_values_aligned, bins=bins, range=combined_range, density=True)
                            aug_hist, _ = np.histogram(aug_values_aligned, bins=bins, range=combined_range, density=True)
                            orig_hist = np.maximum(orig_hist, 1e-10) / np.sum(np.maximum(orig_hist, 1e-10))
                            aug_hist = np.maximum(aug_hist, 1e-10) / np.sum(np.maximum(aug_hist, 1e-10))
                            js_div = jensenshannon(orig_hist, aug_hist)
                        except Exception:
                            js_div = np.nan

                        preservation_metrics[level] = {
                            'Spearman_r': corr,
                            'JS_Divergence': js_div
                        }

                # Add preservation metrics as text annotation
                if preservation_metrics:
                    metrics_text = "Preservation (Orig vs Aug):\n"
                    metrics_text += "\n".join([f"{level}: r={metrics['Spearman_r']:.2f}, JS={metrics['JS_Divergence']:.2f}"
                                            for level, metrics in preservation_metrics.items() if not np.isnan(metrics['Spearman_r'])])

                    ax.text(0.5, -0.25, metrics_text, transform=ax.transAxes, # Adjusted position
                            verticalalignment='top', horizontalalignment='center',
                            fontsize=self.FONTS['annotation'] - 2, # Smaller font
                            color=self.COLORS['Text_Annotation'],
                            bbox={'facecolor': self.COLORS['Annotation_Box_BG'],
                                  'alpha': 0.8, 'pad': 3,
                                  'edgecolor': self.COLORS['Annotation_Box_Edge']})

        # Styling
        ax.set_title(f"Metabolite Data: {factor} Effect", fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel("Metabolite", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.set_ylabel("Factor Level (O=Orig, A=Aug)", fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])

    def _identify_top_metabolites(self, factor, n_top=20):
        """
        Identify top metabolites that differentiate between factor levels using F-statistic.
        Handles potential non-numeric or missing data.
        """
        if factor not in self.metabolite_original.columns or not self.metabolite_cols:
            return []

        # Ensure factor column is suitable
        if self.metabolite_original[factor].isnull().any():
             print(f"Warning: Factor '{factor}' contains NaN values. Dropping NaNs for F-stat calc.")
             metabolite_data = self.metabolite_original.dropna(subset=[factor])
        else:
            metabolite_data = self.metabolite_original

        factor_levels = metabolite_data[factor].unique()

        if len(factor_levels) < 2:
            # Return first n metabolites if only one level
            return self.metabolite_cols[:n_top]

        f_stats = {}
        for metabolite in self.metabolite_cols:
            # Check if metabolite column exists and is numeric
            if metabolite not in metabolite_data.columns or not pd.api.types.is_numeric_dtype(metabolite_data[metabolite]):
                continue

            # Extract values, dropping NaNs within the metabolite column for this calculation
            level_values = [metabolite_data[metabolite_data[factor] == level][metabolite].dropna().values
                            for level in factor_levels]
            level_values = [vals for vals in level_values if len(vals) >= 2] # Need at least 2 points per group

            if len(level_values) < 2: # Need at least 2 groups
                continue

            try:
                # ANOVA F-statistic calculation (simplified version)
                means = [np.mean(vals) for vals in level_values]
                counts = [len(vals) for vals in level_values]
                total_count = sum(counts)
                grand_mean = np.sum([m * c for m, c in zip(means, counts)]) / total_count

                ss_between = sum(c * (m - grand_mean)**2 for m, c in zip(means, counts))
                df_between = len(level_values) - 1

                ss_within = sum(np.sum((vals - m)**2) for vals, m in zip(level_values, means))
                df_within = total_count - len(level_values)

                if df_between <= 0 or df_within <= 0: continue

                ms_between = ss_between / df_between
                ms_within = ss_within / df_within

                if ms_within == 0: # Avoid division by zero; indicates perfect separation or constant values within groups
                    f_stat = np.inf if ms_between > 0 else 0
                else:
                    f_stat = ms_between / ms_within

                if not np.isnan(f_stat):
                    f_stats[metabolite] = f_stat

            except Exception as e:
                # print(f"Warning: F-stat calculation failed for metabolite {metabolite}, factor {factor}: {e}")
                continue

        # Sort by F-statistic (higher is better)
        sorted_metabolites = sorted(f_stats.items(), key=lambda x: x[1], reverse=True)

        # Return top n metabolites
        top_metabolites = [m for m, _ in sorted_metabolites[:n_top]]

        # If we don't have enough, fill in with remaining metabolites
        if len(top_metabolites) < n_top:
            remaining = [m for m in self.metabolite_cols if m not in top_metabolites and m in f_stats]
            top_metabolites.extend(remaining[:n_top - len(top_metabolites)])
            # If still not enough, add any metabolites left
            if len(top_metabolites) < n_top:
                 remaining_any = [m for m in self.metabolite_cols if m not in top_metabolites]
                 top_metabolites.extend(remaining_any[:n_top - len(top_metabolites)])


        return top_metabolites

    def _create_spectral_signature_figure(self):
        """
        Create publication-quality spectral signature comparison plot,
        using median and IQR for non-parametric representation.
        """
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111)

        # Define key spectral regions
        regions = {
            'Blue': (400, 500), 'Green': (500, 600), 'Red': (600, 700),
            'NIR': (700, 1100), 'SWIR1': (1100, 1800), 'SWIR2': (1800, 2500)
        }

        # Get spectral data
        orig_spectra = self.spectral_original[self.wavelength_cols].values
        aug_spectra = self.spectral_augmented_only[self.wavelength_cols].values if not self.spectral_augmented_only.empty else np.array([])

        # Use median and IQR
        orig_median = np.median(orig_spectra, axis=0)
        orig_q25 = np.percentile(orig_spectra, 25, axis=0)
        orig_q75 = np.percentile(orig_spectra, 75, axis=0)

        # Subsample for plotting
        subsample_factor = max(1, len(self.wavelength_cols) // 500)
        wavelength_indices = np.arange(0, len(self.wavelength_cols), subsample_factor)
        wavelength_values = self.wavelengths[wavelength_indices]

        # Plot original data
        ax.plot(wavelength_values, orig_median[wavelength_indices],
                color=self.plot_colors['original'], linestyle='-',
                linewidth=2.5, label=f'Original (n={len(orig_spectra)})')
        ax.fill_between(wavelength_values,
                        orig_q25[wavelength_indices],
                        orig_q75[wavelength_indices],
                        color=self.plot_colors['original'], alpha=0.3, label='Original IQR (25-75%)')

        # Plot augmented data (if available)
        if len(aug_spectra) > 0:
            aug_median = np.median(aug_spectra, axis=0)
            aug_q25 = np.percentile(aug_spectra, 25, axis=0)
            aug_q75 = np.percentile(aug_spectra, 75, axis=0)

            ax.plot(wavelength_values, aug_median[wavelength_indices],
                    color=self.plot_colors['augmented'], linestyle='-', # Solid line for augmented median as well
                    linewidth=2.5, label=f'Augmented (n={len(aug_spectra)})')
            ax.fill_between(wavelength_values,
                            aug_q25[wavelength_indices],
                            aug_q75[wavelength_indices],
                            color=self.plot_colors['augmented'], alpha=0.3, label='Augmented IQR (25-75%)')

        # Highlight spectral regions
        ymin, ymax = ax.get_ylim()
        for region_name, (start, end) in regions.items():
            ax.axvspan(start, end, alpha=0.08, color=self.COLORS['Grid']) # Use grid color for subtle background
            region_middle = start + (end - start) * 0.5
            # Add text label slightly above bottom
            ax.text(region_middle, ymin + (ymax - ymin) * 0.03, region_name,
                    ha='center', va='bottom', fontsize=self.FONTS['annotation'] - 2,
                    color=self.COLORS['Text_Dark'],
                    bbox=dict(facecolor=self.COLORS['Annotation_Box_BG'], alpha=0.6,
                              boxstyle='round,pad=0.2', edgecolor='none'))

        # Add statistical comparison (if augmented data exists)
        if len(aug_spectra) > 0:
            key_wavelengths = [450, 550, 680, 800, 1650, 2200]
            comparison_text = "Statistical Comparison (Mann-Whitney U, Orig vs Aug):\n"
            valid_comparisons = 0
            for wl in key_wavelengths:
                wl_idx = np.argmin(np.abs(self.wavelengths - wl))
                wl_actual = self.wavelengths[wl_idx]
                orig_values = orig_spectra[:, wl_idx]
                aug_values = aug_spectra[:, wl_idx]

                if len(orig_values) >= 3 and len(aug_values) >= 3: # Need enough samples for test
                    try:
                        u_stat, p_value = mannwhitneyu(orig_values, aug_values, alternative='two-sided')
                        sig = '*' if p_value < 0.05 else 'ns'
                        comparison_text += f"{int(wl_actual)}nm: p={p_value:.2e} {sig}\n"
                        valid_comparisons += 1
                    except ValueError: # Handle cases with constant data
                        comparison_text += f"{int(wl_actual)}nm: const data\n"
                else:
                     comparison_text += f"{int(wl_actual)}nm: N/A\n"

            # Add text box only if comparisons were made
            if valid_comparisons > 0:
                plt.figtext(0.15, 0.05, comparison_text.strip(),
                            fontsize=self.FONTS['annotation'] - 1,
                            color=self.COLORS['Text_Annotation'],
                            bbox=dict(facecolor=self.COLORS['Annotation_Box_BG'], alpha=0.8,
                                      boxstyle='round,pad=0.5', edgecolor=self.COLORS['Annotation_Box_Edge']))

        # Styling
        ax.set_title('Spectral Signature Comparison: Original vs. Augmented Data',
                     fontsize=self.FONTS['panel_title'], fontweight='bold', color=self.COLORS['Text_Dark'])
        ax.set_xlabel('Wavelength (nm)', fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.set_ylabel('Reflectance (Median)', fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.grid(True, alpha=0.4, color=self.COLORS['Grid'])
        ax.legend(loc='upper right', fontsize=self.FONTS['legend_text'] - 2)
        ax.tick_params(axis='both', which='major', labelsize=self.FONTS['tick_label'])

        # Add annotation about non-parametric analysis
        plt.figtext(0.5, 0.01,
                    "Note: Lines represent Median; shaded areas represent Interquartile Range (IQR, 25-75%)",
                    ha='center', fontsize=self.FONTS['caption'] - 1, style='italic', color=self.COLORS['Text_Dark'])

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the plot
        output_path = os.path.join(self.output_dir, 'publication_spectral_signatures.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'publication_spectral_signatures.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    def _create_metabolite_profile_figure(self):
        """
        Create publication-quality metabolite profile comparison using boxplots.

        Note: A table component was previously part of this figure but has been
        removed to prevent overlap issues in the plot.
        """
        # Identify the most informative metabolites to visualize (differential between orig/aug)
        metabolites_to_show = self._identify_differential_metabolites(20)

        if not metabolites_to_show or self.metabolite_augmented_only.empty:
             print("Skipping metabolite profile figure: No differential metabolites found or no augmented data.")
             # Optionally create an empty placeholder plot
             fig, ax = plt.subplots(figsize=(15, 6)) # Adjusted size
             ax.text(0.5, 0.5, "Metabolite profile data not available\\nor insufficient for comparison.",
                     ha='center', va='center', fontsize=self.FONTS['panel_title'])
             ax.set_title("Metabolite Profile Comparison", fontsize=self.FONTS['panel_title'])
             ax.axis('off')
             output_path = os.path.join(self.output_dir, 'publication_metabolite_profiles.png')
             plt.savefig(output_path, dpi=300, bbox_inches='tight')
             pdf_path = os.path.join(self.output_dir, 'publication_metabolite_profiles.pdf')
             plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
             plt.close(fig)
             return None


        # Create a single-panel figure for the boxplot
        fig, ax1 = plt.subplots(figsize=(18, 9)) # Adjusted figure size for single panel

        # 1. Create boxplot comparison
        metabolite_data = []
        for metabolite in metabolites_to_show:
            if metabolite not in self.metabolite_original.columns or metabolite not in self.metabolite_augmented_only.columns:
                continue # Skip if metabolite missing in either dataset

            orig_values = self.metabolite_original[metabolite].dropna().values
            aug_values = self.metabolite_augmented_only[metabolite].dropna().values
            met_label = metabolite.split('_')[-1] if '_' in metabolite else metabolite # Clean label

            for val in orig_values:
                metabolite_data.append({'Metabolite': met_label, 'Value': val, 'Source': 'Original'})
            if len(aug_values) > 0:
                for val in aug_values:
                    metabolite_data.append({'Metabolite': met_label, 'Value': val, 'Source': 'Augmented'})

        if not metabolite_data:
            ax1.text(0.5, 0.5, "No data for boxplot.", ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax1.set_title('Metabolite Profile Comparison (Original vs. Augmented)', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax1.axis('off') # Turn off axis if no data
        else:
            boxplot_df = pd.DataFrame(metabolite_data)
            # Get unique metabolite labels in the order they appear for consistent plotting
            ordered_metabolites = boxplot_df['Metabolite'].unique()

            sns.boxplot(x='Metabolite', y='Value', hue='Source',
                        data=boxplot_df, ax=ax1, order=ordered_metabolites,
                        palette={'Original': self.plot_colors['original'],
                                 'Augmented': self.plot_colors['augmented']},
                        fliersize=2) # Smaller outlier markers

            ax1.tick_params(axis='x', rotation=45, labelsize=self.FONTS['tick_label'] - 2)
            ax1.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 2)
            ax1.set_xlabel("Top Differential Metabolites", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
            ax1.set_ylabel("Abundance", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
            ax1.set_title('Metabolite Profile Comparison (Original vs. Augmented)', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax1.legend(title='Data Source', title_fontsize=self.FONTS['legend_title']-2, fontsize=self.FONTS['legend_text']-2)
            ax1.grid(axis='y', alpha=0.4, color=self.COLORS['Grid'])

        # Section for table removed

        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust layout

        # Save the plot
        output_path = os.path.join(self.output_dir, 'publication_metabolite_profiles.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'publication_metabolite_profiles.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    def _identify_differential_metabolites(self, n_top=20):
        """
        Identify metabolites showing the most difference between original and augmented data
        using Mann-Whitney U effect size (Rank-Biserial Correlation).
        """
        if self.metabolite_augmented_only.empty or not self.metabolite_cols:
             return []

        diff_metrics = {}
        for metabolite in self.metabolite_cols:
            if metabolite not in self.metabolite_original.columns or metabolite not in self.metabolite_augmented_only.columns:
                 continue

            orig_values = self.metabolite_original[metabolite].dropna().values
            aug_values = self.metabolite_augmented_only[metabolite].dropna().values

            # Need sufficient samples for a meaningful comparison
            if len(orig_values) < 5 or len(aug_values) < 5:
                continue

            try:
                u_stat, p_value = mannwhitneyu(orig_values, aug_values, alternative='two-sided')
                n1, n2 = len(orig_values), len(aug_values)
                # Rank-Biserial Correlation (effect size for MWU)
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                diff_metrics[metabolite] = abs(effect_size) # Use absolute effect size for ranking

            except ValueError: # Handle constant data or other issues
                continue
            except Exception as e:
                # print(f"Warning: Differential metabolite calculation failed for {metabolite}: {e}")
                continue

        # Rank metabolites by absolute effect size (largest difference first)
        sorted_metabolites = sorted(diff_metrics.items(), key=lambda x: x[1], reverse=True)

        # Return top n metabolite names
        top_metabolites = [m for m, _ in sorted_metabolites[:n_top]]

         # Ensure we return exactly n_top if possible, padding if needed
        if len(top_metabolites) < n_top:
            remaining = [m for m in self.metabolite_cols if m not in top_metabolites and m in diff_metrics]
            top_metabolites.extend(remaining[:n_top - len(top_metabolites)])
        if len(top_metabolites) < n_top:
            remaining_any = [m for m in self.metabolite_cols if m not in top_metabolites]
            top_metabolites.extend(remaining_any[:n_top - len(top_metabolites)])


        return top_metabolites

    def create_parallel_coordinate_plot(self):
        """
        Creates a parallel coordinate plot comparing augmentation methods across key metrics.
        """
        print("Generating parallel coordinate plot...")

        # Calculate metrics for each method and modality
        metrics_data = []

        # --- Distribution Metrics ---
        dist_metrics_spec = self._calculate_distribution_metrics('spectral')
        dist_metrics_mol = self._calculate_distribution_metrics('molecular_features')
        for method, score in dist_metrics_spec.items():
            metrics_data.append({'Method': method, 'Type': 'Spectral', 'Metric': 'Distribution', 'Score': score})
        for method, score in dist_metrics_mol.items():
            metrics_data.append({'Method': method, 'Type': 'Molecular Features', 'Metric': 'Distribution', 'Score': score})

        # --- Structure Metrics ---
        struct_metrics_spec = self._calculate_structure_metrics('spectral')
        struct_metrics_mol = self._calculate_structure_metrics('molecular_features')
        for method, score in struct_metrics_spec.items():
            metrics_data.append({'Method': method, 'Type': 'Spectral', 'Metric': 'Structure', 'Score': score})
        for method, score in struct_metrics_mol.items():
             metrics_data.append({'Method': method, 'Type': 'Molecular Features', 'Metric': 'Structure', 'Score': score})

        # --- Biological Metrics ---
        bio_metrics_spec = self._calculate_biological_metrics('spectral')
        bio_metrics_mol = self._calculate_biological_metrics('molecular_features')
        for method, score in bio_metrics_spec.items():
            metrics_data.append({'Method': method, 'Type': 'Spectral', 'Metric': 'Biological', 'Score': score})
        for method, score in bio_metrics_mol.items():
            metrics_data.append({'Method': method, 'Type': 'Molecular Features', 'Metric': 'Biological', 'Score': score})

        if not metrics_data:
            print("No metrics calculated, cannot create parallel coordinate plot.")
            # Create empty placeholder plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "No augmentation methods or metrics\navailable for comparison.",
                     ha='center', va='center', fontsize=self.FONTS['panel_title'])
            ax.set_title("Comparison of Augmentation Methods", fontsize=self.FONTS['main_title'])
            ax.axis('off')
            output_path = os.path.join(self.output_dir, 'parallel_coordinates.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pdf_path = os.path.join(self.output_dir, 'parallel_coordinates.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
            return

        # Create DataFrame and pivot
        df_long = pd.DataFrame(metrics_data)
        df_wide = df_long.pivot_table(index=['Method', 'Type'], columns='Metric', values='Score').reset_index()
        df_wide = df_wide.fillna(0.5) # Fill missing metrics with a neutral score

        # Define colors for each method based on type
        method_colors = {}
        color_map_spec = plt.cm.get_cmap('Blues', len(self.spectral_methods) + 2) # More distinct blues
        color_map_mol = plt.cm.get_cmap('Greens', len(self.molecular_features_methods) + 2) # More distinct greens

        spec_methods = sorted(df_wide[df_wide['Type'] == 'Spectral']['Method'].unique())
        mol_methods = sorted(df_wide[df_wide['Type'] == 'Molecular Features']['Method'].unique())

        for i, method in enumerate(spec_methods):
            # Use COLORS if defined, otherwise use colormap
            method_colors[method] = self.COLORS.get(method, color_map_spec(i+1)) # Avoid lightest color
        for i, method in enumerate(mol_methods):
            method_colors[method] = self.COLORS.get(method, color_map_mol(i+1)) # Avoid lightest color


        # Create figure
        fig, ax = plt.subplots(figsize=(15, 9))

        # Create parallel coordinate plot
        pd.plotting.parallel_coordinates(
            df_wide, 'Method',
            cols=['Distribution', 'Structure', 'Biological'],
            color=[method_colors.get(method, self.COLORS['UnknownFeature']) for method in df_wide['Method']],
            linewidth=2.5, alpha=0.8, ax=ax
        )

        # Add a color legend for method types/methods
        legend_elements = []
        if spec_methods:
             legend_elements.append(mpatches.Patch(color=self.COLORS.get('Spectral', '#6baed6'), label='Spectral Methods:'))
             for method in spec_methods:
                 legend_elements.append(Line2D([0], [0], color=method_colors.get(method, self.COLORS['UnknownFeature']), lw=3, label=method))
        if mol_methods:
            legend_elements.append(mpatches.Patch(color=self.COLORS.get('Molecular features', '#41ab5d'), label='Molecular Features Methods:'))
            for method in mol_methods:
                 legend_elements.append(Line2D([0], [0], color=method_colors.get(method, self.COLORS['UnknownFeature']), lw=3, label=method))

        ax.legend(handles=legend_elements, title='Methods & Modality',
                  loc='upper right', bbox_to_anchor=(1.2, 1), # Move legend outside plot
                  fontsize=self.FONTS['legend_text'] - 2, title_fontsize=self.FONTS['legend_title'] - 2)


        # Styling
        ax.set_title('Comparison of Augmentation Methods Across Key Metrics',
                     fontsize=self.FONTS['main_title'], fontweight='bold', color=self.COLORS['Text_Dark'], pad=20)
        ax.set_ylabel('Normalized Score (0-1, higher is better)', fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
        ax.grid(True, alpha=0.4, color=self.COLORS['Grid'])
        ax.tick_params(axis='y', labelsize=self.FONTS['tick_label'])
        ax.tick_params(axis='x', labelsize=self.FONTS['tick_label'])

        # Adjust axis limits and labels
        ax.set_ylim(0, 1.05)
        # ax.xaxis.label.set_visible(False) # Hide x-axis label "Metric"

        # Add horizontal reference lines
        ax.axhline(y=0.9, color=self.COLORS.get('Positive_Diff', '#238b45'), linestyle='--', alpha=0.6, label='Excellent')
        ax.axhline(y=0.7, color=self.COLORS.get('T1', '#fe9929'), linestyle='--', alpha=0.6, label='Good')
        ax.axhline(y=0.5, color=self.COLORS.get('Negative_Diff', '#fe9929'), linestyle='--', alpha=0.6, label='Fair')
        # Add legend for reference lines manually if needed, or integrate above
        handles, labels = ax.get_legend_handles_labels() # Get method legend handles
        line_handles = [Line2D([0], [0], color=self.COLORS.get('Positive_Diff', '#238b45'), linestyle='--', alpha=0.6),
                        Line2D([0], [0], color=self.COLORS.get('T1', '#fe9929'), linestyle='--', alpha=0.6),
                        Line2D([0], [0], color=self.COLORS.get('Negative_Diff', '#fe9929'), linestyle='--', alpha=0.6)]
        line_labels = ['Excellent (≥0.9)', 'Good (≥0.7)', 'Fair (≥0.5)']
        ax.legend(handles=handles + line_handles, labels = labels + line_labels,
                  title='Methods & Quality', loc='center left', bbox_to_anchor=(1.05, 0.5),
                  fontsize=self.FONTS['legend_text'] - 2, title_fontsize=self.FONTS['legend_title'] - 2)


        # Add annotation explaining metrics
        metrics_explanation = (
            "Distribution: Preservation of value distributions (e.g., using JS divergence)\n"
            "Structure: Preservation of feature correlation structure (e.g., using Spearman corr.)\n"
            "Biological: Preservation of signals related to factors like Genotype/Tissue (e.g., using median profile corr.)"
        )
        plt.figtext(0.5, 0.01, metrics_explanation, ha='center',
                   fontsize=self.FONTS['caption'], color=self.COLORS['Text_Annotation'],
                   bbox=dict(facecolor=self.COLORS['Annotation_Box_BG'], alpha=0.8,
                             boxstyle='round,pad=0.5', edgecolor=self.COLORS['Annotation_Box_Edge']))

        plt.tight_layout(rect=[0, 0.05, 0.85, 0.95]) # Adjust layout to make space for legend and text

        # Save the plot
        output_path = os.path.join(self.output_dir, 'parallel_coordinates.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'parallel_coordinates.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"Parallel coordinate plot saved to: {output_path}")

    def _calculate_distribution_metrics(self, modality):
        """Calculate distribution preservation (1 - JS divergence) for each method"""
        metrics = {}
        original_data = self.spectral_original if modality == 'spectral' else self.molecular_features_original
        augmented_data = self.spectral_augmented if modality == 'spectral' else self.molecular_features_augmented
        feature_cols = self.wavelength_cols if modality == 'spectral' else self.molecular_features_cols
        methods = self.spectral_methods if modality == 'spectral' else self.molecular_features_methods

        if not feature_cols or not methods or augmented_data.empty: return {}

        # Sample features for efficiency
        sampled_cols = np.random.choice(feature_cols, min(50, len(feature_cols)), replace=False)

        for method in methods:
            method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]
            if method_data.empty: continue

            js_divergences = []
            for col in sampled_cols:
                if col not in original_data.columns or col not in method_data.columns: continue
                orig_values = original_data[col].dropna().values
                method_values = method_data[col].dropna().values

                if len(orig_values) < 5 or len(method_values) < 5: continue

                try:
                    hist_range = (min(np.min(orig_values), np.min(method_values)),
                                 max(np.max(orig_values), np.max(method_values)))
                    bins = min(20, max(5, int(min(len(orig_values), len(method_values)) / 5)))
                    orig_hist, _ = np.histogram(orig_values, bins=bins, range=hist_range, density=True)
                    method_hist, _ = np.histogram(method_values, bins=bins, range=hist_range, density=True)
                    orig_hist = np.maximum(orig_hist, 1e-10) / np.sum(np.maximum(orig_hist, 1e-10))
                    method_hist = np.maximum(method_hist, 1e-10) / np.sum(np.maximum(method_hist, 1e-10))
                    js_div = jensenshannon(orig_hist, method_hist)
                    if not np.isnan(js_div):
                         js_divergences.append(js_div)
                except Exception:
                    continue

            if js_divergences:
                mean_js = np.mean(js_divergences)
                 # JS divergence is between 0 and log(2)~0.693. Scale to 0-1.
                # Score = 1 - normalized JS divergence. Higher is better.
                # Let's use a simple linear scale: 1 - js/log(2)
                # Or scale more aggressively: 1 - min(1, mean_js * 2) to spread scores
                preservation_score = max(0, 1 - min(1, mean_js * 1.5)) # Scale to emphasize differences
                metrics[method] = preservation_score
            else:
                 metrics[method] = 0.5 # Default neutral score


        return metrics

    def _calculate_structure_metrics(self, modality):
        """Calculate correlation structure preservation (Spearman corr of corr matrices)"""
        metrics = {}
        original_data = self.spectral_original if modality == 'spectral' else self.molecular_features_original
        augmented_data = self.spectral_augmented if modality == 'spectral' else self.molecular_features_augmented
        feature_cols = self.wavelength_cols if modality == 'spectral' else self.molecular_features_cols
        methods = self.spectral_methods if modality == 'spectral' else self.molecular_features_methods

        if len(feature_cols) < 2 or not methods or augmented_data.empty: return {}

        # Sample features for efficiency
        n_sample = min(50, len(feature_cols))
        if n_sample < 2: return {}
        sampled_cols = np.random.choice(feature_cols, n_sample, replace=False)

        # Calculate original correlation matrix (using Spearman)
        try:
            # Drop non-numeric if any slipped through, handle NaNs
            orig_numeric = original_data[sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if len(orig_numeric) < 2: raise ValueError("Not enough numeric data in original")
            orig_corr_mat = orig_numeric.corr(method='spearman')
            orig_corr_vec = orig_corr_mat.values[np.triu_indices_from(orig_corr_mat, k=1)] # Upper triangle vector
            if np.isnan(orig_corr_vec).all(): raise ValueError("Original correlation vector is all NaN")
        except Exception as e:
            print(f"Warning: Could not calculate original correlation matrix for {modality}: {e}")
            return {method: 0.5 for method in methods} # Return default if original fails

        for method in methods:
            method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]
            if len(method_data) < 2:
                 metrics[method] = 0.0 # Penalize if too few samples
                 continue

            try:
                # Drop non-numeric, handle NaNs
                method_numeric = method_data[sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
                if len(method_numeric) < 2:
                    metrics[method] = 0.0
                    continue

                method_corr_mat = method_numeric.corr(method='spearman')
                method_corr_vec = method_corr_mat.values[np.triu_indices_from(method_corr_mat, k=1)]

                # Calculate Spearman correlation between the two correlation vectors
                valid_indices = ~np.isnan(orig_corr_vec) & ~np.isnan(method_corr_vec)
                if np.sum(valid_indices) < 2: # Need at least 2 pairs to correlate
                     metrics[method] = 0.5 # Neutral score if cannot compare
                     continue

                corr_similarity, _ = spearmanr(orig_corr_vec[valid_indices], method_corr_vec[valid_indices])

                # Normalize Spearman correlation (-1 to 1) to preservation score (0 to 1)
                preservation_score = (corr_similarity + 1) / 2
                metrics[method] = preservation_score if not np.isnan(preservation_score) else 0.5

            except Exception as e:
                 # print(f"Warning: Structure metric failed for {modality}, method {method}: {e}")
                 metrics[method] = 0.5 # Default neutral score

        return metrics

    def _calculate_biological_metrics(self, modality):
        """Calculate biological signal preservation based on median profile correlations across factor levels"""
        metrics = {}
        original_data = self.spectral_original if modality == 'spectral' else self.molecular_features_original
        augmented_data = self.spectral_augmented if modality == 'spectral' else self.molecular_features_augmented
        feature_cols = self.wavelength_cols if modality == 'spectral' else self.molecular_features_cols
        methods = self.spectral_methods if modality == 'spectral' else self.molecular_features_methods

        # Identify relevant common factors
        potential_factors = ['Treatment', 'Genotype', 'Tissue.type'] # Prioritize these
        common_factors = [f for f in potential_factors if f in self.common_metadata]
        if not common_factors:
             # Fallback to any common categorical factor
             common_factors = [f for f in self.common_metadata if original_data[f].dtype == 'object' or (original_data[f].nunique() < 10 and original_data[f].nunique() > 1)]

        if not common_factors or len(feature_cols) < 2 or not methods or augmented_data.empty:
            return {method: 0.5 for method in methods} # Default if no factors or data


        # Sample features for efficiency
        n_sample = min(50, len(feature_cols))
        if n_sample < 2: return {method: 0.5 for method in methods}
        sampled_cols = np.random.choice(feature_cols, n_sample, replace=False)

        for method in methods:
            method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]
            if method_data.empty:
                metrics[method] = 0.0
                continue

            factor_preservation_scores = []
            for factor in common_factors:
                 # Ensure factor is present in both datasets for this method
                 if factor not in original_data.columns or factor not in method_data.columns: continue

                 # Ensure factor types match
                 if original_data[factor].dtype != method_data[factor].dtype:
                      try:
                          method_data[factor] = method_data[factor].astype(original_data[factor].dtype)
                      except: continue # Skip factor if types can't be matched

                 factor_levels = sorted(original_data[factor].dropna().unique())
                 if len(factor_levels) < 2: continue # Need multiple levels

                 level_medians_orig = []
                 level_medians_method = []
                 valid_levels = []

                 for level in factor_levels:
                    level_data_orig = original_data[(original_data[factor] == level)][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
                    level_data_method = method_data[(method_data[factor] == level)][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()

                    # Need enough data in both original and augmented for this level
                    if len(level_data_orig) > 1 and len(level_data_method) > 1:
                         level_medians_orig.append(level_data_orig.median().values)
                         level_medians_method.append(level_data_method.median().values)
                         valid_levels.append(level)

                 # Need at least two valid levels to compare patterns
                 if len(valid_levels) < 2: continue

                 # Calculate Spearman correlation between the median profiles across levels
                 try:
                    # Stack medians into matrices (rows=levels, cols=features)
                    median_mat_orig = np.vstack(level_medians_orig)
                    median_mat_method = np.vstack(level_medians_method)

                    # Correlate the flattened matrices
                    corr, _ = spearmanr(median_mat_orig.flatten(), median_mat_method.flatten())
                    if not np.isnan(corr):
                         factor_preservation_scores.append((corr + 1) / 2) # Normalize to 0-1
                 except Exception:
                      continue # Skip factor if correlation fails


            # Average score across all valid factors for this method
            if factor_preservation_scores:
                metrics[method] = np.mean(factor_preservation_scores)
            else:
                metrics[method] = 0.5 # Default if no factors could be assessed

        return metrics

    def augmentation_quality_heatmaps(self):
        """Generate augmentation quality heatmaps (Feature not fully implemented)."""
        print("Generating augmentation quality heatmaps... (Note: This feature is not fully implemented)")
        # Current implementation is a pass-through.
        # Future development could involve heatmaps of metric scores
        # (e.g., JS divergence, correlation differences) for features
        # versus augmentation methods.
        pass

    def create_integrated_dashboard(self):
        """
        Create an integrated dashboard summarizing augmentation quality.
        """
        print("Generating integrated dashboard...")
        start_time = time.time()

        # Create figure with multiple panels
        fig = plt.figure(figsize=(22, 18)) # Larger figure size
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45) # Adjusted spacing

        # Panel 1: Sample Counts
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_method_count_panel(ax1)

        # Panel 2: Spectral Quality Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_spectral_quality_panel(ax2)

        # Panel 3: Molecular Features Quality Heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_molecular_features_quality_panel(ax3)

        # Panel 4: Distribution Preservation Scores
        ax4 = fig.add_subplot(gs[1, 0])
        self._create_distribution_panel(ax4)

        # Panel 5: Structure Preservation Scores
        ax5 = fig.add_subplot(gs[1, 1])
        self._create_structure_panel(ax5)

        # Panel 6: Biological Signal Preservation Scores
        ax6 = fig.add_subplot(gs[1, 2])
        self._create_biological_panel(ax6)

        # Panel 7: Factor Preservation Heatmap (spanning bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        self._create_factor_scores_panel(ax7)

        # Add overall title and metadata
        plt.suptitle("Integrated Augmentation Quality Dashboard",
                     fontsize=self.FONTS['main_title'], y=0.99, color=self.COLORS['Text_Dark'])
        fig.text(0.5, 0.01, f"Generated on {time.strftime('%Y-%m-%d')} | Original samples: Spectral={len(self.spectral_original)}, Molecular Features={len(self.molecular_features_original)}",
                 ha='center', fontsize=self.FONTS['caption'], color=self.COLORS['Text_Dark'])

        # Save the dashboard
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
        output_path = os.path.join(self.output_dir, 'integrated_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'integrated_dashboard.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        end_time = time.time()
        print(f"Integrated dashboard completed in {end_time - start_time:.2f} seconds")
        print(f"Saved to: {output_path}")

    def _create_method_count_panel(self, ax):
        """Create panel showing sample counts by method and type"""
        spectral_counts = {'Original': len(self.spectral_original)}
        for method in self.spectral_methods:
            count = len(self.spectral_augmented[self.spectral_augmented['Row_names'].str.contains(f'_{method}', regex=False, na=False)])
            spectral_counts[method] = count

        molecular_features_counts = {'Original': len(self.molecular_features_original)}
        for method in self.molecular_features_methods:
             count = len(self.molecular_features_augmented[self.molecular_features_augmented['Row_names'].str.contains(f'_{method}', regex=False, na=False)])
             molecular_features_counts[method] = count


        spectral_df = pd.DataFrame(list(spectral_counts.items()), columns=['Method', 'Count'])
        spectral_df['Type'] = 'Spectral'
        molecular_features_df = pd.DataFrame(list(molecular_features_counts.items()), columns=['Method', 'Count'])
        molecular_features_df['Type'] = 'Molecular Features'

        df = pd.concat([spectral_df, molecular_features_df])
        # Exclude methods with zero counts unless it's 'Original'
        df = df[(df['Count'] > 0) | (df['Method'] == 'Original')]

        if df.empty:
            ax.text(0.5, 0.5, "No count data", ha='center', va='center', fontsize=self.FONTS['annotation'])
        else:
            # Define palette based on type
            palette = {'Spectral': self.COLORS.get('Spectral', '#6baed6'),
                       'Molecular Features': self.COLORS.get('Molecular features', '#41ab5d')}

            sns.barplot(x='Method', y='Count', hue='Type', data=df, ax=ax, palette=palette, order=sorted(df['Method'].unique()))

            # Add value labels
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height + ax.get_ylim()[1]*0.01, f'{int(height)}',
                            ha='center', va='bottom', fontsize=self.FONTS['annotation'] - 4) # Smaller labels

            # Styling
            ax.set_title('Sample Counts by Method', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax.set_xlabel('Method', fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
            ax.set_ylabel('Number of Samples', fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
            # REMOVED ha='right' from the line below
            ax.tick_params(axis='x', rotation=45, labelsize=self.FONTS['tick_label'] - 4)
            ax.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 4)
            ax.legend(title='Data Type', fontsize=self.FONTS['legend_text'] - 4, title_fontsize=self.FONTS['legend_title'] - 4)
            ax.grid(axis='y', alpha=0.4, color=self.COLORS['Grid'])

    def _create_molecular_features_quality_panel(self, ax):
        """Create Molecular features quality heatmap panel"""
        self._create_quality_panel_base(ax, 'molecular_features')

    def _create_quality_panel_base(self, ax, modality):
        """Base function for creating spectral/metabolite quality panels"""
        methods = self.spectral_methods if modality == 'spectral' else self.metabolite_methods
        if not methods:
            ax.text(0.5, 0.5, f"No {modality} methods", ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax.set_title(f'{modality.capitalize()} Quality Metrics', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            return

        metrics = {}
        dist_m = self._calculate_distribution_metrics(modality)
        struct_m = self._calculate_structure_metrics(modality)
        bio_m = self._calculate_biological_metrics(modality)

        for method in methods:
            d = dist_m.get(method, 0.5)
            s = struct_m.get(method, 0.5)
            b = bio_m.get(method, 0.5)
            metrics[method] = {
                'Distribution': d,
                'Structure': s,
                'Biological': b,
                'Overall': np.mean([d, s, b])
            }

        df = pd.DataFrame(metrics).T # Transpose to get methods as rows
        df = df[['Distribution', 'Structure', 'Biological', 'Overall']] # Ensure column order

        if df.empty:
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', fontsize=self.FONTS['annotation'])
        else:
            sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, ax=ax, fmt=".2f",
                        linewidths=0.5, linecolor=self.COLORS['Grid'],
                        annot_kws={"size": self.FONTS['annotation'] - 4}, # Smaller annotation font
                        cbar_kws={'label': 'Quality Score (0-1)'})
            ax.figure.axes[-1].yaxis.label.set_size(self.FONTS['axis_label']-4) # Cbar label size
            ax.tick_params(axis='y', rotation=0, labelsize=self.FONTS['tick_label'] - 4)
            ax.tick_params(axis='x', labelsize=self.FONTS['tick_label'] - 4)


        ax.set_title(f'{modality.capitalize()} Augmentation Quality', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel("Metric", fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
        ax.set_ylabel("Method", fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])


    def _create_spectral_quality_panel(self, ax):
        """Create spectral quality heatmap panel"""
        self._create_quality_panel_base(ax, 'spectral')

    def _create_metabolite_quality_panel(self, ax):
        """Create metabolite quality heatmap panel"""
        self._create_quality_panel_base(ax, 'metabolite')

    def _create_metric_bar_panel(self, ax, metric_name, metric_func_suffix):
        """Base function for creating Distribution, Structure, Biological bar chart panels"""
        spectral_metrics = getattr(self, f'_calculate_{metric_func_suffix}_metrics')('spectral')
        metabolite_metrics = getattr(self, f'_calculate_{metric_func_suffix}_metrics')('metabolite')

        data = []
        for method, score in spectral_metrics.items():
            data.append({'Method': method, 'Type': 'Spectral', 'Score': score})
        for method, score in metabolite_metrics.items():
            data.append({'Method': method, 'Type': 'Metabolite', 'Score': score})

        if not data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax.set_title(f'{metric_name} Preservation', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark']) # Set title even if no data
        else:
            df = pd.DataFrame(data)
            df = df.sort_values('Score', ascending=False)
             # Define palette based on type
            palette = {'Spectral': self.COLORS.get('Spectral', 'blue'),
                       'Metabolite': self.COLORS.get('Metabolite', 'green')}

            sns.barplot(x='Method', y='Score', hue='Type', data=df, ax=ax, palette=palette,
                        order=df['Method'].unique()) # Use sorted order

            # Add value labels
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{height:.2f}',
                            ha='center', va='bottom', fontsize=self.FONTS['annotation'] - 5) # Even smaller

            # Styling
            ax.set_title(f'{metric_name} Preservation', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax.set_xlabel('Method', fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
            ax.set_ylabel('Score (0-1)', fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
            ax.set_ylim(0, 1.1)
            # REMOVED ha='right' from the line below
            ax.tick_params(axis='x', rotation=45, labelsize=self.FONTS['tick_label'] - 4)
            ax.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 4)
            ax.legend(title='Data Type', fontsize=self.FONTS['legend_text'] - 4, title_fontsize=self.FONTS['legend_title'] - 4)
            ax.grid(axis='y', alpha=0.4, color=self.COLORS['Grid'])

    def _create_distribution_panel(self, ax):
        """Create panel showing distribution preservation"""
        self._create_metric_bar_panel(ax, 'Distribution', 'distribution')

    def _create_structure_panel(self, ax):
        """Create panel showing correlation structure preservation"""
        self._create_metric_bar_panel(ax, 'Structure', 'structure')

    def _create_biological_panel(self, ax):
        """Create panel showing biological signal preservation"""
        self._create_metric_bar_panel(ax, 'Biological Signal', 'biological')

    def _create_factor_scores_panel(self, ax):
        """Create heatmap panel showing preservation scores by experimental factor"""
        # Get common factors with multiple levels
        potential_factors = ['Treatment', 'Genotype', 'Batch', 'Day', 'Tissue.type']
        common_factors = [f for f in potential_factors if f in self.common_metadata and self.spectral_original[f].nunique() > 1]
        if not common_factors:
             common_factors = [f for f in self.common_metadata if self.spectral_original[f].nunique() > 1 and self.spectral_original[f].nunique() < 10]


        if not common_factors:
            ax.text(0.5, 0.5, "No common factors found\nfor preservation analysis",
                    ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax.set_title('Factor Preservation by Method and Modality', fontsize=self.FONTS['panel_title'])
            return

        factor_scores_data = []

        # Helper to get score for a specific factor/method/modality
        def get_factor_preservation_score(modality, method, factor):
            original_data = self.spectral_original if modality == 'Spectral' else self.metabolite_original
            augmented_data = self.spectral_augmented if modality == 'Spectral' else self.metabolite_augmented
            feature_cols = self.wavelength_cols if modality == 'Spectral' else self.metabolite_cols
            method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]

            if method_data.empty or factor not in original_data.columns or factor not in method_data.columns or len(feature_cols) < 2:
                return 0.5 # Default neutral score

             # Ensure factor types match
            if original_data[factor].dtype != method_data[factor].dtype:
                try:
                    method_data[factor] = method_data[factor].astype(original_data[factor].dtype)
                except: return 0.5 # Skip factor if types can't be matched


            factor_levels = sorted(original_data[factor].dropna().unique())
            if len(factor_levels) < 2: return 0.5

            # Sample features
            n_sample = min(50, len(feature_cols))
            if n_sample < 2: return 0.5
            sampled_cols = np.random.choice(feature_cols, n_sample, replace=False)

            level_medians_orig = []
            level_medians_method = []
            valid_levels = []

            for level in factor_levels:
                level_data_orig = original_data[original_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
                level_data_method = method_data[method_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
                if len(level_data_orig) > 1 and len(level_data_method) > 1:
                    level_medians_orig.append(level_data_orig.median().values)
                    level_medians_method.append(level_data_method.median().values)
                    valid_levels.append(level)

            if len(valid_levels) < 2: return 0.5

            try:
                median_mat_orig = np.vstack(level_medians_orig)
                median_mat_method = np.vstack(level_medians_method)
                corr, _ = spearmanr(median_mat_orig.flatten(), median_mat_method.flatten())
                return (corr + 1) / 2 if not np.isnan(corr) else 0.5
            except Exception:
                return 0.5


        # Calculate scores for all combinations
        all_methods = sorted(list(set(self.spectral_methods + self.metabolite_methods)))
        for factor in common_factors:
            for method in all_methods:
                score_spec = np.nan
                if method in self.spectral_methods:
                     score_spec = get_factor_preservation_score('Spectral', method, factor)
                     factor_scores_data.append({'Factor': factor, 'Method': method, 'Modality': 'Spectral', 'Score': score_spec})

                score_met = np.nan
                if method in self.metabolite_methods:
                    score_met = get_factor_preservation_score('Metabolite', method, factor)
                    factor_scores_data.append({'Factor': factor, 'Method': method, 'Modality': 'Metabolite', 'Score': score_met})


        if not factor_scores_data:
            ax.text(0.5, 0.5, "No factor scores calculated", ha='center', va='center', fontsize=self.FONTS['annotation'])
        else:
            df = pd.DataFrame(factor_scores_data)
            pivot_data = df.pivot_table(index=['Factor', 'Method'], columns='Modality', values='Score')
            pivot_data = pivot_data.dropna(how='all').sort_index() # Sort for consistency

            if pivot_data.empty:
                 ax.text(0.5, 0.5, "No factor scores calculated", ha='center', va='center', fontsize=self.FONTS['annotation'])
            else:
                sns.heatmap(pivot_data, annot=True, cmap='viridis', vmin=0, vmax=1, ax=ax, fmt=".2f",
                            linewidths=0.5, linecolor=self.COLORS['Grid'],
                            annot_kws={"size": self.FONTS['annotation'] - 4},
                            cbar_kws={'label': 'Factor Preservation Score (0-1)'})
                ax.figure.axes[-1].yaxis.label.set_size(self.FONTS['axis_label']-4) # Cbar label
                ax.tick_params(axis='y', rotation=0, labelsize=self.FONTS['tick_label'] - 4)
                ax.tick_params(axis='x', labelsize=self.FONTS['tick_label'] - 4)

        # Styling
        ax.set_title('Experimental Factor Preservation by Method & Modality', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel("Modality", fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])
        ax.set_ylabel("Factor & Method", fontsize=self.FONTS['axis_label'] - 2, color=self.COLORS['Text_Dark'])

    def create_publication_figures(self):
        """Generate all publication-quality figures."""
        print("Generating publication figures...")
        spec_sig_path = self._create_spectral_signature_figure()
        met_prof_path = self._create_molecular_features_profile_figure()
        aug_qual_path = self._create_augmentation_quality_figure()
        meth_comp_path = self._create_method_comparison_figure()
        print(f"  - Spectral Signatures: {spec_sig_path}")
        print(f"  - Molecular Features Profiles: {met_prof_path}")
        print(f"  - Augmentation Quality: {aug_qual_path}")
        print(f"  - Method Comparison: {meth_comp_path}")

    def _create_method_comparison_figure(self):
        """
        Create a radar chart comparing different augmentation methods across key metrics.
        """
        print("Generating method comparison radar chart...")
        spectral_metrics = {}
        molecular_features_metrics = {}

        # Categories for radar chart (Using calculated metrics)
        categories = ['Distribution', 'Structure', 'Biological']
        # Add Treatment/Batch if columns exist
        has_treatment = 'Treatment' in self.common_metadata and self.spectral_original['Treatment'].nunique() > 1
        has_batch = 'Batch' in self.common_metadata and self.spectral_original['Batch'].nunique() > 1
        if has_treatment: categories.append('Treatment Effect')
        if has_batch: categories.append('Batch Effect')

        # Calculate metrics for spectral methods
        for method in self.spectral_methods:
            method_data_spec = self.spectral_augmented[self.spectral_augmented['Row_names'].str.contains(f'_{method}', regex=False, na=False)]
            if not method_data_spec.empty:
                 metrics = {
                     'Distribution': self._distribution_metric('spectral', method),
                     'Structure': self._structure_metric('spectral', method),
                     'Biological': self._biological_metric('spectral', method),
                 }
                 if has_treatment: metrics['Treatment Effect'] = self._treatment_metric('spectral', method)
                 if has_batch: metrics['Batch Effect'] = self._batch_metric('spectral', method)
                 spectral_metrics[method] = metrics


        # Calculate metrics for Molecular features methods
        for method in self.molecular_features_methods:
            method_data_mol = self.molecular_features_augmented[self.molecular_features_augmented['Row_names'].str.contains(f'_{method}', regex=False, na=False)]
            if not method_data_mol.empty:
                 metrics = {
                    'Distribution': self._distribution_metric('molecular_features', method),
                    'Structure': self._structure_metric('molecular_features', method),
                    'Biological': self._biological_metric('molecular_features', method),
                 }
                 if has_treatment: metrics['Treatment Effect'] = self._treatment_metric('molecular_features', method)
                 if has_batch: metrics['Batch Effect'] = self._batch_metric('molecular_features', method)
                 molecular_features_metrics[method] = metrics

        # Create figure
        fig = plt.figure(figsize=(18, 9)) # Wider figure
        n_cols = 0
        if spectral_metrics: n_cols += 1
        if molecular_features_metrics: n_cols += 1

        if n_cols == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No methods with calculated metrics found.",
                    ha='center', va='center', fontsize=self.FONTS['panel_title'])
            ax.set_title("Method Comparison")
            ax.axis('off')
        else:
            gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.4) # Add space between radars
            col_idx = 0
            if spectral_metrics:
                ax1 = fig.add_subplot(gs[0, col_idx], polar=True)
                self._create_radar_chart(ax1, categories, spectral_metrics, "Spectral Methods")
                col_idx += 1
            if metabolite_metrics:
                 ax2 = fig.add_subplot(gs[0, col_idx], polar=True)
                 self._create_radar_chart(ax2, categories, metabolite_metrics, "Metabolite Methods")


            # Add overall title
            plt.suptitle("Augmentation Method Comparison Across Key Metrics",
                         fontsize=self.FONTS['main_title'], fontweight='bold', y=1.02, color=self.COLORS['Text_Dark'])

            # Remove annotation explaining metrics
            # metrics_explanation = "Metrics (0-1 Score, Higher is Better):\n"
            # metrics_explanation += " Distribution: Statistical distribution preservation (1 - Scaled JS Divergence)\n"
            # metrics_explanation += " Structure: Correlation structure preservation (Spearman corr. of Spearman corr. matrices)\n"
            # metrics_explanation += " Biological: Preservation of signals across factors (e.g., Genotype, Tissue)\n"
            # if has_treatment: metrics_explanation += " Treatment Effect: Preservation of differences between treatments\n"
            # if has_batch: metrics_explanation += " Batch Effect: Consistency of differences between batches\n"
            #
            # plt.figtext(0.5, -0.02, metrics_explanation, ha='center', # Lower position
            #            fontsize=self.FONTS['caption'], color=self.COLORS['Text_Annotation'],
            #            bbox=dict(facecolor=self.COLORS['Annotation_Box_BG'], alpha=0.8,
            #                      boxstyle='round,pad=0.5', edgecolor=self.COLORS['Annotation_Box_Edge']))


        # Save the figure
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout
        output_path = os.path.join(self.output_dir, 'publication_method_comparison.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'publication_method_comparison.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    def _create_radar_chart(self, ax, categories, metrics_dict, title):
        """Helper method to create a radar chart for method comparison"""
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1] # Close the loop

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=self.FONTS['tick_label']-2, color=self.COLORS['Text_Dark']) # Set tick labels
        # ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=self.FONTS['tick_label']-2) # Alternative way

        ax.set_yticks(np.arange(0, 1.1, 0.25)) # Y ticks 0, 0.25, 0.5, 0.75, 1
        ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize=self.FONTS['tick_label']-4, color=self.COLORS['Text_Dark'])
        ax.set_ylim(0, 1)
        ax.grid(color=self.COLORS['Grid'], linestyle='--', linewidth=0.5)
        ax.spines['polar'].set_color(self.COLORS['Text_Dark']) # Set outline color

        # Add plots for each method
        method_names = sorted(metrics_dict.keys())
        # Use a distinct colormap
        color_map = plt.cm.get_cmap('tab10', len(method_names))


        for i, method in enumerate(method_names):
            metrics = metrics_dict[method]
            values = [metrics.get(cat, 0.5) for cat in categories] # Default to 0.5 if metric missing
            values += values[:1] # Close the loop

            color = self.COLORS.get(method, color_map(i)) # Use defined color first
            ax.plot(angles, values, 'o-', linewidth=2, markersize=4, label=method, color=color, alpha=0.9)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Add legend (position adjusted)
        ax.legend(loc='lower right', bbox_to_anchor=(1.1, -0.15), # Below and to the right
                  fontsize=self.FONTS['legend_text'] - 4, title='Method', title_fontsize=self.FONTS['legend_title'] - 4)

        # Add title
        ax.set_title(title, fontsize=self.FONTS['panel_title'], pad=25, color=self.COLORS['Text_Dark'], fontweight='bold')

    # --- Metric Calculation Methods for Radar Chart ---
    # These reuse the core logic from _calculate_X_metrics but target a single method

    def _distribution_metric(self, modality, method):
        """Calculate distribution preservation score for a single method."""
        metrics = self._calculate_distribution_metrics(modality)
        return metrics.get(method, 0.5) # Default 0.5 if method not found or failed

    def _structure_metric(self, modality, method):
        """Calculate structure preservation score for a single method."""
        metrics = self._calculate_structure_metrics(modality)
        return metrics.get(method, 0.5)

    def _biological_metric(self, modality, method):
        """Calculate biological signal preservation score for a single method."""
        metrics = self._calculate_biological_metrics(modality)
        return metrics.get(method, 0.5)

    def _treatment_metric(self, modality, method):
        """Calculate treatment effect preservation score for a single method."""
        factor = 'Treatment'
        if factor not in self.common_metadata or self.spectral_original[factor].nunique() < 2:
            return 0.5 # Cannot calculate

        original_data = self.spectral_original if modality == 'spectral' else self.metabolite_original
        augmented_data = self.spectral_augmented if modality == 'spectral' else self.metabolite_augmented
        feature_cols = self.wavelength_cols if modality == 'spectral' else self.metabolite_cols
        method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]

        if method_data.empty or len(feature_cols) < 2: return 0.5

        # Ensure factor types match
        if original_data[factor].dtype != method_data[factor].dtype:
            try:
                method_data[factor] = method_data[factor].astype(original_data[factor].dtype)
            except: return 0.5

        factor_levels = sorted(original_data[factor].dropna().unique())
        if len(factor_levels) < 2: return 0.5

        # Sample features
        n_sample = min(50, len(feature_cols))
        if n_sample < 2: return 0.5
        sampled_cols = np.random.choice(feature_cols, n_sample, replace=False)

        level_medians_orig = []
        level_medians_method = []
        valid_levels = []
        for level in factor_levels:
            level_data_orig = original_data[original_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
            level_data_method = method_data[method_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if len(level_data_orig) > 1 and len(level_data_method) > 1:
                 level_medians_orig.append(level_data_orig.median().values)
                 level_medians_method.append(level_data_method.median().values)
                 valid_levels.append(level)

        if len(valid_levels) < 2: return 0.5

        try:
            median_mat_orig = np.vstack(level_medians_orig)
            median_mat_method = np.vstack(level_medians_method)
            corr, _ = spearmanr(median_mat_orig.flatten(), median_mat_method.flatten())
            return (corr + 1) / 2 if not np.isnan(corr) else 0.5
        except Exception:
            return 0.5

    def _batch_metric(self, modality, method):
        """Calculate batch effect preservation score for a single method."""
        factor = 'Batch'
        if factor not in self.common_metadata or self.spectral_original[factor].nunique() < 2:
            return 0.5 # Cannot calculate

        # Reuse the logic from _treatment_metric, just change the factor name
        original_data = self.spectral_original if modality == 'spectral' else self.metabolite_original
        augmented_data = self.spectral_augmented if modality == 'spectral' else self.metabolite_augmented
        feature_cols = self.wavelength_cols if modality == 'spectral' else self.metabolite_cols
        method_data = augmented_data[augmented_data['Row_names'].str.contains(f'_{method}', regex=False, na=False)]

        if method_data.empty or len(feature_cols) < 2: return 0.5

         # Ensure factor types match
        if original_data[factor].dtype != method_data[factor].dtype:
            try:
                method_data[factor] = method_data[factor].astype(original_data[factor].dtype)
            except: return 0.5

        factor_levels = sorted(original_data[factor].dropna().unique())
        if len(factor_levels) < 2: return 0.5

        # Sample features
        n_sample = min(50, len(feature_cols))
        if n_sample < 2: return 0.5
        sampled_cols = np.random.choice(feature_cols, n_sample, replace=False)

        level_medians_orig = []
        level_medians_method = []
        valid_levels = []
        for level in factor_levels:
            level_data_orig = original_data[original_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
            level_data_method = method_data[method_data[factor] == level][sampled_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if len(level_data_orig) > 1 and len(level_data_method) > 1:
                 level_medians_orig.append(level_data_orig.median().values)
                 level_medians_method.append(level_data_method.median().values)
                 valid_levels.append(level)

        if len(valid_levels) < 2: return 0.5

        try:
            median_mat_orig = np.vstack(level_medians_orig)
            median_mat_method = np.vstack(level_medians_method)
            corr, _ = spearmanr(median_mat_orig.flatten(), median_mat_method.flatten())
            return (corr + 1) / 2 if not np.isnan(corr) else 0.5
        except Exception:
            return 0.5
    # --- End Metric Calculation Methods ---

    def _calculate_method_metric(self, modality, method, metric_type):
        """Helper to call the correct single-method metric calculator."""
        if metric_type == 'distribution':
            return self._distribution_metric(modality, method)
        elif metric_type == 'structure':
            return self._structure_metric(modality, method)
        elif metric_type == 'biological':
            return self._biological_metric(modality, method)
        elif metric_type == 'treatment':
            return self._treatment_metric(modality, method)
        elif metric_type == 'batch':
            return self._batch_metric(modality, method)
        else:
            # print(f"Warning: Unknown metric type '{metric_type}'")
            return 0.5

    def _calculate_factor_score(self, factor):
        """Calculate average quality score for a specific factor across methods/modalities"""
        spectral_scores = []
        metabolite_scores = []

        if factor not in self.common_metadata: return 0.5

        # Calculate scores for spectral methods
        for method in self.spectral_methods:
            score = self._treatment_metric('spectral', method) if factor=='Treatment' else \
                    self._batch_metric('spectral', method) if factor=='Batch' else \
                    self._biological_metric('spectral', method) # Use general biological for others
            spectral_scores.append(score)

        # Calculate scores for metabolite methods
        for method in self.metabolite_methods:
             score = self._treatment_metric('metabolite', method) if factor=='Treatment' else \
                     self._batch_metric('metabolite', method) if factor=='Batch' else \
                     self._biological_metric('metabolite', method)
             metabolite_scores.append(score)


        all_scores = [s for s in spectral_scores + metabolite_scores if not np.isnan(s)]
        return np.mean(all_scores) if all_scores else 0.5

    def _create_augmentation_quality_figure(self):
        """
        Create comprehensive augmentation quality summary figure.
        """
        print("Generating comprehensive quality figure...")
        fig = plt.figure(figsize=(18, 14)) # Adjusted size
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.4) # Adjusted spacing

        # Panel 1: Overall Quality Metrics (across methods & modalities)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_quality_metrics_panel(ax1)
        ax1.text(-0.1, 1.05, "A)", transform=ax1.transAxes, size=self.FONTS['panel_label'], weight='bold')


        # Panel 2: Modality Comparison (Spectral vs Metabolite)
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_modality_comparison_panel(ax2)
        ax2.text(-0.1, 1.05, "B)", transform=ax2.transAxes, size=self.FONTS['panel_label'], weight='bold')


        # Panel 3: Method Comparison (Overall score per method)
        ax3 = fig.add_subplot(gs[1, 0])
        self._create_method_comparison_panel(ax3)
        ax3.text(-0.1, 1.05, "C)", transform=ax3.transAxes, size=self.FONTS['panel_label'], weight='bold')


        # Panel 4: Factor-Specific Quality (Average score per factor)
        ax4 = fig.add_subplot(gs[1, 1])
        self._create_factor_quality_panel(ax4)
        ax4.text(-0.1, 1.05, "D)", transform=ax4.transAxes, size=self.FONTS['panel_label'], weight='bold')


        # Add overall title
        plt.suptitle("Comprehensive Augmentation Quality Assessment",
                     fontsize=self.FONTS['main_title'], fontweight='bold', y=0.99, color=self.COLORS['Text_Dark'])

        # Add annotation about non-parametric approaches
        plt.figtext(0.5, 0.01,
                    "Note: Scores based on non-parametric statistics (JS Div., Spearman Corr., Median Profiles)",
                    ha='center', fontsize=self.FONTS['caption'], style='italic', color=self.COLORS['Text_Dark'])

        # Save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        output_path = os.path.join(self.output_dir, 'publication_quality_metrics.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'publication_quality_metrics.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    def _create_quality_metrics_panel(self, ax):
        """Create panel showing overall quality metrics averaged across methods/modalities"""
        metrics = {
            'Distribution': self._calculate_overall_metric('distribution'),
            'Structure': self._calculate_overall_metric('structure'),
            'Biological': self._calculate_overall_metric('biological'),
        }
        # Add Treatment/Batch if available
        if 'Treatment' in self.common_metadata and self.spectral_original['Treatment'].nunique() > 1:
             metrics['Treatment Effect'] = self._calculate_overall_metric('treatment')
        if 'Batch' in self.common_metadata and self.spectral_original['Batch'].nunique() > 1:
             metrics['Batch Consistency'] = self._calculate_overall_metric('batch')


        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
        metrics_df = metrics_df.sort_values('Score', ascending=True) # Horizontal bars look better ascending

        # Define colors based on score thresholds
        def get_color(score):
            if score >= 0.9: return self.COLORS.get('Positive_Diff', 'green')
            if score >= 0.7: return self.COLORS.get('T1', 'orange')
            return self.COLORS.get('Negative_Diff', 'red')
        colors = metrics_df['Score'].apply(get_color)

        bars = ax.barh(metrics_df['Metric'], metrics_df['Score'], color=colors)

        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=self.FONTS['annotation']-2, color=self.COLORS['Text_Dark'])

        # Styling
        ax.set_title('Overall Quality Metrics', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel('Average Score (0-1)', fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='x', labelsize=self.FONTS['tick_label']-2)
        ax.tick_params(axis='y', labelsize=self.FONTS['tick_label']-2)
        ax.grid(axis='x', alpha=0.4, color=self.COLORS['Grid'])

        # Add quality thresholds
        ax.axvline(x=0.9, linestyle='--', color=self.COLORS.get('Positive_Diff', 'green'), alpha=0.7, label='Excellent')
        ax.axvline(x=0.7, linestyle='--', color=self.COLORS.get('T1', 'orange'), alpha=0.7, label='Good')
        # ax.legend(loc='lower right', fontsize=self.FONTS['legend_text']-4) # Legend might clutter bar chart

    def _create_modality_comparison_panel(self, ax):
        """Create panel comparing average quality scores for spectral vs Molecular features"""
        metric_types = ['distribution', 'structure', 'biological']
        if 'Treatment' in self.common_metadata and self.spectral_original['Treatment'].nunique() > 1:
            metric_types.append('treatment')
        if 'Batch' in self.common_metadata and self.spectral_original['Batch'].nunique() > 1:
            metric_types.append('batch')

        metric_names_map = { # Nicer names for plot
            'distribution': 'Distribution', 'structure': 'Structure', 'biological': 'Biological',
            'treatment': 'Treatment Effect', 'batch': 'Batch Consistency'
        }

        data = []
        for mtype in metric_types:
            score_spec = self._calculate_modality_metric('spectral', mtype)
            score_mol = self._calculate_modality_metric('molecular_features', mtype)
            mname = metric_names_map.get(mtype, mtype.capitalize())
            data.append({'Modality': 'Spectral', 'Metric': mname, 'Score': score_spec})
            data.append({'Modality': 'Molecular Features', 'Metric': mname, 'Score': score_mol})

        df = pd.DataFrame(data)

        # Define palette
        palette = {'Spectral': self.COLORS.get('Spectral', '#6baed6'),
                   'Molecular Features': self.COLORS.get('Molecular features', '#41ab5d')}

        sns.barplot(x='Metric', y='Score', hue='Modality', data=df, ax=ax, palette=palette)

        # Styling
        ax.set_title('Quality Comparison: Spectral vs Molecular Features', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel('Metric', fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])
        ax.set_ylabel('Average Score (0-1)', fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])
        ax.set_ylim(0, 1.05)
        # REMOVED ha='right' from the line below
        ax.tick_params(axis='x', rotation=45, labelsize=self.FONTS['tick_label'] - 4)
        ax.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 2)
        ax.legend(title='Modality', fontsize=self.FONTS['legend_text'] - 4, title_fontsize=self.FONTS['legend_title'] - 4)
        ax.grid(axis='y', alpha=0.4, color=self.COLORS['Grid'])


    def _create_method_comparison_panel(self, ax):
        """Create panel comparing overall quality score per method"""
        method_scores = {}
        metrics_to_average = ['distribution', 'structure', 'biological'] # Core metrics

        # Spectral methods
        for method in self.spectral_methods:
            scores = [self._calculate_method_metric('spectral', method, mtype) for mtype in metrics_to_average]
            method_scores[f"{method} (S)"] = np.mean([s for s in scores if not np.isnan(s)]) if any(~np.isnan(scores)) else 0.5

        # Metabolite methods
        for method in self.metabolite_methods:
            scores = [self._calculate_method_metric('metabolite', method, mtype) for mtype in metrics_to_average]
            method_scores[f"{method} (M)"] = np.mean([s for s in scores if not np.isnan(s)]) if any(~np.isnan(scores)) else 0.5

        if not method_scores:
            ax.text(0.5, 0.5, "No method scores calculated", ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax.set_title('Method Quality Comparison', fontsize=self.FONTS['panel_title'])
            return


        method_df = pd.DataFrame(list(method_scores.items()), columns=['Method', 'Score'])
        method_df = method_df.sort_values('Score', ascending=True) # Horizontal bars look better ascending

        # Define colors based on modality marker in name
        colors = [self.COLORS.get('Spectral', 'blue') if '(S)' in name else self.COLORS.get('Metabolite', 'green')
                  for name in method_df['Method']]

        bars = ax.barh(method_df['Method'], method_df['Score'], color=colors)

        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=self.FONTS['annotation']-2, color=self.COLORS['Text_Dark'])

        # Styling
        ax.set_title('Method Quality Comparison (Overall)', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel('Average Score (Dist, Struct, Bio)', fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='x', labelsize=self.FONTS['tick_label']-2)
        ax.tick_params(axis='y', labelsize=self.FONTS['tick_label']-2)
        ax.grid(axis='x', alpha=0.4, color=self.COLORS['Grid'])

        # Add legend for modality
        spectral_patch = mpatches.Patch(color=self.COLORS.get('Spectral', 'blue'), label='Spectral (S)')
        metabolite_patch = mpatches.Patch(color=self.COLORS.get('Metabolite', 'green'), label='Metabolite (M)')
        ax.legend(handles=[spectral_patch, metabolite_patch], loc='lower right',
                  fontsize=self.FONTS['legend_text']-4, title_fontsize=self.FONTS['legend_title']-4)

    def _create_factor_quality_panel(self, ax):
        """Create panel showing average quality score preservation per factor"""
        # Get common factors with multiple levels
        potential_factors = ['Treatment', 'Genotype', 'Batch', 'Day', 'Tissue.type']
        common_factors = [f for f in potential_factors if f in self.common_metadata and self.spectral_original[f].nunique() > 1]
        if not common_factors:
             common_factors = [f for f in self.common_metadata if self.spectral_original[f].nunique() > 1 and self.spectral_original[f].nunique() < 10]


        if not common_factors:
            ax.text(0.5, 0.5, "No common experimental factors found",
                    ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax.set_title('Experimental Factor Preservation', fontsize=self.FONTS['panel_title'])
            return

        factor_scores = {factor: self._calculate_factor_score(factor) for factor in common_factors}

        factor_df = pd.DataFrame(list(factor_scores.items()), columns=['Factor', 'Score'])
        factor_df = factor_df.sort_values('Score', ascending=True)

        # Define colors based on score thresholds
        def get_color(score):
            if score >= 0.9: return self.COLORS.get('Positive_Diff', 'green')
            if score >= 0.7: return self.COLORS.get('T1', 'orange')
            return self.COLORS.get('Negative_Diff', 'red')
        colors = factor_df['Score'].apply(get_color)

        bars = ax.barh(factor_df['Factor'], factor_df['Score'], color=colors)

        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=self.FONTS['annotation']-2, color=self.COLORS['Text_Dark'])

        # Styling
        ax.set_title('Experimental Factor Preservation', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
        ax.set_xlabel('Average Score (across methods/modalities)', fontsize=self.FONTS['axis_label']-2, color=self.COLORS['Text_Dark'])
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='x', labelsize=self.FONTS['tick_label']-2)
        ax.tick_params(axis='y', labelsize=self.FONTS['tick_label']-2)
        ax.grid(axis='x', alpha=0.4, color=self.COLORS['Grid'])


    def _calculate_overall_metric(self, metric_type):
        """Calculate overall quality metric by averaging across modalities"""
        # Get average scores per modality
        spectral_score = self._calculate_modality_metric('spectral', metric_type)
        metabolite_score = self._calculate_modality_metric('metabolite', metric_type)

        # Average the modality scores
        scores = [s for s in [spectral_score, metabolite_score] if not np.isnan(s)]
        return np.mean(scores) if scores else 0.5

    def _calculate_modality_metric(self, modality, metric_type):
        """Calculate quality metric for a modality by averaging across its methods"""
        methods = self.spectral_methods if modality == 'spectral' else self.metabolite_methods
        if not methods: return 0.5 # Return default if no methods for this modality

        method_scores = [self._calculate_method_metric(modality, method, metric_type)
                         for method in methods]

        scores = [s for s in method_scores if not np.isnan(s)]
        return np.mean(scores) if scores else 0.5

    def run_all_visualizations(self):
        """Run all visualization methods with timing."""
        total_start_time = time.time()
        print("Starting visualization suite generation...")

        viz_methods = [
            ("Factor-Specific Augmentation Effect", self.factor_specific_augmentation_effect),
            ("Parallel Coordinate Plot", self.create_parallel_coordinate_plot),
            ("Augmentation Quality Heatmaps (Placeholder)", self.augmentation_quality_heatmaps),
            ("Integrated Dashboard", self.create_integrated_dashboard),
            ("Publication Figures", self.create_publication_figures), # This calls multiple sub-methods
        ]

        for name, method_func in viz_methods:
            print(f"\n--- Running: {name} ---")
            start_time = time.time()
            try:
                method_func()
                end_time = time.time()
                print(f"--- Completed: {name} in {end_time - start_time:.2f} seconds ---")
            except Exception as e:
                end_time = time.time()
                print(f"--- ERROR in {name} after {end_time - start_time:.2f} seconds: {e} ---")
                import traceback
                traceback.print_exc() # Print detailed error


        total_end_time = time.time()
        print(f"\nAll visualizations attempted in {total_end_time - total_start_time:.2f} seconds")
        print(f"Output directory: {self.output_dir}")

    def _create_molecular_features_profile_figure(self):
        """
        Create publication-quality Molecular features profile comparison using boxplots.

        Note: A table component was previously part of this figure but has been
        removed to prevent overlap issues in the plot.
        """
        # Identify the most informative Molecular features to visualize (differential between orig/aug)
        molecular_features_to_show = self._identify_differential_molecular_features(20)

        if not molecular_features_to_show or self.molecular_features_augmented_only.empty:
             print("Skipping Molecular features profile figure: No differential Molecular features found or no augmented data.")
             # Optionally create an empty placeholder plot
             fig, ax = plt.subplots(figsize=(15, 6)) # Adjusted size
             ax.text(0.5, 0.5, "Molecular features profile data not available\\nor insufficient for comparison.",
                     ha='center', va='center', fontsize=self.FONTS['panel_title'])
             ax.set_title("Molecular Features Profile Comparison", fontsize=self.FONTS['panel_title'])
             ax.axis('off')
             output_path = os.path.join(self.output_dir, 'publication_molecular_features_profiles.png')
             plt.savefig(output_path, dpi=300, bbox_inches='tight')
             pdf_path = os.path.join(self.output_dir, 'publication_molecular_features_profiles.pdf')
             plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
             plt.close(fig)
             return None


        # Create a single-panel figure for the boxplot
        fig, ax1 = plt.subplots(figsize=(18, 9)) # Adjusted figure size for single panel

        # 1. Create boxplot comparison
        molecular_features_data = []
        for molecular_feature in molecular_features_to_show:
            if molecular_feature not in self.molecular_features_original.columns or molecular_feature not in self.molecular_features_augmented_only.columns:
                continue # Skip if molecular_feature missing in either dataset

            orig_values = self.molecular_features_original[molecular_feature].dropna().values
            aug_values = self.molecular_features_augmented_only[molecular_feature].dropna().values
            met_label = molecular_feature.split('_')[-1] if '_' in molecular_feature else molecular_feature # Clean label

            for val in orig_values:
                molecular_features_data.append({'Molecular_Feature': met_label, 'Value': val, 'Source': 'Original'})
            if len(aug_values) > 0:
                for val in aug_values:
                    molecular_features_data.append({'Molecular_Feature': met_label, 'Value': val, 'Source': 'Augmented'})

        if not molecular_features_data:
            ax1.text(0.5, 0.5, "No data for boxplot.", ha='center', va='center', fontsize=self.FONTS['annotation'])
            ax1.set_title('Molecular Features Profile Comparison (Original vs. Augmented)', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax1.axis('off') # Turn off axis if no data
        else:
            boxplot_df = pd.DataFrame(molecular_features_data)
            # Get unique Molecular features labels in the order they appear for consistent plotting
            ordered_molecular_features = boxplot_df['Molecular_Feature'].unique()

            sns.boxplot(x='Molecular_Feature', y='Value', hue='Source',
                        data=boxplot_df, ax=ax1, order=ordered_molecular_features,
                        palette={'Original': self.plot_colors['original'],
                                 'Augmented': self.plot_colors['augmented']},
                        fliersize=2) # Smaller outlier markers

            ax1.tick_params(axis='x', rotation=45, labelsize=self.FONTS['tick_label'] - 2)
            ax1.tick_params(axis='y', labelsize=self.FONTS['tick_label'] - 2)
            ax1.set_xlabel("Top Differential Molecular Features", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
            ax1.set_ylabel("Abundance", fontsize=self.FONTS['axis_label'], color=self.COLORS['Text_Dark'])
            ax1.set_title('Molecular Features Profile Comparison (Original vs. Augmented)', fontsize=self.FONTS['panel_title'], color=self.COLORS['Text_Dark'])
            ax1.legend(title='Data Source', title_fontsize=self.FONTS['legend_title']-2, fontsize=self.FONTS['legend_text']-2)
            ax1.grid(axis='y', alpha=0.4, color=self.COLORS['Grid'])

        # Section for table removed

        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust layout

        # Save the plot
        output_path = os.path.join(self.output_dir, 'publication_molecular_features_profiles.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
        pdf_path = os.path.join(self.output_dir, 'publication_molecular_features_profiles.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    def _identify_differential_molecular_features(self, n_top=20):
        """
        Identify Molecular features showing the most difference between original and augmented data
        using Mann-Whitney U effect size (Rank-Biserial Correlation).
        """
        if self.molecular_features_augmented_only.empty or not self.molecular_features_cols:
             return []

        diff_metrics = {}
        for molecular_feature in self.molecular_features_cols:
            if molecular_feature not in self.molecular_features_original.columns or molecular_feature not in self.molecular_features_augmented_only.columns:
                 continue

            orig_values = self.molecular_features_original[molecular_feature].dropna().values
            aug_values = self.molecular_features_augmented_only[molecular_feature].dropna().values

            # Need sufficient samples for a meaningful comparison
            if len(orig_values) < 5 or len(aug_values) < 5:
                continue

            try:
                u_stat, p_value = mannwhitneyu(orig_values, aug_values, alternative='two-sided')
                n1, n2 = len(orig_values), len(aug_values)
                # Rank-Biserial Correlation (effect size for MWU)
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                diff_metrics[molecular_feature] = abs(effect_size) # Use absolute effect size for ranking

            except ValueError: # Handle constant data or other issues
                continue
            except Exception as e:
                # print(f"Warning: Differential Molecular features calculation failed for {molecular_feature}: {e}")
                continue

        # Rank Molecular features by absolute effect size (largest difference first)
        sorted_molecular_features = sorted(diff_metrics.items(), key=lambda x: x[1], reverse=True)

        # Return top n molecular_feature names
        top_molecular_features = [m for m, _ in sorted_molecular_features[:n_top]]

         # Ensure we return exactly n_top if possible, padding if needed
        if len(top_molecular_features) < n_top:
            remaining = [m for m in self.molecular_features_cols if m not in top_molecular_features and m in diff_metrics]
            top_molecular_features.extend(remaining[:n_top - len(top_molecular_features)])
        if len(top_molecular_features) < n_top:
            remaining_any = [m for m in self.molecular_features_cols if m not in top_molecular_features]
            top_molecular_features.extend(remaining_any[:n_top - len(top_molecular_features)])


        return top_molecular_features


def main():
    """Main function to run the visualization suite."""
    # Define file paths (USE THE PATHS FROM THE ORIGINAL CODE)
    spectral_original_path = "C:\\Users\\ms\\Desktop\\hyper\\data\\hyper_full_w.csv"
    spectral_augmented_path = "C:\\Users\\ms\\Desktop\\hyper\\output\\augment\\hyper\\augmented_spectral_data.csv"
    molecular_features_original_path = "C:\\Users\\ms\\Desktop\\hyper\\data\\n_p_r2.csv"
    molecular_features_augmented_path = "C:\\Users\\ms\\Desktop\\hyper\\output\\augment\\metabolite\\root\\augmented_metabolite_data.csv"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\augment\visualisation\test"

    # Create visualization suite instance
    vis_suite = VisualizationSuite(
        spectral_original_path,
        spectral_augmented_path,
        molecular_features_original_path,
        molecular_features_augmented_path,
        output_dir
    )

    # Run all visualizations
    vis_suite.run_all_visualizations()

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
