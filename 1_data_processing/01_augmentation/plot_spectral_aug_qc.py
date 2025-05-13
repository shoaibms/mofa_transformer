"""
SpectralPlotter: A tool for generating publication-quality visualizations of spectral data quality control results.

This module provides functionality to create comprehensive visualizations for assessing the quality of 
original and augmented spectral data. It generates plots for outlier detection, signal-to-noise ratio,
vegetation indices, physical constraint compliance, and overall quality scores.

The tool can create individual plots as well as comprehensive method comparisons and spectral examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


class SpectralPlotter:
    """
    Class for generating publication-quality visualizations for spectral quality control results.
    """
    
    def __init__(self, qc_dir, output_dir=None):
        """
        Initialize the SpectralPlotter class.
        
        Parameters:
        -----------
        qc_dir : str
            Path to directory containing QC results
        output_dir : str, optional
            Directory to save plots (defaults to qc_dir/plots if None)
        """
        self.qc_dir = qc_dir
        
        if output_dir is None:
            self.output_dir = os.path.join(qc_dir, 'plots')
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set up plot styling
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Configure matplotlib for publication-quality plots."""
        # Set general plotting parameters
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # Create custom color scheme, incorporating values from colour.py
        self.colors = {
            'original': '#0072B2',      # Blue
            'augmented': '#D55E00',     # Vermillion
            'GP': '#CC79A7',            # Pink
            'MIX': '#56B4E9',           # Sky blue
            'WARP': '#E69F00',          # Orange
            'SCALE': '#009E73',         # Green
            'NOISE': '#F0E442',         # Yellow
            'ADD': '#0072B2',           # Blue
            'MULT': '#D55E00',          # Vermillion
            
            # Additional colors from colour.py
            'Spectral': '#6baed6',             # General Spectral (Medium Blue)
            'Molecular_features': '#41ab5d',    # General Molecular features (Medium-Dark Yellow-Green)
            'UnknownFeature': '#969696',       # Medium Grey for fallback
            
            'Spectral_Water': '#3182bd',       # Medium-Dark Blue
            'Spectral_Pigment': '#238b45',     # Medium-Dark Green
            'Spectral_Structure': '#7fcdbb',   # Medium Teal
            'Spectral_SWIR': '#636363',        # Dark Grey
            'Spectral_VIS': '#c2e699',         # Light Yellow-Green
            'Spectral_RedEdge': '#78c679',     # Medium Yellow-Green
            'Spectral_UV': '#08519c',          # Darkest Blue
            'Spectral_Other': '#969696',       # Medium Grey
            
            'Molecular_features_PCluster': '#006837',  # Darkest Yellow-Green
            'Molecular_features_NCluster': '#ffffd4',  # Very Light Yellow
            'Molecular_features_Other': '#bdbdbd',     # Light Grey
            
            'Positive_Diff': '#238b45',       # Medium-Dark Green
            'Negative_Diff': '#fe9929',       # Muted Orange/Yellow
            'Significance': '#08519c',        # Dark Blue (for markers/text)
            'NonSignificant': '#bdbdbd',      # Light Grey
            'Difference_Line': '#636363',     # Dark Grey line
            
            # Plot elements
            'Background': '#FFFFFF',          # White plot background
            'Panel_Background': '#f7f7f7',    # Very Light Gray background for panels
            'Grid': '#d9d9d9',                # Lighter Gray grid lines
            'Text_Dark': '#252525',           # Darkest Gray / Near Black text
            'Text_Light': '#FFFFFF',          # White text
            'Text_Annotation': '#000000'      # Black text for annotations
        }
        
        # Set seaborn style
        sns.set_style('white')
        sns.set_context('paper', rc={'lines.linewidth': 2})
        
    def load_qc_data(self):
        """Load QC results from CSV files."""
        self.outlier_data = pd.read_csv(os.path.join(self.qc_dir, 'outlier_detection_combined.csv'))
        self.snr_data = pd.read_csv(os.path.join(self.qc_dir, 'signal_to_noise_assessment.csv'))
        self.region_data = pd.read_csv(os.path.join(self.qc_dir, 'band_specific_regions.csv'))
        self.indices_data = pd.read_csv(os.path.join(self.qc_dir, 'vegetation_indices.csv'))
        self.range_data = pd.read_csv(os.path.join(self.qc_dir, 'range_checks.csv'))
        
        print("Loaded QC data from CSV files.")
        
    def plot_outlier_summary(self):
        """Plot outlier detection summary."""
        plt.figure(figsize=(12, 8))
        
        # Sort by percentage for better visualization
        df = self.outlier_data.sort_values('Percentage')
        
        # Plot bars
        bar_color = [self.colors.get(cat, '#333333') for cat in df['Category']]
        ax = sns.barplot(x='Percentage', y='Category', data=df, palette=bar_color)
        
        # Add data labels
        for i, (p, o, t) in enumerate(zip(df['Percentage'], df['Outliers'], df['Total_Samples'])):
            ax.text(p + 0.2, i, f"{o} / {t} ({p:.1f}%)", va='center')
        
        # Add reference line for original data
        original_pct = float(df[df['Category'] == 'original']['Percentage'])
        ax.axvline(
            x=original_pct, 
            linestyle='--', 
            color='gray', 
            label=f'Original data ({original_pct:.1f}%)'
        )
        
        # Styling
        plt.title('Outlier Detection Summary by Augmentation Method', fontweight='bold')
        plt.xlabel('Percentage of Outliers (%)')
        plt.ylabel('')
        plt.xlim(0, max(df['Percentage']) * 1.5)
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'outlier_summary.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'outlier_summary.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Outlier summary plot saved to: {self.output_dir}")
        
    def plot_snr_comparison(self):
        """Plot signal-to-noise ratio comparison."""
        plt.figure(figsize=(12, 8))
        
        # Sort by mean SNR
        df = self.snr_data.sort_values('Mean_SNR')
        
        # Plot bars with error bars
        ax = sns.barplot(
            x='Mean_SNR', 
            y='Category', 
            data=df, 
            palette=[self.colors.get(cat, '#333333') for cat in df['Category']]
        )
        
        # Add error bars
        for i, (mean, std, min_val, max_val) in enumerate(zip(
                df['Mean_SNR'], df['Std_SNR'], df['Min_SNR'], df['Max_SNR'])):
            ax.errorbar(mean, i, xerr=std, fmt='none', color='black', capsize=5)
            # Add min/max indicators
            ax.plot(min_val, i, marker='|', ms=10, color='black', alpha=0.7)
            ax.plot(max_val, i, marker='|', ms=10, color='black', alpha=0.7)
            
        # Add data labels
        for i, (mean, samples) in enumerate(zip(df['Mean_SNR'], df['Samples'])):
            ax.text(mean + 1, i, f"{mean:.1f} dB (n={samples})", va='center')
        
        # Add reference lines for SNR quality bands
        ax.axvline(x=30, linestyle='--', color='green', alpha=0.5, label='Excellent SNR (>30 dB)')
        ax.axvline(x=20, linestyle='--', color='orange', alpha=0.5, label='Good SNR (>20 dB)')
        
        # Styling
        plt.title('Signal-to-Noise Ratio by Augmentation Method', fontweight='bold')
        plt.xlabel('Mean SNR (dB)')
        plt.ylabel('')
        plt.xlim(0, max(df['Mean_SNR']) * 1.2)
        plt.grid(axis='x', alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'snr_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'snr_comparison.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"SNR comparison plot saved to: {self.output_dir}")
        
    def plot_vegetation_indices(self):
        """Plot vegetation indices comparison."""
        # Filter to just include augmentation methods (not original)
        df = self.indices_data[self.indices_data['Method'] != 'original'].copy()
        
        # Get original data means for reference
        original_means = {}
        for index in df['Index'].unique():
            orig_data = self.indices_data[
                (self.indices_data['Method'] == 'original') & 
                (self.indices_data['Index'] == index)
            ]
            if not orig_data.empty:
                original_means[index] = float(orig_data['Mean'])
        
        # Create figure with subplots for each index
        fig, axes = plt.subplots(
            len(df['Index'].unique()), 
            1, 
            figsize=(12, 4 * len(df['Index'].unique()))
        )
        
        if len(df['Index'].unique()) == 1:
            axes = [axes]  # Convert to list if only one subplot
        
        # Plot each index
        for i, index in enumerate(sorted(df['Index'].unique())):
            index_df = df[df['Index'] == index].sort_values('Pct_Diff')
            
            # Plot bars
            sns.barplot(
                x='Pct_Diff', 
                y='Method', 
                data=index_df, 
                ax=axes[i], 
                palette=[self.colors.get(method, '#333333') for method in index_df['Method']]
            )
            
            # Add data labels
            for j, (method, mean, pct) in enumerate(zip(
                    index_df['Method'], index_df['Mean'], index_df['Pct_Diff'])):
                axes[i].text(pct + (0.5 if pct > 0 else -2), j, f"{mean:.4f} ({pct:+.1f}%)", va='center')
            
            # Add reference line for original data
            if index in original_means:
                axes[i].axvline(
                    x=0, 
                    linestyle='--', 
                    color='gray', 
                    label=f'Original {index}: {original_means[index]:.4f}'
                )
            
            # Styling
            axes[i].set_title(f'{index} Preservation by Augmentation Method', fontweight='bold')
            axes[i].set_xlabel('Percent Difference from Original (%)')
            axes[i].set_ylabel('')
            # Set limits based on data range
            max_abs_pct = max(abs(min(index_df['Pct_Diff'])), abs(max(index_df['Pct_Diff'])))
            axes[i].set_xlim(-max_abs_pct * 1.2, max_abs_pct * 1.2)
            axes[i].grid(axis='x', alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'vegetation_indices.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'vegetation_indices.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Vegetation indices plot saved to: {self.output_dir}")
        
    def plot_range_check_heatmap(self):
        """Plot heatmap of range check results."""
        # Pivot data for heatmap
        pivot_df = self.range_data.pivot(index='Category', columns='Constraint', values='Percentage')
        
        # Sort index to put original and augmented first
        categories = ['original', 'augmented'] + [c for c in pivot_df.index if c not in ['original', 'augmented']]
        pivot_df = pivot_df.reindex(categories)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FF9999', '#FFFF99', '#99FF99'], N=100)
        
        # Plot heatmap
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.1f', 
            cmap=cmap, 
            vmin=0, 
            vmax=100, 
            linewidths=0.5, 
            cbar_kws={'label': 'Compliance (%)'}
        )
        
        # Styling
        plt.title('Physical Constraint Compliance by Augmentation Method', fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'range_check_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'range_check_heatmap.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Range check heatmap saved to: {self.output_dir}")
        
    def plot_overall_quality_radar(self):
        """Plot radar chart of overall quality scores."""
        # Extract quality scores from the QC report
        categories = [
            'Outlier Quality', 
            'Signal-to-Noise Quality', 
            'Spectral Features Quality', 
            'Physical Constraints Quality'
        ]
        
        # These values would typically come from the integrated_qc_report.html
        values = [69.0, 100.0, 82.2, 79.6]
        
        # Calculate angles for radar chart
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add values (and close the loop)
        values += values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot radar
        ax.plot(angles, values, 'o-', linewidth=2, color='#D55E00')
        ax.fill(angles, values, alpha=0.25, color='#D55E00')
        
        # Add reference circle for good quality
        good_values = [80] * N + [80]  # 80 is threshold for "good"
        ax.plot(angles, good_values, '--', linewidth=1.5, color='green', alpha=0.7, label='Good Quality (80)')
        
        # Add reference circle for excellent quality
        excellent_values = [90] * N + [90]  # 90 is threshold for "excellent"
        ax.plot(angles, excellent_values, '--', linewidth=1.5, color='blue', alpha=0.7, label='Excellent Quality (90)')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set radial limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        
        # Add overall score
        overall_score = sum(values[:-1]) / N
        ax.text(
            0, 
            -20, 
            f"Overall Quality Score: {overall_score:.1f}/100", 
            ha='center', 
            va='center', 
            fontsize=14, 
            fontweight='bold'
        )
        
        # Styling
        plt.title('Augmented Spectral Data Quality Assessment', fontweight='bold', pad=20)
        plt.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'quality_radar.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'quality_radar.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Quality radar plot saved to: {self.output_dir}")
        
    def plot_augmentation_method_comparison(self):
        """Plot comprehensive comparison of augmentation methods."""
        # Create figure with multiple panels
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # 1. Outlier percentages
        ax1 = fig.add_subplot(gs[0, 0])
        outlier_df = self.outlier_data[self.outlier_data['Category'] != 'augmented'].copy()
        sns.barplot(
            x='Category', 
            y='Percentage', 
            data=outlier_df, 
            ax=ax1,
            palette=[self.colors.get(cat, '#333333') for cat in outlier_df['Category']]
        )
        ax1.set_title('Outlier Percentage', fontweight='bold')
        ax1.set_xlabel('')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. SNR comparison
        ax2 = fig.add_subplot(gs[0, 1])
        snr_df = self.snr_data[self.snr_data['Category'] != 'augmented'].copy()
        sns.barplot(
            x='Category', 
            y='Mean_SNR', 
            data=snr_df, 
            ax=ax2,
            palette=[self.colors.get(cat, '#333333') for cat in snr_df['Category']]
        )
        ax2.set_title('Signal-to-Noise Ratio', fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('Mean SNR (dB)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. NDVI preservation
        ax3 = fig.add_subplot(gs[1, 0])
        ndvi_df = self.indices_data[
            (self.indices_data['Index'] == 'NDVI') & 
            (self.indices_data['Method'] != 'augmented') &
            (self.indices_data['Method'] != 'original')
        ].copy()
        sns.barplot(
            x='Method', 
            y='Pct_Diff', 
            data=ndvi_df, 
            ax=ax3,
            palette=[self.colors.get(method, '#333333') for method in ndvi_df['Method']]
        )
        ax3.set_title('NDVI Preservation', fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylabel('Percent Difference (%)')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.axhline(y=0, linestyle='--', color='gray')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Overall method ranking
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate combined ranking score
        methods = self.outlier_data[self.outlier_data['Category'] != 'augmented']['Category'].tolist()
        methods = [m for m in methods if m != 'original']
        
        ranking_scores = []
        for method in methods:
            # Low outlier percentage is good
            outlier_score = 100 - float(outlier_df[outlier_df['Category'] == method]['Percentage'])
            
            # High SNR is good
            snr_value = float(snr_df[snr_df['Category'] == method]['Mean_SNR'])
            max_snr = snr_df['Mean_SNR'].max()
            snr_score = (snr_value / max_snr) * 100
            
            # Low absolute NDVI difference is good
            if not ndvi_df[ndvi_df['Method'] == method].empty:
                ndvi_diff = abs(float(ndvi_df[ndvi_df['Method'] == method]['Pct_Diff']))
                ndvi_score = 100 - (ndvi_diff * 10)  # Scale it appropriately
            else:
                ndvi_score = 50  # Default if missing
                
            # Calculate combined score (equal weighting for simplicity)
            combined_score = (outlier_score + snr_score + ndvi_score) / 3
            ranking_scores.append((method, combined_score))
        
        # Sort by score
        ranking_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Plot ranking
        methods_sorted = [x[0] for x in ranking_scores]
        scores_sorted = [x[1] for x in ranking_scores]
        
        sns.barplot(
            x=methods_sorted, 
            y=scores_sorted, 
            ax=ax4,
            palette=[self.colors.get(method, '#333333') for method in methods_sorted]
        )
        
        ax4.set_title('Overall Method Ranking', fontweight='bold')
        ax4.set_xlabel('')
        ax4.set_ylabel('Combined Score')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, score in enumerate(scores_sorted):
            ax4.text(i, score + 1, f"{score:.1f}", ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'method_comparison.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Method comparison plot saved to: {self.output_dir}")
        
    def load_spectra_and_plot_examples(self, original_path, augmented_path):
        """
        Load spectral data and plot representative examples of original and augmented spectra.
        
        Parameters:
        -----------
        original_path : str
            Path to original spectral data CSV file
        augmented_path : str
            Path to augmented spectral data CSV file
        """
        print("Loading spectral data for example plots...")
        
        # Load data
        original_data = pd.read_csv(original_path)
        augmented_data = pd.read_csv(augmented_path)
        
        # Identify wavelength and metadata columns
        wavelength_cols = [col for col in original_data.columns if col.startswith('W_')]
        
        # Extract wavelength values
        wavelengths = np.array([float(col.split('_')[1]) for col in wavelength_cols])
        
        # Get a few example spectra
        original_spectra = original_data[wavelength_cols].values[:5]  # First 5 samples
        
        # Get examples of each augmentation method
        augmentation_methods = ['GP', 'MIX', 'WARP', 'SCALE', 'NOISE', 'ADD', 'MULT']
        augmented_examples = {}
        
        for method in augmentation_methods:
            method_rows = augmented_data[augmented_data['Row_names'].str.contains(method, case=False)]
            if len(method_rows) > 0:
                augmented_examples[method] = method_rows[wavelength_cols].values[0]  # First example
        
        # Create figure for spectral curve comparison
        plt.figure(figsize=(15, 10))
        
        # Plot original spectrum
        plt.plot(wavelengths, original_spectra[0], 'k-', linewidth=2.5, label='Original')
        
        # Plot augmented examples
        for method, spectrum in augmented_examples.items():
            plt.plot(
                wavelengths, 
                spectrum, 
                '-', 
                linewidth=1.5, 
                label=method, 
                color=self.colors.get(method, '#333333'), 
                alpha=0.8
            )
        
        # Styling
        plt.title('Comparison of Original and Augmented Spectral Curves', fontweight='bold')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.xlim(350, 2500)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(ncol=2)
        
        # Annotate key spectral regions
        regions = {
            'Visible': (400, 700),
            'NIR': (700, 1300),
            'SWIR1': (1300, 1900),
            'SWIR2': (1900, 2500)
        }
        
        for region, (start, end) in regions.items():
            mid = (start + end) / 2
            plt.axvspan(start, end, alpha=0.1, color='gray')
            plt.text(mid, 0.05, region, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'spectral_examples.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'spectral_examples.pdf'), bbox_inches='tight')
        plt.close()
        
        # Create figure for spectral difference plot
        plt.figure(figsize=(15, 8))
        
        # Calculate differences
        for method, spectrum in augmented_examples.items():
            difference = spectrum - original_spectra[0]
            # Smooth difference for better visualization
            smoothed_diff = savgol_filter(difference, 51, 3)
            
            plt.plot(
                wavelengths, 
                smoothed_diff, 
                '-', 
                label=method, 
                color=self.colors.get(method, '#333333'), 
                alpha=0.8
            )
        
        # Styling
        plt.title('Spectral Differences (Augmented - Original)', fontweight='bold')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance Difference')
        plt.xlim(350, 2500)
        plt.axhline(y=0, linestyle='--', color='k')
        plt.grid(alpha=0.3)
        plt.legend(ncol=2)
        
        # Annotate key spectral regions
        for region, (start, end) in regions.items():
            mid = (start + end) / 2
            plt.axvspan(start, end, alpha=0.1, color='gray')
            plt.text(mid, -0.05, region, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'spectral_differences.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'spectral_differences.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Example spectral plots saved to: {self.output_dir}")
        
    def generate_all_plots(self, original_path=None, augmented_path=None):
        """
        Generate all plots for QC results visualization.
        
        Parameters:
        -----------
        original_path : str, optional
            Path to original spectral data CSV file (for spectral examples)
        augmented_path : str, optional
            Path to augmented spectral data CSV file (for spectral examples)
        """
        print("\nGenerating all plots for QC visualization...")
        
        # Load QC data
        self.load_qc_data()
        
        # Generate basic QC plots
        self.plot_outlier_summary()
        self.plot_snr_comparison()
        self.plot_vegetation_indices()
        self.plot_range_check_heatmap()
        self.plot_overall_quality_radar()
        self.plot_augmentation_method_comparison()
        
        # If paths provided, generate spectral example plots
        if original_path and augmented_path:
            self.load_spectra_and_plot_examples(original_path, augmented_path)
        
        print("\nAll plots generated successfully!")


# Usage example
if __name__ == "__main__":
    # File paths
    qc_dir = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\quality_control"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\quality_control\publication_plots"
    original_path = r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv"
    augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\augmented_spectral_data.csv"
    
    # Create plotter and generate all plots
    plotter = SpectralPlotter(qc_dir, output_dir)
    plotter.generate_all_plots(original_path, augmented_path)