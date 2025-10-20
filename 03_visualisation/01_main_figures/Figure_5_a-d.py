#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates Figure 5: Multi-scale Quantification of Coordination Network Dynamics
================================================================================

This script produces a publication-quality, four-panel figure for the MOFA+ 
Transformer manuscript, analyzing system dynamics, feature integration, and 
coordination efficiency with a consistent and polished aesthetic.

Panels:
- (a) S→M Attention Structure Dynamics (Coordination Focus)
- (b) S→M Attention Intensity Dynamics (Peak Connection Strength) 
- (c) Predictive Feature Integration (SHAP vs. Attention)
- (d) Coordination Efficiency Landscape (Composite Performance Metric)

The script encapsulates all data loading, processing, and visualization in a 
self-contained class and exports the data underlying each panel for full 
reproducibility.

Author: MOFA+ Transformer Manuscript Contributor
Date: July 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
import logging
from typing import Dict, Optional, Any, Tuple

# --- Configuration ---
# Suppress routine warnings for cleaner output
warnings.filterwarnings('ignore')
# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Figure5Generator:
    """
    Encapsulates all logic for generating Figure 5 to publication standards.
    """
    
    def __init__(self, base_path: str):
        """
        Initializes the generator with the base data path and configuration.

        Args:
            base_path (str): The absolute path to the root directory 
                             containing the experiment's data.
        """
        self.base_path = Path(base_path)
        self.config = self._setup_configuration()
        self.data = {}
        
    def _setup_configuration(self) -> Dict[str, Any]:
        """Sets up paths and publication-quality styling."""
        # Centralized configuration for easy modification
        return {
            'paths': {
                'output_dir': self.base_path / "output" / "transformer" / "novility_plot" / "test",
                'leaf_view': self.base_path / "output" / "transformer" / "v3_feature_attention" / "processed_attention_data_leaf" / "processed_view_level_attention_Leaf.csv",
                'root_view': self.base_path / "output" / "transformer" / "v3_feature_attention" / "processed_attention_data_root" / "processed_view_level_attention_Root.csv",
                'shap': {
                    'leaf_genotype': self.base_path / "output" / "transformer" / "shap_analysis_ggl" / "importance_data" / "shap_importance_Leaf_Genotype.csv",
                    'root_genotype': self.base_path / "output" / "transformer" / "shap_analysis_ggl" / "importance_data" / "shap_importance_Root_Genotype.csv",
                },
                'attention_pairs': {
                    'leaf': self.base_path / "output" / "transformer" / "v3_feature_attention" / "processed_attention_data_leaf" / "processed_top_500_pairs_overall_Leaf.csv",
                    'root': self.base_path / "output" / "transformer" / "v3_feature_attention" / "processed_attention_data_root" / "processed_top_500_pairs_overall_Root.csv",
                },
            },
            'style': {
                'colors': {
                    'G1': "#00FA9A",        # Tolerant Genotype
                    'G2': "#48D1CC",        # Susceptible Genotype
                    'Spectral': '#4169E1',   # Royal Blue
                    'Molecular': '#08722C',  # Dark Green
                    'Hub': "#EFED64",        # Highlight Yellow
                    'Text': '#2f2f2f',
                    'Grid': '#e5e5e5',
                    'Panel_BG': '#f9f9f9',
                    'Efficiency_CMap': ["#f4fff8", "#2def9b", "#06946E"]
                },
                'fonts': {
                    'family': 'Arial',
                    'main_title': 18,
                    'panel_label': 20,
                    'panel_title': 16,
                    'axis_label': 14,
                    'tick_label': 13,
                    'legend': 12,
                    'annotation': 12
                },
                'figure': {'size': (22, 5), 'dpi': 300}
            },
            'mappings': {
                'genotype': {'1': 'G1', '2': 'G2', '1.0': 'G1', '2.0': 'G2'},
                'treatment': {'1': 'T1', '1.0': 'T1', '0': 'T0', '0.0': 'T0'},
            }
        }
    
    def load_data(self) -> bool:
        """Loads, validates, and preprocesses all required datasets."""
        logger.info("Loading and preprocessing all required datasets...")
        try:
            paths = self.config['paths']
            # Load primary view-level data
            self.data['leaf_view'] = pd.read_csv(paths['leaf_view'])
            self.data['root_view'] = pd.read_csv(paths['root_view'])
            
            # Load SHAP and attention pair data, handling missing files gracefully
            for tissue in ['leaf', 'root']:
                shap_path = paths['shap'][f'{tissue}_genotype']
                if shap_path.exists():
                    self.data[f'shap_{tissue}_genotype'] = pd.read_csv(shap_path)
                else:
                    logger.warning(f"SHAP data not found, Panel (c) may be incomplete: {shap_path}")
                
                attn_path = paths['attention_pairs'][tissue]
                if attn_path.exists():
                    self.data[f'{tissue}_pairs'] = pd.read_csv(attn_path)
                else:
                    logger.warning(f"Attention pairs not found, Panel (c) may be incomplete: {attn_path}")

            self._preprocess_data()
            logger.info("Data loading and preprocessing completed successfully.")
            return True
        except Exception as e:
            logger.error(f"A critical error occurred during data loading: {e}")
            return False
            
    def _preprocess_data(self):
        """Applies standard mappings to the loaded dataframes."""
        maps = self.config['mappings']
        for key in ['leaf_view', 'root_view']:
            if key in self.data:
                df = self.data[key]
                df['Genotype'] = df['Genotype'].astype(str).replace(maps['genotype'])
                df['Treatment'] = df['Treatment'].astype(str).replace(maps['treatment'])
                df['Time_Point'] = df.get('Day', df.get('Time_Point')).astype(int)

    def _prepare_dynamics_data(self) -> Optional[pd.DataFrame]:
        """Prepares temporal dynamics data for panels (a) and (b)."""
        logger.info("Calculating data for Panels (a) and (b)...")
        leaf = self.data.get('leaf_view')
        root = self.data.get('root_view')
        if leaf is None or root is None: return None
        
        leaf['Tissue'], root['Tissue'] = 'Leaf', 'Root'
        combined = pd.concat([leaf, root], ignore_index=True)
        stress_data = combined[combined['Treatment'] == 'T1']
        if stress_data.empty: return None

        agg_data = stress_data.groupby(['Tissue', 'Genotype', 'Time_Point']).agg(
            StdAttn_Mean=('StdAttn_S2M', 'mean'), StdAttn_SEM=('StdAttn_S2M', 'sem'),
            P95Attn_Mean=('P95Attn_S2M', 'mean'), P95Attn_SEM=('P95Attn_S2M', 'sem')
        ).reset_index()
        return agg_data

    def _prepare_integration_data(self) -> Optional[pd.DataFrame]:
        """Prepares SHAP-attention integration data for panel (c)."""
        logger.info("Calculating data for Panel (c)...")
        integration_list = []
        for tissue in ['leaf', 'root']:
            shap_df = self.data.get(f'shap_{tissue}_genotype')
            pairs_df = self.data.get(f'{tissue}_pairs')
            if shap_df is None or pairs_df is None: continue

            attn_col = next((c for c in pairs_df.columns if 'Attention_S2M' in c and 'Mean' in c), None)
            if not attn_col: continue
            
            shap_lookup = dict(zip(shap_df['Feature'], shap_df['MeanAbsoluteShap']))
            spec_attn = pairs_df.groupby('Spectral_Feature')[attn_col].sum()
            mol_attn = pairs_df.groupby('Metabolite_Feature')[attn_col].sum()
            attn_lookup = {**spec_attn.to_dict(), **mol_attn.to_dict()}
            
            all_features = set(shap_lookup.keys()) | set(attn_lookup.keys())
            for feature in all_features:
                integration_list.append({
                    'Tissue': tissue.capitalize(),
                    'Feature': feature,
                    'SHAP_Importance': shap_lookup.get(feature, 0),
                    'Attention_Score': attn_lookup.get(feature, 0),
                    'FeatureType': self._classify_feature(feature)
                })
        return pd.DataFrame(integration_list) if integration_list else None

    def _classify_feature(self, feature: str) -> str:
        """Classifies feature type based on naming convention."""
        name = str(feature)
        if name.startswith('W_'): return 'Spectral'
        if name.startswith(('N_', 'P_')): return 'Molecular'
        return 'Unknown'

    def _prepare_efficiency_data(self) -> Optional[pd.DataFrame]:
        """Prepares coordination efficiency data for panel (d)."""
        logger.info("Calculating data for Panel (d)...")
        combined = pd.concat([self.data.get('leaf_view'), self.data.get('root_view')], ignore_index=True)
        stress_data = combined[combined['Treatment'] == 'T1']
        if stress_data.empty: return None

        efficiency_list = []
        for (genotype, time_point), subset in stress_data.groupby(['Genotype', 'Time_Point']):
            s2m_avg, m2s_avg = subset['AvgAttn_S2M'].mean(), subset['AvgAttn_M2S'].mean()
            s2m_std, m2s_std = subset['StdAttn_S2M'].mean(), subset['StdAttn_M2S'].mean()
            s2m_snr = s2m_avg / s2m_std if s2m_std > 0 else 0
            m2s_snr = m2s_avg / m2s_std if m2s_std > 0 else 0
            max_avg = max(s2m_avg, m2s_avg)
            balance = 1 - abs(s2m_avg - m2s_avg) / max_avg if max_avg > 0 else 0
            valid_comps = [v for v in [s2m_snr, m2s_snr, balance] if np.isfinite(v)]
            efficiency_list.append({
                'Genotype': genotype, 'Time_Point': time_point,
                'Efficiency': np.mean(valid_comps) if valid_comps else 0
            })
        return pd.DataFrame(efficiency_list) if efficiency_list else None
    
    def _setup_panel_axis(self, ax: plt.Axes, title: str, panel_label: str):
        """Applies consistent styling to a panel's axis."""
        fonts = self.config['style']['fonts']
        colors = self.config['style']['colors']
        ax.set_title(title, fontsize=fonts['panel_title'], loc='left', pad=20)
        ax.text(-0.12, 1.1, panel_label.lower(), transform=ax.transAxes, 
                fontsize=fonts['panel_label'], fontweight='bold', va='top')
        ax.set_facecolor(colors['Panel_BG'])
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color=colors['Grid'])
        ax.tick_params(axis='both', which='major', labelsize=fonts['tick_label'])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['Text'])

    def _plot_dynamics(self, ax: plt.Axes, data: pd.DataFrame, y_col: str, y_err_col: str, y_label: str):
        """Generic plotting function for dynamics panels."""
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', color='gray')
            return
            
        colors = self.config['style']['colors']
        for (tissue, genotype), subset in data.groupby(['Tissue', 'Genotype']):
            style = {'linestyle': '-' if tissue == 'Leaf' else '--', 
                     'marker': 'o' if tissue == 'Leaf' else '^'}
            ax.errorbar(subset['Time_Point'], subset[y_col], yerr=subset[y_err_col],
                        color=colors[genotype], label=f'{genotype} {tissue}', 
                        linewidth=2.5, markersize=7, capsize=4, alpha=0.9, **style)

        ax.set_xlabel('Time Point', fontsize=self.config['style']['fonts']['axis_label'])
        ax.set_ylabel(y_label, fontsize=self.config['style']['fonts']['axis_label'])
        ax.legend(fontsize=self.config['style']['fonts']['legend'])
        ax.set_xticks(sorted(data['Time_Point'].unique()))

    def plot_panel_a(self, ax: plt.Axes, data: Optional[pd.DataFrame]):
        """Plots S→M Attention Structure Dynamics (Coordination Focus)."""
        self._setup_panel_axis(ax, 'S→M Attention Structure Dynamics', 'a')
        self._plot_dynamics(ax, data, 'StdAttn_Mean', 'StdAttn_SEM', 
                            'Network Coordination Focus\n(Mean SD of S→M Attention)')
    
    def plot_panel_b(self, ax: plt.Axes, data: Optional[pd.DataFrame]):
        """Plots S→M Attention Intensity Dynamics (Peak Connection Strength)."""
        self._setup_panel_axis(ax, 'S→M Attention Intensity Dynamics', 'b')
        self._plot_dynamics(ax, data, 'P95Attn_Mean', 'P95Attn_SEM',
                            'Peak Connection Strength\n(Mean 95th Percentile of S→M Attention)')

    def plot_panel_c(self, ax: plt.Axes, data: Optional[pd.DataFrame]):
        """Plots Predictive Feature Integration (SHAP vs. Attention)."""
        self._setup_panel_axis(ax, 'Predictive Feature Integration', 'c')
        ax.set_xlabel('Predictive Power (SHAP Importance)', fontsize=self.config['style']['fonts']['axis_label'])
        ax.set_ylabel('Coordination Strength (Total Attention)', fontsize=self.config['style']['fonts']['axis_label'])

        if data is None or data.empty:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', color='gray'); return

        style = self.config['style']
        palette = {'Spectral': style['colors']['Spectral'], 'Molecular': style['colors']['Molecular'], 'Unknown': 'gray'}

        # Background density plot
        sns.kdeplot(data=data, x='SHAP_Importance', y='Attention_Score', ax=ax,
                    fill=True, cmap="Greys", alpha=0.4, levels=5, zorder=1)

        # Main scatter plot
        sns.scatterplot(data=data, x='SHAP_Importance', y='Attention_Score', hue='FeatureType',
                        palette=palette, alpha=0.7, s=60, edgecolor='w', linewidth=0.5, ax=ax, zorder=2)
        
        # Identify and highlight Key Hubs
        shap_thresh = data['SHAP_Importance'].quantile(0.95)
        attn_thresh = data['Attention_Score'].quantile(0.95)
        hubs = data[(data['SHAP_Importance'] > shap_thresh) & (data['Attention_Score'] > attn_thresh)]
        hubs = hubs.sort_values(by=['SHAP_Importance', 'Attention_Score'], ascending=[False, False])
        
        if not hubs.empty:
            ax.scatter(hubs['SHAP_Importance'], hubs['Attention_Score'], s=150,
                       facecolors='none', edgecolors=style['colors']['Hub'], linewidth=2.5,
                       label='Key Hubs', zorder=3)
            
            # Annotate top 5 hubs
            top_5_hubs = hubs.head(5)
            for i, (_, hub) in enumerate(top_5_hubs.iterrows()):
                feature_label = str(hub['Feature']).replace('Cluster_', '')
                ax.annotate(feature_label,
                            xy=(hub['SHAP_Importance'], hub['Attention_Score']),
                            xytext=(30, (i - 2) * 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color=style['colors']['Text']),
                            fontsize=style['fonts']['annotation'], fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title='Feature Type', fontsize=style['fonts']['legend'])
        ax.set_xlim(left=-0.01, right=data['SHAP_Importance'].max() * 1.15)
        ax.set_ylim(bottom=-0.1, top=data['Attention_Score'].max() * 1.15)
    
    def plot_panel_d(self, ax: plt.Axes, data: Optional[pd.DataFrame]):
        """Plots Coordination Efficiency Landscape."""
        self._setup_panel_axis(ax, 'Coordination Efficiency Landscape', 'd')
        ax.set_xlabel('Time Point', fontsize=self.config['style']['fonts']['axis_label'])
        ax.set_ylabel('Genotype', fontsize=self.config['style']['fonts']['axis_label'])
        ax.grid(False) # Turn off grid for heatmap

        if data is None or data.empty:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', color='gray'); return

        pivot_data = data.pivot_table(index='Genotype', columns='Time_Point', values='Efficiency')
        cmap = LinearSegmentedColormap.from_list('efficiency', self.config['style']['colors']['Efficiency_CMap'])
        
        im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_xticklabels([f'{int(tp)}' for tp in pivot_data.columns])
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)

        # Add value annotations
        for i, idx in enumerate(pivot_data.index):
            for j, col in enumerate(pivot_data.columns):
                value = pivot_data.loc[idx, col]
                text_color = 'black' if value < 0.65 else 'white'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color,
                        fontsize=self.config['style']['fonts']['annotation'], fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.9, aspect=20, pad=0.03)
        cbar.set_label('Coordination Efficiency Score', fontsize=self.config['style']['fonts']['legend'])

    def generate_figure(self):
        """Orchestrates data preparation, plotting, and saving."""
        if not self.load_data():
            return # Abort if data loading fails
            
        dynamics_data = self._prepare_dynamics_data()
        integration_data = self._prepare_integration_data()
        efficiency_data = self._prepare_efficiency_data()

        logger.info("Assembling final figure...")
        fig, axes = plt.subplots(1, 4, figsize=self.config['style']['figure']['size'])
        
        self.plot_panel_a(axes[0], dynamics_data)
        self.plot_panel_b(axes[1], dynamics_data)
        self.plot_panel_c(axes[2], integration_data)
        self.plot_panel_d(axes[3], efficiency_data)
        
        plt.tight_layout(pad=1.0, w_pad=2.5)

        # Save figure and data
        output_dir = self.config['paths']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path_base = output_dir / "Figure5_Publication_Ready"
        
        try:
            fig.savefig(f"{save_path_base}.png", dpi=self.config['style']['figure']['dpi'], bbox_inches='tight')
            fig.savefig(f"{save_path_base}.pdf", format='pdf', bbox_inches='tight')
            fig.savefig(f"{save_path_base}.svg", format='svg', bbox_inches='tight')
            logger.info(f"Figure saved successfully to {output_dir}")
            
            # Export underlying data
            if dynamics_data is not None: dynamics_data.to_csv(output_dir / "Figure5_Panel_AB_Data.csv", index=False)
            if integration_data is not None: integration_data.to_csv(output_dir / "Figure5_Panel_C_Data.csv", index=False)
            if efficiency_data is not None: efficiency_data.to_csv(output_dir / "Figure5_Panel_D_Data.csv", index=False)
            logger.info("Supporting data exported for reproducibility.")

        except Exception as e:
            logger.error(f"Failed to save figure or data: {e}")
            
        plt.show()

def main():
    """Main execution function."""
    logger.info("--- Starting Figure 5 Generation ---")
    
    # --- IMPORTANT ---
    # The user must update this path to the root directory of the project data.
    BASE_PROJECT_PATH = r"C:\Users\ms\Desktop\hyper"
    
    if not Path(BASE_PROJECT_PATH).exists():
        logger.error(f"The specified BASE_PROJECT_PATH does not exist: {BASE_PROJECT_PATH}")
        logger.error("Please update the path in the main() function and run the script again.")
        return

    # Initialize and run the generator
    fig_generator = Figure5Generator(BASE_PROJECT_PATH)
    fig_generator.generate_figure()
    
    logger.info("--- Figure 5 Generation Complete ---")

if __name__ == "__main__":
    main()