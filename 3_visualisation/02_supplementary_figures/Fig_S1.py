#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOFA+ Cross-View Integration Network Visualization

This script creates a network visualization from MOFA+ output files to display
relationships between features across different data modalities (views). It loads
a MOFA+ model file and corresponding metadata, then generates a network graph
showing connections between factors and important features.

Usage:
    python Fig_S1.py
"""

import os
import sys
import traceback
from datetime import datetime
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Configuration & Style Settings ---

# Color palette for consistent visualization
COLORS = {
    # Core Experimental Variables
    'G1': '#00FA9A',             # Tolerant Genotype
    'G2': '#48D1CC',             # Susceptible Genotype
    'T0': '#4682B4',             # Control Treatment
    'T1': '#BDB76B',             # Stress Treatment
    'Leaf': '#00FF00',           # Leaf Tissue
    'Root': '#40E0D0',           # Root Tissue
    'Day1': '#ffffcc',
    'Day2': '#9CBA79',
    'Day3': '#3e7d5a',

    # Data Types / Omics / Features
    'Spectral': '#ECDA79',
    'Metabolite': '#84ab92',
    'UnknownFeature': '#B0E0E6',
    'Spectral_Water': '#6DCAFA',
    'Spectral_Pigment': '#00FA9A',
    'Spectral_Structure': '#7fcdbb',
    'Spectral_SWIR': '#636363',
    'Spectral_VIS': '#c2e699',
    'Spectral_RedEdge': '#78c679',
    'Spectral_UV': '#00BFFF',
    'Spectral_Other': '#969696',
    'Metabolite_PCluster': '#3DB3BF',
    'Metabolite_NCluster': '#ffffd4',
    'Metabolite_Other': '#bdbdbd',

    # Methods & Model Comparison
    'MOFA': '#FFEBCD',
    'SHAP': '#F0E68C',
    'Overlap': '#AFEEEE',
    'Transformer': '#fae3a2',
    'RandomForest': '#40E0D0',
    'KNN': '#729c87',

    # Network Visualization Elements
    'Edge_Low': '#f0f0f0',
    'Edge_High': '#CDBE70',
    'Node_Spectral': '#6baed6',
    'Node_Metabolite': '#FFC4A1',
    'Node_Edge': '#252525',

    # Statistical & Difference Indicators
    'Positive_Diff': '#66CDAA',
    'Negative_Diff': '#fe9929',
    'Significance': '#08519c',
    'NonSignificant': '#bdbdbd',
    'Difference_Line': '#636363',

    # Plot Elements & Annotations
    'Background': '#FFFFFF',
    'Panel_Background': '#f7f7f7',
    'Grid': '#d9d9d9',
    'Text_Dark': '#252525',
    'Text_Light': '#FFFFFF',
    'Text_Annotation': '#000000',
    'Annotation_Box_BG': '#FFFFFF',
    'Annotation_Box_Edge': '#bdbdbd',
    'Table_Header_BG': '#deebf7',
    'Table_Highlight_BG': '#fff7bc',

    # Temporal Patterns
    'Pattern_Increasing': '#238b45',
    'Pattern_Decreasing': '#fe9929',
    'Pattern_Peak': '#78c679',
    'Pattern_Valley': '#6baed6',
    'Pattern_Stable': '#969696',
}

# Font settings for consistent visualization
FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'main_title': 26,
    'panel_label': 23,
    'panel_title': 21,
    'axis_label': 21,
    'tick_label': 20,
    'legend_title': 23,
    'legend_text': 20,
    'annotation': 19,
    'caption': 19,
    'table_header': 19,
    'table_cell': 19,
}

# Set up plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = FONTS_SANS['family']
plt.rcParams['font.sans-serif'] = FONTS_SANS['sans_serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.labelsize'] = FONTS_SANS['tick_label']
plt.rcParams['ytick.labelsize'] = FONTS_SANS['tick_label']

# Define colors for different data views
VIEW_COLORS = {
    'Leaf_spectral': COLORS['Spectral_VIS'],
    'Root_spectral': COLORS['Spectral_Water'],
    'Leaf_molecular feature': COLORS['Metabolite'],
    'Root_molecular feature': COLORS['Metabolite_PCluster']
}

def safe_decode(byte_string):
    """
    Safely decodes byte strings to UTF-8.

    Args:
        byte_string: Input that might be a byte string.

    Returns:
        Decoded string or original input if not bytes.
    """
    if isinstance(byte_string, bytes):
        try:
            return byte_string.decode('utf-8')
        except UnicodeDecodeError:
            return byte_string.decode('latin-1', errors='replace')
    return byte_string

def load_data(mofa_file, metadata_file):
    """
    Load data from MOFA+ HDF5 file and metadata CSV.
    
    Args:
        mofa_file (str): Path to the HDF5 model file.
        metadata_file (str): Path to the metadata CSV file.
        
    Returns:
        dict: Dictionary containing processed model data and metadata, or None if loading fails.
    """
    print(f"Loading data from:\n  Model: {mofa_file}\n  Metadata: {metadata_file}")
    results = {}
    metadata = None

    # --- Metadata Loading ---
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} does not exist.")
        return None
    try:
        metadata = pd.read_csv(metadata_file)
        # Check for essential columns
        required_meta_cols = ['Genotype', 'Treatment', 'Day', 'Tissue.type', 'Batch']
        missing_cols = [col for col in required_meta_cols if col not in metadata.columns]
        if missing_cols:
            print(f"Warning: Metadata missing required columns: {missing_cols}")
        
        results['metadata'] = metadata
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        traceback.print_exc()
        return None

    # --- MOFA+ Model Loading ---
    if not os.path.exists(mofa_file):
        print(f"Error: MOFA+ model file {mofa_file} not found.")
        return None
    try:
        with h5py.File(mofa_file, 'r') as f:
            # Extract views
            if 'views' in f and 'views' in f['views']:
                views_data = f['views']['views'][()]
                views = [safe_decode(v) for v in views_data]
                results['views'] = views
            else:
                print("Error: 'views/views' dataset not found in HDF5 file.")
                results['views'] = []

            # Extract samples
            if 'samples' in f and 'group0' in f['samples']:
                sample_data = f['samples']['group0'][()]
                results['samples'] = {'group0': [safe_decode(s) for s in sample_data]}
                num_samples = len(results['samples']['group0'])
                
                if metadata is not None and len(metadata) != num_samples:
                    print(f"Error: Metadata row count ({len(metadata)}) does not match MOFA sample count ({num_samples}).")
                    return None
            else:
                print("Error: 'samples/group0' not found in HDF5 file.")
                return None

            # Extract factors (Z)
            if 'expectations' in f and 'Z' in f['expectations'] and 'group0' in f['expectations']['Z']:
                z_data = f['expectations']['Z']['group0'][()]
                results['factors'] = z_data.T
                if results['factors'].shape[0] != num_samples:
                    print(f"Error: Factor matrix row count ({results['factors'].shape[0]}) does not match sample count ({num_samples}).")
                    return None
            else:
                print("Error: 'expectations/Z/group0' not found.")
                results['factors'] = None

            # Extract weights (W)
            if 'expectations' in f and 'W' in f['expectations']:
                weights = {}
                expected_factors = results['factors'].shape[1] if results.get('factors') is not None else None
                for view_key in f['expectations']['W']:
                    view_name = safe_decode(view_key)
                    w_data = f['expectations']['W'][view_key][()]
                    weights[view_name] = w_data
                    if expected_factors is not None and w_data.shape[0] != expected_factors:
                        print(f"Warning: Weight matrix for view '{view_name}' has {w_data.shape[0]} factors, expected {expected_factors}.")
                results['weights'] = weights
            else:
                print("Error: 'expectations/W' not found.")
                results['weights'] = {}

            # Extract feature names
            if 'features' in f:
                features = {}
                for view_key in f['features']:
                    view_name = safe_decode(view_key)
                    try:
                        feature_data = f['features'][view_key][()]
                        features[view_name] = [safe_decode(feat) for feat in feature_data]
                        if view_name in results['weights'] and len(features[view_name]) != results['weights'][view_name].shape[1]:
                            print(f"Warning: Feature count for view '{view_name}' ({len(features[view_name])}) doesn't match weight matrix dimension.")
                    except Exception as e:
                        print(f"Error extracting features for {view_name}: {e}")
                results['features'] = features
            else:
                print("Error: 'features' group not found.")
                results['features'] = {}

            # Extract variance explained
            variance_dict = {}
            if 'variance_explained' in f:
                # Extract r2_total
                if 'r2_total' in f['variance_explained'] and 'group0' in f['variance_explained']['r2_total']:
                    try:
                        r2_total_data = f['variance_explained']['r2_total']['group0'][()]
                        if len(r2_total_data) == len(results.get('views', [])):
                            variance_dict['r2_total_per_view'] = r2_total_data
                        else:
                            variance_dict['r2_total_raw'] = r2_total_data
                    except Exception as e:
                        print(f"Error extracting r2_total: {e}")

                # Extract r2_per_factor
                r2_pf_path = 'variance_explained/r2_per_factor/group0'
                if r2_pf_path in f:
                    try:
                        r2_per_factor_data = f[r2_pf_path][()]
                        n_views = len(results.get('views', []))
                        n_factors = results.get('factors', np.array([])).shape[1]

                        if r2_per_factor_data.shape == (n_views, n_factors):
                            variance_dict['r2_per_factor'] = r2_per_factor_data
                        else:
                            if r2_per_factor_data.shape == (n_factors, n_views):
                                variance_dict['r2_per_factor'] = r2_per_factor_data.T
                            else:
                                variance_dict['r2_per_factor_raw'] = r2_per_factor_data
                    except Exception as e:
                        print(f"Error extracting r2_per_factor from {r2_pf_path}: {e}")
                
            results['variance'] = variance_dict

        # --- Post-Loading Processing ---
        # Create combined factor + metadata DataFrame
        if results.get('factors') is not None and metadata is not None:
            factors = results['factors']
            factor_cols = [f"Factor{i+1}" for i in range(factors.shape[1])]
            factors_df = pd.DataFrame(factors, columns=factor_cols)

            # Add all metadata columns
            meta_cols_to_add = [col for col in metadata.columns if col not in factors_df.columns]
            for col in meta_cols_to_add:
                try:
                    factors_df[col] = metadata[col].values
                except ValueError as ve:
                    print(f"Error adding metadata column '{col}': {ve}. Length mismatch likely.")
                    return None

            results['factors_df'] = factors_df

        # Calculate feature importance
        if results.get('weights') and results.get('features'):
            feature_importance = {}
            weights = results['weights']
            features = results['features']

            for view_name, view_weights in weights.items():
                if view_name not in features:
                    continue

                view_features = features[view_name]
                if len(view_features) != view_weights.shape[1]:
                    continue

                # Calculate importance as sum of absolute weights across factors
                importance = np.sum(np.abs(view_weights), axis=0)
                importance_df = pd.DataFrame({
                    'Feature': view_features,
                    'Importance': importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                feature_importance[view_name] = importance_df

            results['feature_importance'] = feature_importance

    except Exception as e:
        print(f"Error during HDF5 loading or processing: {e}")
        traceback.print_exc()
        return None

    # Check for essential components
    essential = ['views', 'samples', 'factors', 'weights', 'features', 'metadata', 'factors_df']
    missing_essential = []
    for item in essential:
        if item not in results or results[item] is None:
            missing_essential.append(item)
            continue
        
        value = results[item]
        is_empty = False
        if isinstance(value, (dict, list)) and not value:
            is_empty = True
        elif isinstance(value, pd.DataFrame) and value.empty:
            is_empty = True
        elif isinstance(value, np.ndarray) and value.size == 0:
            is_empty = True

        if is_empty:
            missing_essential.append(item)

    if missing_essential:
        print(f"Error: Missing essential data components: {missing_essential}")
        return None

    return results


def create_cross_view_integration(data, output_dir):
    """
    Create a network visualization showing relationships between views.
    
    Args:
        data (dict): Dictionary containing MOFA+ model data.
        output_dir (str): Directory to save output files.
        
    Returns:
        str: Path to output file if successful, None otherwise.
    """
    print("\n--- Creating cross-view integration visualization ---")

    # --- Data Validation ---
    if 'weights' not in data or not data['weights']:
        print("Error: Weights data not available.")
        return None
    if 'features' not in data or not data['features']:
        print("Error: Features data not available.")
        return None
    if 'views' not in data or len(data['views']) < 2:
        print(f"Error: Need at least 2 views for integration. Found: {len(data.get('views', []))}")
        return None

    weights = data['weights']
    features = data['features']
    views = data['views']
    n_views = len(views)
    
    # Infer n_factors from first weight matrix
    first_view = next(iter(weights.keys()))
    n_factors = weights[first_view].shape[0]

    print(f"Integrating {n_views} views across {n_factors} factors.")

    try:
        # --- Set up Figure Layout ---
        fig = plt.figure(figsize=(20, 16))
        ax_net = fig.add_subplot(111)
        fig.suptitle("Cross-View Integration", fontsize=FONTS_SANS['main_title'], fontweight='bold')

        # --- Cross-View Network Construction ---
        G = nx.Graph()
        MAX_FEATURES_PER_VIEW = 15  # Limit features shown per view for clarity

        # Add Factor nodes
        factor_nodes = [f"Factor {i+1}" for i in range(n_factors)]
        for factor_name in factor_nodes:
            G.add_node(factor_name, node_type='factor', size=1200)

        # Add Top Feature nodes and Factor-Feature edges
        for view in views:
            if view not in weights or view not in features: continue
            view_weights_matrix = weights[view]  # factors x features
            view_feature_list = features[view]

            if len(view_feature_list) != view_weights_matrix.shape[1]:
                print(f"Warning: Feature/Weight mismatch for {view}. Skipping network contributions.")
                continue

            # Calculate feature importance
            importance = np.sum(np.abs(view_weights_matrix), axis=0)
            top_indices = np.argsort(importance)[-MAX_FEATURES_PER_VIEW:]
            
            # Get appropriate node color for this view
            view_display = view.lower()
            if 'metabolite' in view_display:
                view_display = view_display.replace('metabolite', 'molecular feature')
            if 'leaf' in view_display:
                view_display = view_display.replace('leaf', 'Leaf')
            if 'root' in view_display:
                view_display = view_display.replace('root', 'Root')
                
            if view_display in VIEW_COLORS:
                node_color = VIEW_COLORS[view_display]
            elif view in COLORS:
                node_color = COLORS[view]
            elif 'spectral' in view_display:
                node_color = COLORS['Node_Spectral']
            elif 'molecular feature' in view_display:
                node_color = COLORS['Node_Metabolite']
            else:
                node_color = 'blue'
                
            # Add nodes and edges for these top features
            for feature_idx in top_indices:
                feature_name = view_feature_list[feature_idx]
                
                # Add feature node
                display_name = feature_name.replace(f"_{view}", "")
                display_name = display_name.replace("Cluster_", "")
                if len(display_name) > 12: display_name = display_name[:10] + ".."
                
                G.add_node(feature_name, node_type='feature', view=view,
                           importance=importance[feature_idx], display_name=display_name, 
                           size=400, color=node_color)

                # Add edges to factors based on weights
                for factor_idx in range(n_factors):
                    weight = view_weights_matrix[factor_idx, feature_idx]
                    factor_weights_for_view = view_weights_matrix[factor_idx, :]
                    weight_threshold = np.percentile(np.abs(factor_weights_for_view), 95)  # Connect top 5%

                    if abs(weight) > weight_threshold and abs(weight) > 0.05:
                        factor_name = f"Factor {factor_idx+1}"
                        G.add_edge(feature_name, factor_name, weight=abs(weight), type='factor_feature')

        # Add edges between features (based on co-association with factors)
        for factor_node in factor_nodes:
            connected_features = [n for n in G.neighbors(factor_node) if G.nodes[n]['node_type'] == 'feature']
            features_by_view = {}
            for feat in connected_features:
                view = G.nodes[feat]['view']
                if view not in features_by_view: features_by_view[view] = []
                features_by_view[view].append(feat)

            # Connect top features between pairs of views
            for view1, view2 in combinations(features_by_view.keys(), 2):
                for feat1 in features_by_view[view1]:
                    for feat2 in features_by_view[view2]:
                        if not G.has_edge(feat1, feat2):
                            G.add_edge(feat1, feat2, weight=0.5, type='cross_view')

        # --- Network Layout ---
        try:
            fixed_nodes = factor_nodes
            pos_factors = nx.circular_layout(G.subgraph(factor_nodes))
            pos = nx.spring_layout(G, pos=pos_factors, fixed=fixed_nodes, k=0.3, iterations=100, seed=42)
        except Exception as layout_err:
            print(f"Warning: Network layout failed ({layout_err}). Falling back to random layout.")
            pos = nx.random_layout(G)

        # --- Draw Network ---
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'factor':
                node_colors.append(COLORS['Table_Header_BG'])
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = node
            elif G.nodes[node]['node_type'] == 'feature':
                node_colors.append(G.nodes[node].get('color', COLORS['UnknownFeature']))
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = G.nodes[node]['display_name']
            else:
                node_colors.append('red')
                node_sizes.append(200)
                node_labels[node] = node[:5] + ".."

        factor_feature_edges = []
        cross_view_edges = []

        for u, v, d in G.edges(data=True):
            edge_type = d.get('type', 'unknown')
            if edge_type == 'factor_feature':
                factor_feature_edges.append((u, v))
            elif edge_type == 'cross_view':
                cross_view_edges.append((u, v))
            else:
                factor_feature_edges.append((u, v))

        if cross_view_edges:
            widths_cv = [G[u][v].get('weight', 0.5) * 2 for u, v in cross_view_edges]
            nx.draw_networkx_edges(G, pos, edgelist=cross_view_edges, width=widths_cv,
                                  edge_color=COLORS['Edge_High'], style='dashed', alpha=0.6, ax=ax_net)
                                  
        if factor_feature_edges:
            widths_ff = [G[u][v].get('weight', 0.1) * 5 for u, v in factor_feature_edges]
            nx.draw_networkx_edges(G, pos, edgelist=factor_feature_edges, width=widths_ff,
                                  edge_color=COLORS['Edge_High'], style='solid', alpha=0.8, ax=ax_net)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.9, edgecolors=COLORS['Node_Edge'], linewidths=0.5, ax=ax_net)
                              
        nx.draw_networkx_labels(G, pos, labels=node_labels, 
                               font_size=FONTS_SANS['annotation'], ax=ax_net)

        # --- Enhance Network Plot Appearance ---
        ax_net.set_title('Cross-View Feature Integration Network (Top Features)', 
                        fontsize=FONTS_SANS['panel_title'], fontweight='bold')
        ax_net.axis('off')

        # Add legend
        legend_handles = []
        for view in set(d['view'] for n, d in G.nodes(data=True) if 'view' in d):
            view_display = view.lower()
            if 'metabolite' in view_display:
                view_display = view_display.replace('metabolite', 'molecular feature')
            if 'leaf' in view_display:
                view_display = view_display.replace('leaf', 'Leaf')
            if 'root' in view_display:
                view_display = view_display.replace('root', 'Root')
                
            if view_display in VIEW_COLORS:
                color = VIEW_COLORS[view_display]
            elif view in COLORS:
                color = COLORS[view]
            elif 'spectral' in view_display:
                color = COLORS['Node_Spectral']
            elif 'molecular feature' in view_display:
                color = COLORS['Node_Metabolite']
            else:
                color = 'blue'
                
            display_label = view
            if 'metabolite' in display_label.lower():
                display_label = display_label.replace('metabolite', 'molecular feature')
                display_label = display_label.replace('Metabolite', 'Molecular feature')
            if 'leaf' in display_label.lower():
                display_label = display_label.replace('leaf', 'Leaf')
            if 'root' in display_label.lower():
                display_label = display_label.replace('root', 'Root')
                
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{display_label} Feature',
                                            markerfacecolor=color, markersize=20))
                                            
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Factor',
                                         markerfacecolor=COLORS['Table_Header_BG'], markersize=24))
        legend_handles.append(plt.Line2D([0], [0], color=COLORS['Edge_High'], lw=2, 
                                        label='Factor-Feature Link'))
                                        
        if cross_view_edges:
            legend_handles.append(plt.Line2D([0], [0], color=COLORS['Edge_High'], lw=1.5, 
                                           linestyle='--', label='Inferred Cross-View Link'))

        ax_net.legend(handles=legend_handles, loc='lower right', 
                     fontsize=FONTS_SANS['legend_text'], frameon=True, 
                     facecolor=COLORS['Annotation_Box_BG'], framealpha=0.7)

        # --- Save Plot ---
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved cross-view integration plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def main():
    """Main function to run the visualization."""
    # File paths (hardcoded as requested)
    mofa_file = "C:/Users/ms/Desktop/hyper/output/mofa/mofa_model_for_transformer.hdf5"
    metadata_file = "C:/Users/ms/Desktop/hyper/output/mofa/aligned_combined_metadata.csv"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MOFA+ Cross-View Integration Visualization - Start")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load data
    data = load_data(mofa_file, metadata_file)
    if data is None:
        print("\nError: Data loading failed.")
        return 1

    # Generate the visualization
    result = create_cross_view_integration(data, output_dir)
    
    print("\n" + "=" * 80)
    if result:
        print("Visualization generated successfully.")
    else:
        print("Visualization generation failed.")
    print("=" * 80)
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())