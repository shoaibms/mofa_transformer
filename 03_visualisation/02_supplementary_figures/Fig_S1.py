#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOFA+ Cross-View Integration Network Visualization

This script creates a publication-quality network visualization from MOFA+ output files
to display the relationships between features across different data modalities (views).
It loads a MOFA+ model file and corresponding metadata, then generates a network graph
showing connections between factors and important features from each view.

The visualization helps in understanding how different omics data types are integrated
in the MOFA+ model and identifies key features driving the integration.

Usage:
    python enhanced_mofa_viz_network_only.py
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
from scipy.stats import spearmanr 
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
import traceback
from datetime import datetime
import networkx as nx
from itertools import combinations

# Import color palette and font settings
# Color palette for consistent visualization
COLORS = {
    # Core Experimental Variables
    'G1': '#00FA9A',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#48D1CC',             # Susceptible Genotype (Medium Teal)
    'T0': '#4682B4',             # Control Treatment (Medium Green)
    'T1': '#BDB76B',             # Stress Treatment (Muted Orange/Yellow)
    'Leaf': '#00FF00',           # Leaf Tissue (Darkest Green)
    'Root': '#40E0D0',           # Root Tissue (Darkest Blue)
    'Day1': '#ffffcc',           # Very Light Yellow-Green
    'Day2': '#9CBA79',           # Light Yellow-Green
    'Day3': '#3e7d5a',           # Medium Yellow-Green

    # Data Types / Omics / Features
    'Spectral': '#ECDA79',       # General Spectral (Medium Blue)
    'Metabolite': '#84ab92',     # General Metabolite (Medium-Dark Yellow-Green)
    'UnknownFeature': '#B0E0E6', # Medium Grey for fallback
    'Spectral_Water': '#6DCAFA', # Medium-Dark Blue
    'Spectral_Pigment': '#00FA9A', # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb', # Medium Teal
    'Spectral_SWIR': '#636363',   # Dark Grey
    'Spectral_VIS': '#c2e699',    # Light Yellow-Green
    'Spectral_RedEdge': '#78c679', # Medium Yellow-Green
    'Spectral_UV': '#00BFFF',     # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',  # Medium Grey
    'Metabolite_PCluster': '#3DB3BF', # Darkest Yellow-Green
    'Metabolite_NCluster': '#ffffd4', # Very Light Yellow
    'Metabolite_Other': '#bdbdbd', # Light Grey

    # Methods & Model Comparison
    'MOFA': '#FFEBCD',           # Dark Blue
    'SHAP': '#F0E68C',           # Dark Green
    'Overlap': '#AFEEEE',        # Medium-Dark Yellow-Green
    'Transformer': '#fae3a2',    # Medium Blue
    'RandomForest': '#40E0D0',   # Medium Green
    'KNN': '#729c87',            # Medium Teal

    # Network Visualization Elements
    'Edge_Low': '#f0f0f0',       # Very Light Gray
    'Edge_High': '#CDBE70',      # Darker Pale Gold
    'Node_Spectral': '#6baed6',  # Default Spectral Node (Medium Blue)
    'Node_Metabolite': '#FFC4A1', # Default Metabolite Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',      # Darkest Gray / Near Black border

    # Statistical & Difference Indicators
    'Positive_Diff': '#66CDAA',  # Medium-Dark Green
    'Negative_Diff': '#fe9929',  # Muted Orange/Yellow (Matches T1)
    'Significance': '#08519c',   # Dark Blue (for markers/text)
    'NonSignificant': '#bdbdbd', # Light Grey
    'Difference_Line': '#636363', # Dark Grey line

    # Plot Elements & Annotations
    'Background': '#FFFFFF',     # White plot background
    'Panel_Background': '#f7f7f7', # Very Light Gray background for some panels
    'Grid': '#d9d9d9',           # Lighter Gray grid lines
    'Text_Dark': '#252525',      # Darkest Gray / Near Black text
    'Text_Light': '#FFFFFF',     # White text
    'Text_Annotation': '#000000', # Black text for annotations
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd', # Light Grey border for text boxes
    'Table_Header_BG': '#deebf7', # Very Light Blue table header
    'Table_Highlight_BG': '#fff7bc', # Pale Yellow for highlighted table cells

    # Temporal Patterns
    'Pattern_Increasing': '#238b45', # Medium-Dark Green
    'Pattern_Decreasing': '#fe9929', # Muted Orange/Yellow
    'Pattern_Peak': '#78c679',     # Medium Yellow-Green
    'Pattern_Valley': '#6baed6',   # Medium Blue
    'Pattern_Stable': '#969696',   # Medium Grey
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

# Set up plotting parameters for publication quality
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42  # Output as Type 42 (TrueType)
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
    Safely decodes byte strings, returns original if not bytes.
    
    Args:
        byte_string: Input that might be a byte string
        
    Returns:
        Decoded string or original input if not bytes
    """
    if isinstance(byte_string, bytes):
        try:
            return byte_string.decode('utf-8')
        except UnicodeDecodeError:
            return byte_string.decode('latin-1', errors='replace')
    return byte_string

def load_data(mofa_file, metadata_file):
    """Load data from MOFA+ HDF5 file and metadata CSV"""
    print(f"Loading data from {mofa_file} and {metadata_file}")
    results = {}
    metadata = None

    # --- Metadata Loading ---
    if not os.path.exists(metadata_file):
        print(f"FATAL ERROR: Metadata file {metadata_file} does not exist!")
        return None
    try:
        metadata = pd.read_csv(metadata_file)
        print(f"Metadata loaded: {len(metadata)} rows, {len(metadata.columns)} columns")
        print(f"Metadata columns: {metadata.columns.tolist()}")
        # Check for essential columns
        required_meta_cols = ['Genotype', 'Treatment', 'Day', 'Tissue.type', 'Batch']
        missing_cols = [col for col in required_meta_cols if col not in metadata.columns]
        if missing_cols:
            print(f"WARNING: Metadata missing required columns: {missing_cols}")
        else:
            print("All essential metadata columns found.")
            # Print unique values for key columns
            for col in required_meta_cols:
                unique_values = metadata[col].unique()
                print(f"  Column '{col}' unique values ({len(unique_values)}): {unique_values}")
        results['metadata'] = metadata
    except Exception as e:
        print(f"FATAL ERROR reading metadata file: {e}")
        traceback.print_exc()
        return None

    # --- MOFA+ Model Loading ---
    if not os.path.exists(mofa_file):
        print(f"FATAL ERROR: MOFA+ model file {mofa_file} not found")
        return None
    try:
        with h5py.File(mofa_file, 'r') as f:
            # Extract views
            if 'views' in f and 'views' in f['views']:
                views_data = f['views']['views'][()]
                views = [safe_decode(v) for v in views_data]
                results['views'] = views
                print(f"Found views: {views}")
            else:
                print("ERROR: 'views/views' dataset not found in HDF5 file.")
                results['views'] = []

            # Extract samples
            if 'samples' in f and 'group0' in f['samples']:
                sample_data = f['samples']['group0'][()]
                results['samples'] = {'group0': [safe_decode(s) for s in sample_data]}
                num_samples = len(results['samples']['group0'])
                print(f"Loaded {num_samples} sample names for group 'group0'")
                if metadata is not None and len(metadata) != num_samples:
                    print(f"FATAL ERROR: Metadata row count ({len(metadata)}) does not match MOFA sample count ({num_samples})!")
                    return None
            else:
                print("ERROR: 'samples/group0' not found in HDF5 file.")
                return None

            # Extract factors (Z)
            if 'expectations' in f and 'Z' in f['expectations'] and 'group0' in f['expectations']['Z']:
                z_data = f['expectations']['Z']['group0'][()]
                results['factors'] = z_data.T
                print(f"Loaded factors (Z): shape {results['factors'].shape} (samples x factors)")
                if results['factors'].shape[0] != num_samples:
                    print(f"FATAL ERROR: Factor matrix row count ({results['factors'].shape[0]}) does not match sample count ({num_samples})!")
                    return None
            else:
                print("ERROR: 'expectations/Z/group0' not found.")
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
                        print(f"WARNING: Weight matrix for view '{view_name}' has {w_data.shape[0]} factors, expected {expected_factors}.")
                results['weights'] = weights
                print(f"Loaded weights (W) for {len(weights)} views.")
            else:
                print("ERROR: 'expectations/W' not found.")
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
                            print(f"WARNING: Feature count for view '{view_name}' ({len(features[view_name])}) doesn't match weight matrix dimension ({results['weights'][view_name].shape[1]}).")
                    except Exception as e:
                        print(f"Error extracting features for {view_name}: {e}")
                results['features'] = features
                print(f"Loaded feature names for {len(features)} views.")
            else:
                print("ERROR: 'features' group not found.")
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
                            print(f"Loaded r2_total_per_view: {r2_total_data}")
                        else:
                            print(f"Warning: r2_total shape {r2_total_data.shape} mismatch with view count {len(results.get('views', []))}. Storing raw.")
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
                            print(f"Loaded r2_per_factor: shape {r2_per_factor_data.shape} (views x factors)")
                        else:
                            print(f"WARNING: r2_per_factor shape {r2_per_factor_data.shape} mismatch with expected ({n_views}, {n_factors}). Check model output.")
                            if r2_per_factor_data.shape == (n_factors, n_views):
                                variance_dict['r2_per_factor'] = r2_per_factor_data.T
                                print(f"  -> Transposed to expected shape ({n_views}, {n_factors})")
                            else:
                                variance_dict['r2_per_factor_raw'] = r2_per_factor_data
                                print(f"  -> Stored raw r2_per_factor data.")
                    except Exception as e:
                        print(f"Error extracting r2_per_factor from {r2_pf_path}: {e}")
                        traceback.print_exc()
                else:
                    print(f"Dataset {r2_pf_path} not found.")

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
                    print(f"ERROR adding metadata column '{col}': {ve}. Length mismatch likely.")
                    print(f"  Factor df length: {len(factors_df)}, Metadata column length: {len(metadata[col])}")
                    return None

            results['factors_df'] = factors_df
            print(f"Created combined factors + metadata DataFrame: {factors_df.shape}")
            print(f"  factors_df columns: {factors_df.columns.tolist()}")

        # Calculate feature importance
        if results.get('weights') and results.get('features'):
            feature_importance = {}
            weights = results['weights']
            features = results['features']

            for view_name, view_weights in weights.items():
                if view_name not in features:
                    print(f"Warning: No features found for view '{view_name}', cannot calculate importance.")
                    continue

                view_features = features[view_name]
                if len(view_features) != view_weights.shape[1]:
                    print(f"ERROR: Feature count ({len(view_features)}) != weight dim ({view_weights.shape[1]}) for view '{view_name}'. Skipping importance.")
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
            print(f"Calculated feature importance for {len(feature_importance)} views.")
    except Exception as e:
        print(f"FATAL ERROR during HDF5 loading or processing: {e}")
        traceback.print_exc()
        return None

    print("-" * 80)
    print(f"Views loaded: {results.get('views')}")
    print(f"Samples loaded: {len(results.get('samples', {}).get('group0', []))}")
    print(f"Factors loaded shape: {results.get('factors', np.array([])).shape}")
    print(f"Weights loaded for views: {list(results.get('weights', {}).keys())}")
    print(f"Features loaded for views: {list(results.get('features', {}).keys())}")
    print(f"Variance data keys: {list(results.get('variance', {}).keys())}")
    print(f"Factors + Metadata DF shape: {results.get('factors_df', pd.DataFrame()).shape}")
    print(f"Feature Importance calculated for views: {list(results.get('feature_importance', {}).keys())}")
    print("-" * 80)

    # Check for essential components
    essential = ['views', 'samples', 'factors', 'weights', 'features', 'metadata', 'factors_df']
    missing_essential = []
    for item in essential:
        if item not in results or results[item] is None:
            missing_essential.append(item)
            continue

        value = results[item]
        is_empty = False
        if isinstance(value, (dict, list)):
            if not value:
                is_empty = True
        elif isinstance(value, pd.DataFrame):
            if value.empty:
                is_empty = True
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                is_empty = True

        if is_empty:
            missing_essential.append(item)

    if missing_essential:
        print(f"FATAL ERROR: Missing or empty essential data components after loading: {missing_essential}")
        print("Detailed Check:")
        for item in essential:
            status = "Missing/None"
            if item in results and results[item] is not None:
                value = results[item]
                if isinstance(value, (dict, list)) and not value: status="Empty dict/list"
                elif isinstance(value, pd.DataFrame) and value.empty: status="Empty DataFrame"
                elif isinstance(value, np.ndarray) and value.size == 0: status="Empty ndarray"
                else: status="Present"
            print(f"  - {item}: {status}")
        return None

    print("SUCCESS: All essential data components loaded successfully.")
    return results


def create_cross_view_integration(data, output_dir):
    """
    Create a network visualization showing relationships between views.
    
    Args:
        data: Dictionary containing MOFA+ model data
        output_dir: Directory to save output files
        
    Returns:
        Path to output file if successful, None otherwise
    """
    print("\n--- Creating cross-view integration visualization ---")

    # --- Data Validation ---
    if 'weights' not in data or not data['weights']:
        print("ERROR: Weights data not available.")
        return None
    if 'features' not in data or not data['features']:
        print("ERROR: Features data not available.")
        return None
    if 'views' not in data or len(data['views']) < 2:
        print(f"ERROR: Need at least 2 views for integration. Found: {len(data.get('views', []))}")
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
        fig = plt.figure(figsize=(20, 16))  # Single figure for network only
        ax_net = fig.add_subplot(111)  # Single axes for network
        fig.suptitle("Cross-View Integration", fontsize=FONTS_SANS['main_title'], fontweight='bold')

        # --- 2. Cross-View Network (Panel B - ax_net) ---
        G = nx.Graph()
        top_features_per_view = {}
        MAX_FEATURES_PER_VIEW = 15  # Limit features shown per view for clarity

        # Add Factor nodes first
        factor_nodes = [f"Factor {i+1}" for i in range(n_factors)]
        for factor_name in factor_nodes:
            G.add_node(factor_name, node_type='factor', size=1200)  # Larger size for factors

        # Add Top Feature nodes and Factor-Feature edges
        max_abs_weight_overall = 0  # For scaling edge widths later
        for view in views:
            if view not in weights or view not in features: continue
            view_weights_matrix = weights[view]  # factors x features
            view_feature_list = features[view]

            if len(view_feature_list) != view_weights_matrix.shape[1]:
                print(f"Warning: Feature/Weight mismatch for {view}. Skipping network contributions.")
                continue

            # Calculate feature importance (sum abs weight across factors)
            importance = np.sum(np.abs(view_weights_matrix), axis=0)
            top_indices = np.argsort(importance)[-MAX_FEATURES_PER_VIEW:]  # Indices of top features for this view
            top_features_per_view[view] = [view_feature_list[i] for i in top_indices]

            # Get appropriate node color for this view
            view_display = view.lower()
            # Replace metabolite with molecular feature for display purposes
            if 'metabolite' in view_display:
                view_display = view_display.replace('metabolite', 'molecular feature')
            # Capitalize Leaf and Root
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
                node_color = 'blue'  # Default fallback
                
            # Add nodes and edges for these top features
            for feature_idx in top_indices:
                feature_name = view_feature_list[feature_idx]
                # Add feature node
                display_name = feature_name.replace(f"_{view}", "")  # Cleaner label
                display_name = display_name.replace("Cluster_", "")  # Remove "Cluster_" from labels
                if len(display_name) > 12: display_name = display_name[:10]+".."
                G.add_node(feature_name, node_type='feature', view=view,
                            importance=importance[feature_idx], display_name=display_name, 
                            size=400, color=node_color)  # Use appropriate color

                # Add edges to factors based on weights
                for factor_idx in range(n_factors):
                    weight = view_weights_matrix[factor_idx, feature_idx]
                    # Add edge if weight is substantial
                    factor_weights_for_view = view_weights_matrix[factor_idx, :]
                    weight_threshold = np.percentile(np.abs(factor_weights_for_view), 95)  # Connect top 5% weights

                    if abs(weight) > weight_threshold and abs(weight) > 0.05:  # Absolute min threshold too
                        factor_name = f"Factor {factor_idx+1}"
                        G.add_edge(feature_name, factor_name, weight=abs(weight), type='factor_feature')
                        max_abs_weight_overall = max(max_abs_weight_overall, abs(weight))

        # Optional: Add edges between features (based on co-association with factors)
        for factor_node in factor_nodes:
            connected_features = [n for n in G.neighbors(factor_node) if G.nodes[n]['node_type'] == 'feature']
            # Group features by view
            features_by_view = {}
            for feat in connected_features:
                view = G.nodes[feat]['view']
                if view not in features_by_view: features_by_view[view] = []
                features_by_view[view].append(feat)

            # Connect top features between pairs of views
            for view1, view2 in combinations(features_by_view.keys(), 2):
                for feat1 in features_by_view[view1]:
                    for feat2 in features_by_view[view2]:
                        # Check if edge doesn't already exist
                        if not G.has_edge(feat1, feat2):
                            # Add edge with weight based on combined strength
                            G.add_edge(feat1, feat2, weight=0.5, type='cross_view')  # Use a smaller weight

        # --- Network Layout ---
        # Use a layout that separates views and centers factors
        try:
            # Refined spring layout: position factors first, then let features arrange
            fixed_nodes = factor_nodes
            pos_factors = nx.circular_layout(G.subgraph(factor_nodes))  # Arrange factors in a circle
            pos = nx.spring_layout(G, pos=pos_factors, fixed=fixed_nodes, k=0.3, iterations=100, seed=42)
        except Exception as layout_err:
            print(f"Warning: Network layout failed ({layout_err}). Falling back to random layout.")
            pos = nx.random_layout(G)

        # --- Draw Network ---
        # Node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'factor':
                node_colors.append(COLORS['Table_Header_BG'])  # Light blue for factors
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = node  # Use full factor name
            elif G.nodes[node]['node_type'] == 'feature':
                node_colors.append(G.nodes[node].get('color', COLORS['UnknownFeature']))
                node_sizes.append(G.nodes[node]['size'])
                node_labels[node] = G.nodes[node]['display_name']  # Use shortened name
            else:  # Fallback
                node_colors.append('red')
                node_sizes.append(200)
                node_labels[node] = node[:5]+".."

        # Edge attributes
        factor_feature_edges = []
        cross_view_edges = []

        for u, v, d in G.edges(data=True):
            edge_type = d.get('type', 'unknown')
            if edge_type == 'factor_feature':
                factor_feature_edges.append((u, v))
            elif edge_type == 'cross_view':
                cross_view_edges.append((u, v))
            else:  # Fallback
                factor_feature_edges.append((u, v))  # Treat as factor-feature for drawing

        # Draw edges (draw cross-view first, so factor-feature are on top)
        if cross_view_edges:
            widths_cv = [G[u][v].get('weight', 0.5) * 2 for u, v in cross_view_edges]
            nx.draw_networkx_edges(G, pos, edgelist=cross_view_edges, width=widths_cv,
                                  edge_color=COLORS['Edge_High'], style='dashed', alpha=0.6, ax=ax_net)
                                  
        if factor_feature_edges:
            widths_ff = [G[u][v].get('weight', 0.1) * 5 for u, v in factor_feature_edges]
            nx.draw_networkx_edges(G, pos, edgelist=factor_feature_edges, width=widths_ff,
                                  edge_color=COLORS['Edge_High'], style='solid', alpha=0.8, ax=ax_net)

        # Draw nodes on top
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.9, edgecolors=COLORS['Node_Edge'], linewidths=0.5, ax=ax_net)
                              
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, 
                               font_size=FONTS_SANS['annotation'], ax=ax_net)

        # --- Enhance Network Plot Appearance ---
        ax_net.set_title('Cross-View Feature Integration Network (Top Features)', 
                        fontsize=FONTS_SANS['panel_title'], fontweight='bold')
        ax_net.axis('off')

        # Add legend manually
        legend_handles = []
        # Views
        for view in set(d['view'] for n, d in G.nodes(data=True) if 'view' in d):
            view_display = view.lower()
            # Replace metabolite with molecular feature for display
            if 'metabolite' in view_display:
                view_display = view_display.replace('metabolite', 'molecular feature')
            # Capitalize Leaf and Root
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
                color = 'blue'  # Default fallback
                
            # Format the display label
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
                                            
        # Factor
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Factor',
                                         markerfacecolor=COLORS['Table_Header_BG'], markersize=24))
                                         
        # Edges
        legend_handles.append(plt.Line2D([0], [0], color=COLORS['Edge_High'], lw=2, 
                                        label='Factor-Feature Link (Weight Scaled)'))
                                        
        if cross_view_edges:
            legend_handles.append(plt.Line2D([0], [0], color=COLORS['Edge_High'], lw=1.5, 
                                           linestyle='--', label='Inferred Cross-View Link'))

        ax_net.legend(handles=legend_handles, loc='lower right', 
                     fontsize=FONTS_SANS['legend_text'], frameon=True, 
                     facecolor=COLORS['Annotation_Box_BG'], framealpha=0.7)

        # --- Save Plot ---
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.png")
        svg_file = os.path.join(output_dir, f"cross_view_integration_{timestamp}.svg")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"SUCCESS: Saved cross-view integration plot to {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR creating cross-view integration visualization: {e}")
        traceback.print_exc()
        plt.close('all')
        return None


def main():
    """Main function to run the visualization"""
    # File paths for the MOFA+ model, metadata, and output directory
    mofa_file = "C:/Users/ms/Desktop/hyper/output/mofa/mofa_model_for_transformer.hdf5"
    metadata_file = "C:/Users/ms/Desktop/hyper/output/mofa/aligned_combined_metadata.csv"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"MOFA+ Cross-View Integration Visualization Script - START")
    print(f"MOFA+ model: {mofa_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load data
    data = load_data(mofa_file, metadata_file)
    if data is None:
        print("\nFATAL ERROR: Data loading failed. Exiting.")
        return 1

    # Generate the cross-view integration plot
    result = create_cross_view_integration(data, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Visualization Generation Summary:")
    if result:
        print(f"  ✅ Cross-View Integration Network")
        print("\nVisualization generated successfully.")
    else:
        print(f"  ❌ Cross-View Integration Network")
        print("\nVisualization generation failed. Please check the logs above for errors.")
    
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())