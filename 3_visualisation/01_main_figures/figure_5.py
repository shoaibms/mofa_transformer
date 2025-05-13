"""
Figure 5: Cross-Modal Network Visualization and Analysis

This script generates an integrated visualization of cross-modal attention networks
between spectral and molecular features in plant tissues under different experimental
conditions. It analyzes how drought-tolerant (G1) and drought-susceptible (G2) genotypes
deploy attention differently when processing multimodal data under stress conditions.

The figure contains:
- Network visualizations showing attention connections between spectral and molecular features
- Bar charts comparing network metrics between genotypes
- Comprehensive legends and annotations

Requirements:
- pandas, numpy, matplotlib, networkx, seaborn, scipy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx
import scipy.cluster.hierarchy as sch
import seaborn as sns

# Suppress warnings that might arise from empty dataframes or NaN values
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

#######################################################################################
# APPEARANCE SETTINGS - EDIT THESE TO CHANGE FIGURE STYLE
#######################################################################################

# Color scheme
COLORS = {
    # ==========================================================================
    # == Core Experimental Variables ==
    # ==========================================================================
    # Using distinct core families: Blues for Genotypes, Greens for Treatments
    'G1': '#00FA9A',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#48D1CC',             # Susceptible Genotype (Medium Teal)

    'T0': '#4682B4',            # Control Treatment (Medium Green)
    'T1': '#BDB76B',             # Stress Treatment (Muted Orange/Yellow)

    'Leaf': '#00FF00',            # Leaf Tissue (Darkest Green)
    'Root': '#40E0D0',            # Root Tissue (Darkest Blue)

    # --- Days (Subtle Yellow-Green sequence) ---
    'Day1': '#ffffcc',            # Very Light Yellow-Green
    'Day2': '#9CBA79',            # Light Yellow-Green
    'Day3': '#3e7d5a',            # Medium Yellow-Green

    # ==========================================================================
    # == Data Types / Omics / Features ==
    # ==========================================================================
    # Using distinct Blue/Green families for general types
    'Spectral': '#ECDA79',        # General Spectral (Medium Blue)
    'Molecular_feature': '#84ab92', # General Molecular feature (Medium-Dark Yellow-Green)
    'UnknownFeature': '#B0E0E6',  # Medium Grey for fallback

    # --- Specific Spectral Categories --- (Using blues, teals, greens, greys)
    'Spectral_Water': '#6DCAFA',     # Medium-Dark Blue
    'Spectral_Pigment': '#00FA9A',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#00BFFF',         # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',      # Medium Grey

    # --- Specific Metabolite Categories --- (Using Yellow/Greens)
    'P': '#3DB3BF',                 # P Molecular features (Darkest Yellow-Green)
    'N': '#ffffd4',                 # N Molecular features (Very Light Yellow)
    'Molecular_feature_Other': '#bdbdbd', # Other Molecular features (Light Grey)

    # ==========================================================================
    # == Methods & Model Comparison ==
    # ==========================================================================
    # Using distinct shades for clarity
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
    'Node_Molecular_feature': '#FFC4A1', # Default Molecular feature Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # ==========================================================================
    # == Statistical & Difference Indicators ==
    # ==========================================================================
    # Using Green for positive, muted Yellow for negative, Dark Blue for significance
    'Positive_Diff': '#66CDAA',     # Medium-Dark Green
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

# Text sizes
TEXT = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'main_title': 24,
    'panel_label': 21,
    'panel_title': 19,
    'axis_label': 19,
    'tick_label': 18,
    'legend_title': 21,
    'legend_text': 18,
    'annotation': 17,
    'caption': 17,
    'table_header': 17,
    'table_cell': 17,
    'tissue_label': 16
}

# Network visualization parameters
NETWORK = {
    'node_size_min': 100,
    'node_size_max': 800,
    'edge_width_min': 0.8,
    'edge_width_max': 6,
    'edge_alpha': 0.7,
    'node_alpha': 0.9,
    'node_edge_width': 0.5,
    'label_offset_x': 5,
    'label_offset_y': 0,
    'n_top_connections': 40,
    'min_attention': 0.01
}

# Bar chart parameters
BARS = {
    'bar_width': 0.35,
    'bar_alpha': 0.9,
    'value_offset': 3,
    'percent_offset': 1.1,
    'ylim_buffer': 1.15,
    'edge_color': 'black',
    'edge_width': 0.5
}

# Layout parameters
LAYOUT = {
    'figure_width': 20,
    'figure_height': 20,
    'top_margin': 0.90,
    'bottom_margin': 0.05,
    'left_margin': 0,
    'right_margin': 1,
    'colorbar_width': 0.4,
    'colorbar_height': 0.02,
    'colorbar_position_x': 0.3,
    'colorbar_position_y': 0.32
}

# Define a custom colormap for edge weights
edge_color_map = LinearSegmentedColormap.from_list('attention_cmap', [
    (0.0, COLORS['Edge_Low']),
    (0.2, '#d1e5f0'),
    (0.4, '#92c5de'),
    (0.6, '#4393c3'),
    (0.8, '#2166ac'),
    (1.0, COLORS['Edge_High'])
])

#######################################################################################
# END OF APPEARANCE SETTINGS
#######################################################################################

#######################################################################################
# DATA LOADING CONFIGURATION
#######################################################################################

# --- Configuration: Update these paths ---
output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
# Create output directory
os.makedirs(output_dir, exist_ok=True)

# ----- Network visualization data paths -----
# Files needed for network visualization
leaf_attn_cond_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_leaf\processed_mean_attention_conditional_Leaf.csv"
root_attn_cond_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_root\processed_mean_attention_conditional_Root.csv"
leaf_overall_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_leaf\processed_top_500_pairs_overall_Leaf.csv"
root_overall_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_root\processed_top_500_pairs_overall_Root.csv"

# ----- Network statistics data paths -----
# View level statistics for StdDev and P95
leaf_view_level_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_leaf\processed_view_level_attention_Leaf.csv"
root_view_level_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_root\processed_view_level_attention_Root.csv"

print(f"Data will be loaded from source files")
print(f"Output plots will be saved to: {output_dir}")

#######################################################################################
# UTILITY FUNCTIONS
#######################################################################################

def load_data_safe(filepath, default_name="Unknown"):
    """
    Safely load data with error handling.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    default_name : str
        Name to use for logging if the file doesn't have a name
        
    Returns:
    --------
    DataFrame
        Loaded data or empty DataFrame with expected columns if loading fails
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully loaded {default_name} data: {data.shape} rows, "
              f"{len(data.columns)} columns")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Genotype', 'Treatment', 'Time point', 
                                     'Spectral_Feature', 'Molecular_Feature', 
                                     'Mean_Attention_S2M_Group_AvgHeads'])

def clean_metadata_columns(df):
    """
    Standardizes metadata columns (Genotype, Treatment, Time point).
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with metadata columns
        
    Returns:
    --------
    DataFrame
        DataFrame with standardized metadata columns
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Genotype: Convert to string 'G1' or 'G2'
    if 'Genotype' in df.columns:
        df['Genotype'] = df['Genotype'].astype(str).replace(
            {'1': 'G1', '1.0': 'G1', '2': 'G2', '2.0': 'G2'})
        # Keep only G1, G2 rows
        df = df[df['Genotype'].isin(['G1', 'G2'])]

    # Treatment: Convert to string 'T0' or 'T1'
    if 'Treatment' in df.columns:
        df['Treatment'] = df['Treatment'].astype(str).replace(
            {'0': 'T0', '0.0': 'T0', '1': 'T1', '1.0': 'T1'})
        df = df[df['Treatment'].isin(['T0', 'T1'])]

    # Time point: Convert to numeric and then to int
    if 'Time point' in df.columns:
        df['Time point'] = pd.to_numeric(df['Time point'], errors='coerce')
        df = df.dropna(subset=['Time point'])  # Remove rows where conversion failed
        df['Time point'] = df['Time point'].astype(int)
    # Rename original 'Day' column if it exists
    elif 'Day' in df.columns:
        print("Renaming 'Day' column to 'Time point'")
        df = df.rename(columns={'Day': 'Time point'})
        df['Time point'] = pd.to_numeric(df['Time point'], errors='coerce')
        df = df.dropna(subset=['Time point'])  # Remove rows where conversion failed
        df['Time point'] = df['Time point'].astype(int)

    return df

def rename_legacy_columns(df):
    """
    Rename legacy column names and values for compatibility.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame that may contain legacy column names
        
    Returns:
    --------
    DataFrame
        DataFrame with standardized column names and values
    """
    if df is None:
        return None
    
    df = df.copy()  # Avoid SettingWithCopyWarning
    rename_dict = {}
    column_renamed = False
    
    # Rename column headers
    if 'Day' in df.columns and 'Time point' not in df.columns:
        rename_dict['Day'] = 'Time point'
        column_renamed = True
    if 'Metabolite_Feature' in df.columns and 'Molecular_Feature' not in df.columns:
        rename_dict['Metabolite_Feature'] = 'Molecular_Feature'
        column_renamed = True
    
    if rename_dict:
        print(f"Renaming legacy columns: {rename_dict}")
        df = df.rename(columns=rename_dict)

    # Rename values within the (potentially renamed) Molecular_Feature column
    feature_col = 'Molecular_Feature' if 'Molecular_Feature' in df.columns else 'Metabolite_Feature'

    if feature_col in df.columns:
        # Check if renaming is needed to avoid unnecessary operations
        needs_value_rename = df[feature_col].astype(str).str.contains('P_Cluster_', regex=False).any() or \
                             df[feature_col].astype(str).str.contains('N_Cluster_', regex=False).any()

        if needs_value_rename:
            print(f"Renaming values in column: {feature_col}")
            # Ensure column is string type for replacement
            df[feature_col] = df[feature_col].astype(str)
            df[feature_col] = df[feature_col].str.replace('P_Cluster_', 'P_', regex=False)
            df[feature_col] = df[feature_col].str.replace('N_Cluster_', 'N_', regex=False)

    return df

#######################################################################################
# FEATURE CATEGORIZATION FUNCTIONS
#######################################################################################

def categorize_spectral_feature(feature):
    """
    Categorize spectral features into meaningful biological groups.
    
    Parameters:
    -----------
    feature : str
        The spectral feature name to categorize
        
    Returns:
    --------
    tuple
        (category_name, color) for the feature
    """
    feature_str = str(feature).lower()
    
    # Water absorption bands
    if any(s in feature_str for s in ['1450', '1940', '970', 'water']):
        return "Water Band", COLORS['Spectral_Water']
    
    # Chlorophyll/pigment related
    elif any(s in feature_str for s in ['550', '660', '700', '720', 'chloro', 'pigment']):
        return "Pigment", COLORS['Spectral_Pigment']
    
    # NIR structural features
    elif any(s in feature_str for s in ['800', '900', '1100', '1200', 'nir', 'structure']):
        return "Structure", COLORS['Spectral_Structure']
    
    # SWIR regions
    elif any(s in feature_str for s in ['1600', '1700', '2000', '2100', '2200', '2300', 'swir']):
        return "SWIR", COLORS['Spectral_SWIR']
    
    # Default category - try to extract wavelength if present
    else:
        # Try to extract a wavelength number if it exists in the feature name
        import re
        wavelength_match = re.search(r'\d{3,4}', feature_str)
        if wavelength_match:
            wavelength = int(wavelength_match.group())
            if 400 <= wavelength <= 700:
                return "Visible", COLORS['Spectral_VIS']
            elif 700 <= wavelength <= 1300:
                return "NIR", COLORS['Spectral_Structure']
            elif 1300 <= wavelength <= 2500:
                return "SWIR", COLORS['Spectral_SWIR']
        
        return "Other", COLORS['Spectral_Other']


def categorize_molecular_feature(feature):
    """
    Categorize molecular features into meaningful groups.
    
    Parameters:
    -----------
    feature : str
        The molecular feature name to categorize
        
    Returns:
    --------
    tuple
        (category_name, color) for the feature
    """
    feature_str = str(feature).upper()
    
    # Positive mode features (often associated with sugars, amino acids)
    if feature_str.startswith('P_'):
        return "P", COLORS['P']

    # Negative mode features (often associated with organic acids, phenolics)
    elif feature_str.startswith('N_'):
        return "N", COLORS['N']

    # Default category
    else:
        return "Other Molecular", COLORS['Molecular_feature_Other']

def calculate_node_importance(data):
    """
    Calculate node importance metrics for sizing.
    
    Parameters:
    -----------
    data : dict
        Network data dictionary from prepare_network_data
        
    Returns:
    --------
    tuple
        (spectral_importance, molecular_feature_importance) dictionaries
        mapping node names to importance scores
    """
    # Initialize dictionaries
    spectral_importance = {}
    molecular_feature_importance = {}
    
    # Check if data is valid
    if data is None or 'G1' not in data or 'G2' not in data:
        return {}, {}
    
    # Combine G1 and G2 data to get overall importance
    combined_df = pd.concat([data['G1'], data['G2']])
    
    if combined_df.empty:
        return {}, {}
    
    # Calculate total incoming/outgoing attention for each node
    for _, row in combined_df.iterrows():
        spec = row['Spectral_Feature']
        molecular_feature = row['Molecular_Feature']
        attention = row['Mean_Attention_S2M_Group_AvgHeads']
        
        # Update spectral feature importance
        if spec in spectral_importance:
            spectral_importance[spec] += attention
        else:
            spectral_importance[spec] = attention
            
        # Update molecular feature importance
        if molecular_feature in molecular_feature_importance:
            molecular_feature_importance[molecular_feature] += attention
        else:
            molecular_feature_importance[molecular_feature] = attention
    
    # Calculate network centrality for each feature
    # Create temporary graph for centrality calculations
    G = nx.Graph()
    for _, row in combined_df.iterrows():
        G.add_edge(row['Spectral_Feature'], row['Molecular_Feature'],
                  weight=row['Mean_Attention_S2M_Group_AvgHeads'])
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Combine attention sum and centrality (with equal weights)
    for node, centrality in degree_centrality.items():
        # Apply to spectral features
        if node in spectral_importance:
            # Normalize by dividing by max attention
            norm_attention = (spectral_importance[node] / 
                             max(spectral_importance.values()) if spectral_importance else 0)
            # Combine metrics
            spectral_importance[node] = 0.5 * norm_attention + 0.5 * centrality
        
        # Apply to molecular features
        if node in molecular_feature_importance:
            norm_attention = (molecular_feature_importance[node] / 
                             max(molecular_feature_importance.values()) if molecular_feature_importance else 0)
            molecular_feature_importance[node] = 0.5 * norm_attention + 0.5 * centrality
    
    return spectral_importance, molecular_feature_importance

# Create network visualization with consistent layout and enhanced features
def create_network_visualization(fig, ax, data, geno, tissue, node_importances, pos=None):
    """
    Create an enhanced network visualization for a specific condition.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to draw on
    ax : matplotlib.axes.Axes
        The axes to draw on
    data : dict
        The network data
    geno : str
        Genotype ('G1' or 'G2')
    tissue : str
        Tissue type ('Leaf' or 'Root')
    node_importances : tuple
        Tuple of (spectral_importance, molecular_feature_importance) dictionaries
    pos : dict, optional
        Node positions for consistent layout
        
    Returns:
    --------
    tuple
        (Graph, edge_weights, node_positions)
    """
    if data is None:
        ax.text(0.5, 0.5, f"No data available for {tissue} - {geno}", 
               ha='center', va='center', fontsize=TEXT['panel_title'])
        return None, [], None
    
    # Unpack node importances
    spec_importance, molecular_feature_importance = node_importances
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for spec in data['all_spectral']:
        category, color = categorize_spectral_feature(spec)
        # Get importance or set default
        importance = spec_importance.get(spec, 0.001)
        G.add_node(spec, bipartite=0, type='spectral', category=category, 
                  color=color, importance=importance)
    
    for molecular_feature in data['all_molecular_feature']:
        category, color = categorize_molecular_feature(molecular_feature)
        # Get importance or set default
        importance = molecular_feature_importance.get(molecular_feature, 0.001)
        G.add_node(molecular_feature, bipartite=1, type='molecular_feature', 
                  category=category, color=color, importance=importance)
    
    # Add edges from this genotype's data
    edge_weights = []
    edge_pairs = []
    df = data[geno]
    
    # Check if we have data
    if df.empty:
        ax.text(0.5, 0.5, f"No connections for {tissue} - {geno}", 
               ha='center', va='center', fontsize=TEXT['panel_title'])
        return G, [], pos
    
    for _, row in df.iterrows():
        spec = row['Spectral_Feature']
        molecular_feature = row['Molecular_Feature']
        attention = row['Mean_Attention_S2M_Group_AvgHeads']
        
        # Only add edge if both nodes exist
        if spec in G.nodes and molecular_feature in G.nodes:
            G.add_edge(spec, molecular_feature, weight=attention)
            edge_weights.append(attention)
            edge_pairs.append((spec, molecular_feature))
    
    # Create bipartite layout if not provided
    if pos is None:
        # Get spectral and molecular feature nodes
        spec_nodes = {n for n, d in G.nodes(data=True) if d['type'] == 'spectral'}
        molecular_feature_nodes = {n for n, d in G.nodes(data=True) 
                                 if d['type'] == 'molecular_feature'}
        
        # Cluster nodes for better organization
        # First, create distance matrices based on connection patterns
        spec_matrix = np.zeros((len(spec_nodes), len(spec_nodes)))
        spec_list = list(spec_nodes)
        
        molecular_feature_matrix = np.zeros((len(molecular_feature_nodes), 
                                          len(molecular_feature_nodes)))
        molecular_feature_list = list(molecular_feature_nodes)
        
        # Fill matrices based on shared connections
        for i, spec1 in enumerate(spec_list):
            for j, spec2 in enumerate(spec_list):
                if i == j:
                    continue
                # Count number of shared molecular feature connections
                spec1_connections = set(G.neighbors(spec1))
                spec2_connections = set(G.neighbors(spec2))
                shared = len(spec1_connections.intersection(spec2_connections))
                # More shared connections = smaller distance
                spec_matrix[i, j] = 1.0 / (shared + 1)
        
        for i, molecular_feature1 in enumerate(molecular_feature_list):
            for j, molecular_feature2 in enumerate(molecular_feature_list):
                if i == j:
                    continue
                # Count number of shared spectral connections
                molecular_feature1_connections = set(G.neighbors(molecular_feature1))
                molecular_feature2_connections = set(G.neighbors(molecular_feature2))
                shared = len(molecular_feature1_connections.intersection(
                    molecular_feature2_connections))
                # More shared connections = smaller distance
                molecular_feature_matrix[i, j] = 1.0 / (shared + 1)
        
        # Perform hierarchical clustering
        try:
            spec_linkage = sch.linkage(spec_matrix, method='ward')
            spec_dendro = sch.dendrogram(spec_linkage, no_plot=True)
            spec_order = spec_dendro['leaves']
        except Exception as e:
            print(f"Error in spectral clustering, using default order: {e}")
            spec_order = list(range(len(spec_list)))
        
        try:
            molecular_feature_linkage = sch.linkage(molecular_feature_matrix, method='ward')
            molecular_feature_dendro = sch.dendrogram(molecular_feature_linkage, no_plot=True)
            molecular_feature_order = molecular_feature_dendro['leaves']
        except Exception as e:
            print(f"Error in molecular feature clustering, using default order: {e}")
            molecular_feature_order = list(range(len(molecular_feature_list)))
        
        # Create positions with clustered ordering
        pos = {}
        
        # Position spectral features on left side based on clustering
        for i, idx in enumerate(spec_order):
            if idx < len(spec_list):  # Safety check
                node = spec_list[idx]
                # Position based on clustering with slight vertical jitter
                jitter = np.random.uniform(-0.02, 0.02)
                pos[node] = (-0.7, 0.8 - (i / max(1, len(spec_order)) * 1.6) + jitter)
        
        # Position molecular features on right side based on clustering
        for i, idx in enumerate(molecular_feature_order):
            if idx < len(molecular_feature_list):  # Safety check
                node = molecular_feature_list[idx]
                # Position based on clustering with slight vertical jitter
                jitter = np.random.uniform(-0.02, 0.02)
                pos[node] = (0.7, 0.8 - (i / max(1, len(molecular_feature_order)) * 1.6) + jitter)
    
    # Calculate node sizes based on importance
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        # Get node importance
        if G.nodes[node]['type'] == 'spectral':
            importance = G.nodes[node]['importance']
            size = importance * 5000
        elif G.nodes[node]['type'] == 'molecular_feature':
            importance = G.nodes[node]['importance']
            size = importance * 5000
        else:  # Fallback for safety
            size = NETWORK['node_size_min']

        # Cap size between min and max
        node_sizes.append(max(NETWORK['node_size_min'], 
                             min(NETWORK['node_size_max'], size)))
        
        # Set node color based on category
        node_colors.append(G.nodes[node]['color'])
    
    # Normalize edge weights for coloring
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        weight_range = max_weight - min_weight
        
        if weight_range > 0:
            norm_weights = [(w - min_weight) / weight_range for w in edge_weights]
        else:
            norm_weights = [0.5] * len(edge_weights)
            
        edge_colors = [edge_color_map(nw) for nw in norm_weights]
        
        # Edge widths based on weights, scaled for visibility
        edge_widths = [max(NETWORK['edge_width_min'], 
                          min(NETWORK['edge_width_max'], w * 25)) for w in edge_weights]
    else:
        edge_colors = ['#d3d3d3']
        edge_widths = [0.5]
    
    # Draw the network using LineCollection for better performance with many edges
    lines = []
    colors = []
    linewidths = []
    
    for i, (u, v) in enumerate(G.edges()):
        if u in pos and v in pos:  # Safety check
            lines.append([(pos[u][0], pos[u][1]), (pos[v][0], pos[v][1])])
            colors.append(edge_colors[i] if i < len(edge_colors) else edge_colors[0])
            linewidths.append(edge_widths[i] if i < len(edge_widths) else edge_widths[0])
    
    # Create the line collection for edges
    if lines:
        lc = LineCollection(lines, colors=colors, linewidths=linewidths, 
                           alpha=NETWORK['edge_alpha'], zorder=1)
        ax.add_collection(lc)
    
    # Draw nodes
    for i, node in enumerate(G.nodes()):
        if node in pos:  # Safety check
            ax.scatter(pos[node][0], pos[node][1], s=node_sizes[i], 
                      c=node_colors[i], edgecolor='black', 
                      linewidth=NETWORK['node_edge_width'], 
                      alpha=NETWORK['node_alpha'], zorder=2)
    
    # Add node labels for top nodes (top 3 spectral and top 3 molecular feature by importance)
    spec_nodes = {n: G.nodes[n]['importance'] for n, d in G.nodes(data=True) 
                 if d['type'] == 'spectral'}
    molecular_feature_nodes = {n: G.nodes[n]['importance'] for n, d in G.nodes(data=True) 
                             if d['type'] == 'molecular_feature'}
    
    # Sort by importance
    top_spec = sorted(spec_nodes.items(), key=lambda x: x[1], reverse=True)[:3]
    top_molecular_feature = sorted(molecular_feature_nodes.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]

    for node, _ in top_spec + top_molecular_feature:
        if node not in pos:  # Safety check
            continue
            
        label = str(node)
        # Shorten long labels
        if len(label) > 12:
            if 'Cluster' in label or label.startswith('P_') or label.startswith('N_'):
                # For cluster IDs, keep format but truncate number
                parts = label.split('_')
                if len(parts) > 1:
                    prefix = parts[0]
                    number = parts[1]
                    label = f"{prefix}_{number[:4]}..."
                else:  # Handle cases like just 'P' or 'N' if they occur as nodes
                    label = label[:10] + '...' if len(label) > 10 else label
            else:
                label = label[:10] + '...'
                
        # Position label based on node type (left or right side)
        if G.nodes[node]['type'] == 'spectral':
            offset = (-NETWORK['label_offset_x'], NETWORK['label_offset_y'])  # Text to the left
            ha = 'right'
        elif G.nodes[node]['type'] == 'molecular_feature': # Check new type
            offset = (NETWORK['label_offset_x'], NETWORK['label_offset_y'])  # Text to the right
            ha = 'left'
        else: # Fallback
            offset = (0, 0)
            ha = 'center'
            
        ax.annotate(label, xy=pos[node], xytext=offset, 
                   textcoords="offset points", fontsize=TEXT['annotation'], ha=ha,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    
    # Set title and remove axis ticks
    genotype_label = "Drought-Tolerant (G1)" if geno == 'G1' else "Drought-Susceptible (G2)"
    ax.set_title(f"{tissue} - {genotype_label}", fontsize=TEXT['panel_title'], fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add x-axis labels for the network structure
    ax.text(-0.7, -0.95, "Spectral Features", ha='center', fontsize=TEXT['axis_label'])
    ax.text(0.7, -0.95, "Molecular Features", ha='center', fontsize=TEXT['axis_label']) # Renamed label
    ax.text(0, -0.95, "→", ha='center', fontsize=TEXT['panel_title'])
    
    # Add network statistics from the data
    if 'G1_stats' in data and 'G2_stats' in data:
        stats = data[f'{geno}_stats']
        
        stats_text = (
            f"Connections: {stats['connections']}\n"
            f"Mean Attention: {stats['mean']:.4f}\n"
            f"Max Connection: {stats['max']:.4f}"
        )
        
        ax.text(0, -0.82, stats_text, ha='center', fontsize=TEXT['annotation'], color='#7f7f7f',
               bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="gray", alpha=0.8))
    
    # Add G1 vs G2 comparison text for G1 panels
    if geno == 'G1' and 'common_pairs' in data and not data['common_pairs'].empty:
        # Get fold-change statistics for common pairs
        common_df = data['common_pairs']
        avg_fold = common_df['Fold_Change'].mean()
        
        if avg_fold > 1:
            highlight_color = "#e6f7ff"  # Light blue
            text_color = "#0066cc"  # Blue
            sign = "+"
        else:
            highlight_color = "#fff7e6"  # Light orange
            text_color = "#cc6600"  # Orange
            sign = "-"
            avg_fold = 1/avg_fold if avg_fold > 0 else 0
        
        comparison_text = (
            f"G1 vs G2 Attention:\n"
            f"{sign}{avg_fold:.2f}x on average\n"
            f"for shared connections"
        )
        
        ax.text(-0.7, 0.85, comparison_text, ha='center', fontsize=TEXT['annotation'], color=text_color,
               bbox=dict(boxstyle="round,pad=0.2", fc=highlight_color, ec=text_color, alpha=0.9))
    
    return G, edge_weights, pos

# ===== NETWORK STATISTICS FUNCTIONS =====

def calculate_network_stats(leaf_overall_df, root_overall_df, leaf_cond_df, root_cond_df, 
                           leaf_view_df, root_view_df):
    """Calculate network statistics for both tissues and genotypes"""

    calculated_metrics = []

    for tissue, overall_df, cond_df, view_df in [
        ('Leaf', leaf_overall_df, leaf_cond_df, leaf_view_df),
        ('Root', root_overall_df, root_cond_df, root_view_df)
    ]:
        print(f"\nProcessing {tissue} tissue for metrics...")

        # 1. Get Top 100 Pairs
        if overall_df is not None and not overall_df.empty and 'Spectral_Feature' in overall_df.columns:
            top_100_pairs = set(zip(overall_df['Spectral_Feature'].head(100),
                                    overall_df['Molecular_Feature'].head(100))) # Renamed column
        else:
            print(f"Warning: Could not identify Top 100 pairs for {tissue}.")
            top_100_pairs = set()

        # 2. Filter conditional and view-level data for T1, Time point 3
        if cond_df is not None and not cond_df.empty:
            cond_t1d3 = cond_df[(cond_df['Time point'] == 3) & (cond_df['Treatment'] == 'T1')] # Renamed column
        else:
            cond_t1d3 = pd.DataFrame()
            
        if view_df is not None and not view_df.empty:
            view_t1d3 = view_df[(view_df['Time point'] == 3) & (view_df['Treatment'] == 'T1')] # Renamed column
        else:
            view_t1d3 = pd.DataFrame()

        # 3. Calculate metrics for G1 and G2
        for genotype in ['G1', 'G2']:
            # Filter data for current genotype
            geno_cond_t1d3 = cond_t1d3[cond_t1d3['Genotype'] == genotype] if not cond_t1d3.empty else pd.DataFrame()
            geno_view_t1d3 = view_t1d3[view_t1d3['Genotype'] == genotype] if not view_t1d3.empty else pd.DataFrame()

            # Metric 1: Average Attention (Top 100 Pairs)
            avg_attention = np.nan # Default to NaN
            if not geno_cond_t1d3.empty and top_100_pairs:
                if 'Spectral_Feature' in geno_cond_t1d3.columns and \
                   'Molecular_Feature' in geno_cond_t1d3.columns and \
                   'Mean_Attention_S2M_Group_AvgHeads' in geno_cond_t1d3.columns:

                    # Filter conditional data to only include top 100 pairs
                    geno_top_pairs_cond = geno_cond_t1d3[
                        geno_cond_t1d3.apply(
                            lambda row: (row['Spectral_Feature'], row['Molecular_Feature']) in top_100_pairs, # Renamed column
                            axis=1
                        )
                    ]
                    if not geno_top_pairs_cond.empty:
                        avg_attention = geno_top_pairs_cond['Mean_Attention_S2M_Group_AvgHeads'].mean()
                    else:
                        print(f"Warning: No data for {tissue} {genotype} T1D3 within Top 100 pairs.")

            # Metric 2: StdDev Attention
            std_dev_attention = np.nan
            if not geno_view_t1d3.empty:
                if 'StdAttn_S2M' in geno_view_t1d3.columns:
                    std_dev_attention = geno_view_t1d3['StdAttn_S2M'].mean()
                else:
                    print(f"Warning: 'StdAttn_S2M' column missing in {tissue} view-level data.")

            # Metric 3: P95 Attention
            p95_attention = np.nan
            if not geno_view_t1d3.empty:
                if 'P95Attn_S2M' in geno_view_t1d3.columns:
                    p95_attention = geno_view_t1d3['P95Attn_S2M'].mean()
                else:
                    print(f"Warning: 'P95Attn_S2M' column missing in {tissue} view-level data.")

            print(f"  {genotype}: AvgAttn(Top100)={avg_attention:.4f}, StdDev={std_dev_attention:.4f}, P95={p95_attention:.4f}")

            # Store results
            calculated_metrics.append({
                'Tissue': tissue,
                'Genotype': genotype,
                'Avg_Attention_Top100': avg_attention,
                'StdDev_Attention': std_dev_attention,
                'P95_Attention': p95_attention
            })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(calculated_metrics)
    return metrics_df

def create_network_stats_visualization(fig, network_metrics_df, ax_row_idx):
    """
    Create the network statistics visualization in the specified row
    
    Parameters:
    -----------
    fig : Figure
        The matplotlib figure
    network_metrics_df : DataFrame
        The dataframe containing network metrics
    ax_row_idx : int
        The row index for the axes where the stats visualizations should be placed
    """
    if network_metrics_df is None or network_metrics_df.empty:
        print("No metrics were calculated. Cannot generate network stats visualization.")
        return
    
    # --- Calculate Percentage Differences and Prepare Plotting DF ---
    plot_data = []
    metrics_to_plot = ['Avg_Attention_Top100', 'StdDev_Attention', 'P95_Attention']

    for tissue in ['Leaf', 'Root']:
        tissue_data = network_metrics_df[network_metrics_df['Tissue'] == tissue]
        if len(tissue_data) == 2:  # Ensure we have both G1 and G2 data
            g1_data = tissue_data[tissue_data['Genotype'] == 'G1'].iloc[0]
            g2_data = tissue_data[tissue_data['Genotype'] == 'G2'].iloc[0]

            for metric in metrics_to_plot:
                g1_val = g1_data[metric]
                g2_val = g2_data[metric]

                pct_diff = np.nan
                # Calculate percent difference only if values are valid and G2 is not zero
                if pd.notna(g1_val) and pd.notna(g2_val) and g2_val != 0:
                    pct_diff = ((g1_val - g2_val) / abs(g2_val)) * 100  # Use abs(g2_val) to avoid issues if g2_val is small negative

                # Add data for plotting
                for geno, val in [('G1', g1_val), ('G2', g2_val)]:
                    # Only add if value is not NaN
                    if pd.notna(val):
                        plot_data.append({
                            'Tissue': tissue,
                            'Genotype': geno,
                            'Metric': metric,
                            'Value': val,
                            'Pct_Diff': pct_diff  # Store the single calculated pct_diff for both G1/G2 rows of this metric/tissue
                        })
        else:
            print(f"Warning: Did not find data for both G1 and G2 in {tissue}. Skipping percentage calculation.")

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("Plotting DataFrame is empty. Cannot generate plot.")
        return

    # --- Create Plot ---
    metric_display_names = {
        'Avg_Attention_Top100': 'Average Attention\n(Top 100 Pairs)',
        'StdDev_Attention': 'Attention StdDev\n(Feature Focus)',
        'P95_Attention': 'P95 Attention\n(Peak Strength)'
    }
    plot_metrics_order = [m for m in metrics_to_plot if m in plot_df['Metric'].unique()]  # Only plot metrics we have data for

    # Create a single subplot that spans the bottom row
    ax = fig.add_subplot(3, 1, 3)  # One row that spans the entire width
    axes = [ax]
    
    # Set up the color palette
    palette = {'G1': '#00FA9A', 'G2': '#48D1CC'}  # Blue for G1, Teal for G2
    
    # Create a grouped barplot with all metrics side by side
    # First, prepare data for a grouped bar plot
    metric_data_list = []
    for i, metric in enumerate(plot_metrics_order):
        metric_data = plot_df[plot_df['Metric'] == metric].copy()
        metric_data['Metric_Name'] = metric_display_names.get(metric, metric)
        metric_data['Position'] = i  # To help with positioning
        metric_data_list.append(metric_data)
    
    if metric_data_list:
        combined_data = pd.concat(metric_data_list)
        
        # Create custom positions for bars
        positions = []
        current_pos = 0
        for tissue in ['Leaf', 'Root']:
            for metric_pos in range(len(plot_metrics_order)):
                positions.append(current_pos)
                current_pos += 1
            current_pos += 1  # Add space between tissue groups
        
        # Create barplot manually for better control
        bar_width = 0.35  # Width of each bar
        x_positions = []
        bar_heights = []
        bar_colors = []
        bar_labels = []
        
        # Process each tissue, metric, genotype combination
        for tissue_idx, tissue in enumerate(['Leaf', 'Root']):
            base_position = tissue_idx * (len(plot_metrics_order) + 1)  # +1 for spacing
            
            for metric_idx, metric in enumerate(plot_metrics_order):
                metric_name = metric_display_names.get(metric, metric)
                position = base_position + metric_idx
                
                for geno_idx, genotype in enumerate(['G1', 'G2']):
                    # Get the bar data
                    subset = combined_data[(combined_data['Tissue'] == tissue) & 
                                         (combined_data['Metric'] == metric) &
                                         (combined_data['Genotype'] == genotype)]
                    
                    if not subset.empty:
                        value = subset['Value'].iloc[0]
                        
                        # Calculate bar position
                        bar_pos = position + (geno_idx - 0.5) * bar_width
                        
                        # Store for plotting
                        x_positions.append(bar_pos)
                        bar_heights.append(value)
                        bar_colors.append(palette[genotype])
                        bar_labels.append(f"{tissue}-{metric_name}-{genotype}")
        
        # Create the bars
        bars = ax.bar(x_positions, bar_heights, bar_width * 0.9, color=bar_colors)
        
        # Set x-ticks and labels
        group_positions = []
        group_labels = []
        
        for tissue_idx, tissue in enumerate(['Leaf', 'Root']):
            for metric_idx, metric in enumerate(plot_metrics_order):
                position = tissue_idx * (len(plot_metrics_order) + 1) + metric_idx
                group_positions.append(position)
                group_labels.append(metric_display_names.get(metric, metric).split('\n')[0])
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels(group_labels, rotation=0, fontsize=TEXT['tick_label'])
        
        # Increase y-tick font size
        ax.tick_params(axis='y', labelsize=TEXT['tick_label'])
        
        # Add tissue labels - ensure they don't overlap with x-axis labels
        for tissue_idx, tissue in enumerate(['Leaf', 'Root']):
            pos = tissue_idx * (len(plot_metrics_order) + 1) + len(plot_metrics_order)/2 - 0.5
            ax.text(pos, -0.12, tissue, 
                   ha='center', va='center', fontsize=TEXT['axis_label'], fontweight='bold',
                   transform=ax.get_xaxis_transform())
                   
        # Create legend with consistent font sizes with network plots
        legend_elements = [
            mpatches.Patch(facecolor=palette['G1'], edgecolor='black', label='G1 (Tolerant)'),
            mpatches.Patch(facecolor=palette['G2'], edgecolor='black', label='G2 (Susceptible)')
        ]
        ax.legend(handles=legend_elements, title="Genotype", loc='upper right', 
                 fontsize=14, title_fontsize=16)
        
        # Add x-axis title with sufficient spacing from the tissue labels
        ax.text(0.5, -0.2, "Network Metrics", ha='center', va='center', 
                fontsize=TEXT['axis_label'], transform=ax.transAxes)
        
        # Add more padding at the bottom to accommodate all labels
        fig.subplots_adjust(bottom=0.2)
        
        # Add percentage difference annotations with improved positioning
        for tissue_idx, tissue in enumerate(['Leaf', 'Root']):
            for metric_idx, metric in enumerate(plot_metrics_order):
                position = tissue_idx * (len(plot_metrics_order) + 1) + metric_idx
                
                # Get G1 and G2 values
                g1_data = combined_data[(combined_data['Tissue'] == tissue) & 
                                      (combined_data['Metric'] == metric) &
                                      (combined_data['Genotype'] == 'G1')]
                
                g2_data = combined_data[(combined_data['Tissue'] == tissue) & 
                                      (combined_data['Metric'] == metric) &
                                      (combined_data['Genotype'] == 'G2')]
                
                if not g1_data.empty and not g2_data.empty:
                    g1_val = g1_data['Value'].iloc[0]
                    g2_val = g2_data['Value'].iloc[0]
                    pct_diff = g1_data['Pct_Diff'].iloc[0]
                    
                    if pd.notna(g1_val) and pd.notna(g2_val) and pd.notna(pct_diff):
                        # Position closer to the bars (adaptive based on height)
                        max_height = max(g1_val, g2_val)
                        # Instead of fixed multiplier, use adaptive positioning
                        if max_height > 0.04:
                            y_pos = max_height * 1.12  # Less space for tall bars
                        else:
                            y_pos = max_height * 1.18  # More space for short bars
                        
                        diff_text = f"+{pct_diff:.1f}%" if pct_diff >= 0 else f"{pct_diff:.1f}%"
                        diff_color = 'green' if pct_diff >= 0 else 'red'
                        
                        ax.text(position, y_pos, diff_text,
                               ha='center', va='bottom', fontweight='bold',
                               color=diff_color, fontsize=TEXT['annotation'])
        
        # Add value labels on bars - now with consistent positioning inside bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if pd.notna(height):
                # Place all values inside bars for consistency
                ax.annotate(f"{height:.2f}",
                           (bar.get_x() + bar.get_width() / 2., height / 2),
                           ha='center', va='center', fontsize=TEXT['annotation'], 
                           color='black', weight='bold')
        
        # Set title with E) panel label
        ax.set_title("E) Cross-Modal Network Properties: G1 (Tolerant) vs G2 (Susceptible) under Stress (T1, Time point 3)",
                    fontsize=TEXT['panel_title'], pad=20, fontweight='bold')
        ax.set_ylabel('Value', fontsize=TEXT['axis_label'])
        ax.set_xlabel('')
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=palette['G1'], edgecolor='black', label='G1 (Tolerant)'),
            mpatches.Patch(facecolor=palette['G2'], edgecolor='black', label='G2 (Susceptible)')
        ]
        ax.legend(handles=legend_elements, title="Genotype", loc='upper right', 
                 fontsize=TEXT['legend_text'], title_fontsize=TEXT['legend_title'])
        
        # Clean up the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Ensure y starts at 0
        ax.set_ylim(bottom=0)
        
        # Add more buffer at the top for percentage annotations
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)

    else:
        ax.text(0.5, 0.5, "No data available for network statistics", 
               ha='center', va='center', fontsize=14)

    return axes

#######################################################################################
# NETWORK VISUALIZATION FUNCTIONS
#######################################################################################

def prepare_network_data(df, tissue, n_top_connections=40, min_attention=0.01):
    """
    Filter and prepare network data for visualization.
    
    Parameters:
    -----------
    df : DataFrame
        The attention data
    tissue : str
        Tissue name for logging
    n_top_connections : int
        Number of top connections to include
    min_attention : float
        Minimum attention score to include
        
    Returns:
    --------
    dict
        Dictionary containing processed data for both genotypes
    """
    # Ensure we have the expected columns and standardize the format
    print(f"Processing {tissue} data...")
    
    if df is None or df.empty:
        print(f"No data available for {tissue}")
        return None
    
    # Check for and standardize column names if needed
    attention_col = None
    for col in df.columns:
        if 'Attention' in col and 'S2M' in col:
            attention_col = col
            break
    
    if attention_col is None:
        print(f"ERROR: Could not find attention column in {tissue} data")
        return None
    
    if attention_col != 'Mean_Attention_S2M_Group_AvgHeads':
        print(f"Renaming column {attention_col} to 'Mean_Attention_S2M_Group_AvgHeads'")
        df = df.rename(columns={attention_col: 'Mean_Attention_S2M_Group_AvgHeads'})
    
    # Filter for stress condition (T1) - accommodate different formats
    stress_mask = (df['Treatment'] == 'T1') | (df['Treatment'] == '1') | (df['Treatment'] == 1)
    stress_df = df[stress_mask].copy()
    print(f"Found {len(stress_df)} stress condition (T1) rows")
    
    # For time point 3 (peak stress) - accommodate different formats
    # Convert Time point to numeric if it's not already
    if df['Time point'].dtype == 'object':
        df['Time point'] = pd.to_numeric(df['Time point'], errors='coerce')
    
    day3_mask = df['Time point'] == 3
    day3_df = stress_df[day3_mask].copy()
    print(f"Found {len(day3_df)} Time point 3 rows under stress")
    
    # If time point 3 has too few data points, use the latest available time point
    if len(day3_df) < 10:
        max_day = stress_df['Time point'].max()
        print(f"WARNING: Few data points for Time point 3. Using Time point {max_day} instead")
        day_mask = stress_df['Time point'] == max_day
        day3_df = stress_df[day_mask].copy()
        if len(day3_df) < 10:
            print(f"WARNING: Still few data points for peak time point. "
                  f"Using all time points under stress.")
            day3_df = stress_df.copy()
    
    # Create separate dataframes for each genotype
    g1_mask = (day3_df['Genotype'] == 'G1') | (day3_df['Genotype'] == '1') | (day3_df['Genotype'] == 1)
    g2_mask = (day3_df['Genotype'] == 'G2') | (day3_df['Genotype'] == '2') | (day3_df['Genotype'] == 2)
    
    g1_df = day3_df[g1_mask].copy()
    g2_df = day3_df[g2_mask].copy()
    
    print(f"G1 samples: {len(g1_df)}, G2 samples: {len(g2_df)}")
    
    # Filter by minimum attention score if specified
    if min_attention > 0:
        g1_df = g1_df[g1_df['Mean_Attention_S2M_Group_AvgHeads'] >= min_attention]
        g2_df = g2_df[g2_df['Mean_Attention_S2M_Group_AvgHeads'] >= min_attention]
        print(f"After min_attention filter: G1: {len(g1_df)}, G2: {len(g2_df)}")
    
    # Get top connections by attention score
    g1_top = g1_df.nlargest(n_top_connections, 'Mean_Attention_S2M_Group_AvgHeads')
    g2_top = g2_df.nlargest(n_top_connections, 'Mean_Attention_S2M_Group_AvgHeads')
    
    # Get all unique features from both genotypes for consistent visualization
    all_spectral = set(g1_top['Spectral_Feature'].unique()) | set(g2_top['Spectral_Feature'].unique())
    all_molecular_feature = set(g1_top['Molecular_Feature'].unique()) | set(g2_top['Molecular_Feature'].unique())
    
    # Find common connections between genotypes
    g1_pairs = set(zip(g1_top['Spectral_Feature'], g1_top['Molecular_Feature']))
    g2_pairs = set(zip(g2_top['Spectral_Feature'], g2_top['Molecular_Feature']))
    common_pairs = g1_pairs.intersection(g2_pairs)
    
    print(f"Unique spectral features: {len(all_spectral)}")
    print(f"Unique molecular features: {len(all_molecular_feature)}")
    print(f"Common connections between G1 and G2: {len(common_pairs)}")
    
    # Calculate summary statistics
    g1_stats = {
        'mean': g1_top['Mean_Attention_S2M_Group_AvgHeads'].mean() if not g1_top.empty else 0,
        'median': g1_top['Mean_Attention_S2M_Group_AvgHeads'].median() if not g1_top.empty else 0,
        'max': g1_top['Mean_Attention_S2M_Group_AvgHeads'].max() if not g1_top.empty else 0,
        'connections': len(g1_top)
    }
    
    g2_stats = {
        'mean': g2_top['Mean_Attention_S2M_Group_AvgHeads'].mean() if not g2_top.empty else 0,
        'median': g2_top['Mean_Attention_S2M_Group_AvgHeads'].median() if not g2_top.empty else 0,
        'max': g2_top['Mean_Attention_S2M_Group_AvgHeads'].max() if not g2_top.empty else 0,
        'connections': len(g2_top)
    }
    
    # Prepare data for common pairs comparison
    common_pair_comparison = []
    for spec, molecular_feature in common_pairs:
        g1_row = g1_top[(g1_top['Spectral_Feature'] == spec) & 
                         (g1_top['Molecular_Feature'] == molecular_feature)]
        g2_row = g2_top[(g2_top['Spectral_Feature'] == spec) & 
                         (g2_top['Molecular_Feature'] == molecular_feature)]
        
        if not g1_row.empty and not g2_row.empty:
            g1_score = g1_row['Mean_Attention_S2M_Group_AvgHeads'].values[0]
            g2_score = g2_row['Mean_Attention_S2M_Group_AvgHeads'].values[0]
            fold_change = g1_score / g2_score if g2_score > 0 else float('inf')
            
            common_pair_comparison.append({
                'Spectral_Feature': spec,
                'Molecular_Feature': molecular_feature,
                'G1_Score': g1_score,
                'G2_Score': g2_score,
                'Fold_Change': fold_change
            })
    
    return {
        'G1': g1_top,
        'G2': g2_top,
        'all_spectral': all_spectral,
        'all_molecular_feature': all_molecular_feature,
        'G1_full': g1_df,
        'G2_full': g2_df,
        'G1_stats': g1_stats,
        'G2_stats': g2_stats,
        'common_pairs': pd.DataFrame(common_pair_comparison) if common_pair_comparison else pd.DataFrame()
    }

#######################################################################################
# MAIN PROGRAM
#######################################################################################

# Load all required data
print("\nLoading data for all visualizations...")

# Load data for network visualization
leaf_attn_df = load_data_safe(leaf_attn_cond_path, "leaf conditional attention")
root_attn_df = load_data_safe(root_attn_cond_path, "root conditional attention")
leaf_overall_df = load_data_safe(leaf_overall_path, "leaf overall attention")
root_overall_df = load_data_safe(root_overall_path, "root overall attention")

# Load data for network statistics
leaf_view_df = load_data_safe(leaf_view_level_path, "leaf view-level stats")
root_view_df = load_data_safe(root_view_level_path, "root view-level stats")

# Standardize format
leaf_attn_df = clean_metadata_columns(leaf_attn_df)
root_attn_df = clean_metadata_columns(root_attn_df)
leaf_view_df = clean_metadata_columns(leaf_view_df)
root_view_df = clean_metadata_columns(root_view_df)

# Rename columns explicitly if they exist from loading old data formats
leaf_attn_df = rename_legacy_columns(leaf_attn_df)
root_attn_df = rename_legacy_columns(root_attn_df)
leaf_overall_df = rename_legacy_columns(leaf_overall_df)
root_overall_df = rename_legacy_columns(root_overall_df)
leaf_view_df = rename_legacy_columns(leaf_view_df)
root_view_df = rename_legacy_columns(root_view_df)

# Process network data
print("\nPreparing network data...")
leaf_data = prepare_network_data(leaf_attn_df, "Leaf", n_top_connections=40, min_attention=0.01)
root_data = prepare_network_data(root_attn_df, "Root", n_top_connections=40, min_attention=0.01)

# Calculate network metrics
print("\nCalculating network metrics...")
network_metrics_df = calculate_network_stats(
    leaf_overall_df, root_overall_df, 
    leaf_attn_df, root_attn_df,
    leaf_view_df, root_view_df
)

# Calculate node importances for network visualization
if leaf_data:
    leaf_spec_importance, leaf_molecular_feature_importance = calculate_node_importance(leaf_data)
else:
    leaf_spec_importance, leaf_molecular_feature_importance = {}, {}

if root_data:
    root_spec_importance, root_molecular_feature_importance = calculate_node_importance(root_data)
else:
    root_spec_importance, root_molecular_feature_importance = {}, {}

# Create the integrated figure
print("\nCreating integrated visualization...")
fig = plt.figure(figsize=(20, 20))

# Create consistent layouts for both genotypes within each tissue
leaf_positions = None
root_positions = None

# Create the G1 leaf network (top left)
ax_leaf_g1 = fig.add_subplot(3, 2, 1)
if leaf_data:
    leaf_g1_G, leaf_g1_weights, leaf_positions = create_network_visualization(
        fig, ax_leaf_g1, leaf_data, 'G1', 'Leaf',
        (leaf_spec_importance, leaf_molecular_feature_importance)
    )
else:
    ax_leaf_g1.text(0.5, 0.5, "Leaf - G1: Data not available", 
                   ha='center', va='center', fontsize=14)

# Create the G2 leaf network (top right)
ax_leaf_g2 = fig.add_subplot(3, 2, 2)
if leaf_data:
    leaf_g2_G, leaf_g2_weights, _ = create_network_visualization(
        fig, ax_leaf_g2, leaf_data, 'G2', 'Leaf',
        (leaf_spec_importance, leaf_molecular_feature_importance),
        pos=leaf_positions
    )
else:
    ax_leaf_g2.text(0.5, 0.5, "Leaf - G2: Data not available", 
                   ha='center', va='center', fontsize=14)

# Create the G1 root network (middle left)
ax_root_g1 = fig.add_subplot(3, 2, 3)
if root_data:
    root_g1_G, root_g1_weights, root_positions = create_network_visualization(
        fig, ax_root_g1, root_data, 'G1', 'Root',
        (root_spec_importance, root_molecular_feature_importance)
    )
else:
    ax_root_g1.text(0.5, 0.5, "Root - G1: Data not available", 
                   ha='center', va='center', fontsize=14)

# Create the G2 root network (middle right)
ax_root_g2 = fig.add_subplot(3, 2, 4)
if root_data:
    root_g2_G, root_g2_weights, _ = create_network_visualization(
        fig, ax_root_g2, root_data, 'G2', 'Root',
        (root_spec_importance, root_molecular_feature_importance),
        pos=root_positions
    )
else:
    ax_root_g2.text(0.5, 0.5, "Root - G2: Data not available", 
                   ha='center', va='center', fontsize=14)

# Add panel labels for all subplot components
# Leaf networks
ax_leaf_g1.text(-0.12, 1.1, 'A)', fontsize=22, fontweight='bold',
               ha='center', va='center', transform=ax_leaf_g1.transAxes)
ax_leaf_g2.text(-0.12, 1.1, 'B)', fontsize=22, fontweight='bold', 
               ha='center', va='center', transform=ax_leaf_g2.transAxes)

# Root networks
ax_root_g1.text(-0.12, 1.1, 'C)', fontsize=22, fontweight='bold', 
               ha='center', va='center', transform=ax_root_g1.transAxes)
ax_root_g2.text(-0.12, 1.1, 'D)', fontsize=22, fontweight='bold', 
               ha='center', va='center', transform=ax_root_g2.transAxes)

# Create the network statistics visualization in the bottom row
stats_axes = create_network_stats_visualization(fig, network_metrics_df, 2)

# Create common legends for the network visualizations
# Make space for network visualization legend below the networks
fig.subplots_adjust(bottom=0.2)

# Create legend for node categories
spectral_elements = [
    mpatches.Patch(color=COLORS['Spectral_Water'], label='Water Bands'),
    mpatches.Patch(color=COLORS['Spectral_Pigment'], label='Pigment Regions'),
    mpatches.Patch(color=COLORS['Spectral_Structure'], label='Structure/NIR'),
    mpatches.Patch(color=COLORS['Spectral_SWIR'], label='SWIR Regions'),
    mpatches.Patch(color=COLORS['Spectral_VIS'], label='Visible'),
    mpatches.Patch(color=COLORS['Spectral_Other'], label='Other')
]

molecular_feature_elements = [
    mpatches.Patch(color=COLORS['P'], label='P Molecular Features'),
    mpatches.Patch(color=COLORS['N'], label='N Molecular Features')
]

# Create size legend
size_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
          label='Low Importance'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
          label='Medium Importance'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=16, 
          label='High Importance')
]

# Create legends
# Spectral features legend at top right
legend1 = fig.legend(handles=spectral_elements, loc='upper right', ncol=1, fontsize=16,
                   title="Spectral Features", title_fontsize=18,
                   bbox_to_anchor=(0.98, 0.95))

# Molecular features legend below spectral features
legend2 = fig.legend(handles=molecular_feature_elements, loc='upper right', ncol=1, fontsize=16,
                   title="Molecular Features", title_fontsize=18,
                   bbox_to_anchor=(0.98, 0.80))

# Node size legend below molecular features
legend3 = fig.legend(handles=size_elements, loc='upper right', ncol=1, fontsize=16,
                   title="Node Size", title_fontsize=18,
                   bbox_to_anchor=(0.98, 0.65))

# Add colorbar
cbar_ax = fig.add_axes([0.3, 0.97, 0.4, 0.015])
sm = plt.cm.ScalarMappable(cmap=edge_color_map)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Attention Strength', fontsize=18, labelpad=1)
cbar.ax.tick_params(labelsize=16)

# Add comprehensive caption
caption = (
    "This figure reveals differential cross-modal attention patterns between drought-tolerant (G1) and "
    "drought-susceptible (G2) genotypes under osmotic stress. Panels A-D show the top 40 strongest spectral-molecular feature "
    "connections for leaf (A,B) and root (C,D) tissues, with node size indicating feature importance and edge thickness "
    "representing attention strength. Connections are directed from spectral features (left) to molecular features (right), "
    "highlighting the flow of information from physiological status to biochemical response. Panel E quantifies these "
    "differences through three network metrics: mean attention across top pairs (indicating overall communication strength), "
    "standard deviation (reflecting focused vs. distributed attention), and 95th percentile values (showing peak connection strength). "
    "Percentages indicate G1-G2 differences relative to G2. The tolerant genotype (G1) demonstrates distinct attention "
    "deployment patterns compared to the susceptible genotype (G2), particularly in utilizing key drought-adaptive metabolic "
    "pathways."
)
fig.text(0.5, 0.01, caption, wrap=True, ha='center', va='top', fontsize=TEXT['caption'])

# Add vertical spacing between subplot rows and adjust figure layout
fig.subplots_adjust(hspace=3.5)

# Adjust layout
plt.tight_layout(rect=[0, 0.15, 0.85, 0.92])

# Manually adjust the bottom bar plot to use full width but be higher up
for i in range(len(fig.axes)):
    if i == 4:  # The bar plot is the 5th axes (index 4)
        fig.axes[i].set_position([0.1, 0.12, 0.8, 0.25])

output_path = os.path.join(output_dir, "fig5_integrated_networks_and_statistics.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nIntegrated Figure 5 saved to: {output_path}")
print("Figure generation complete!\n")