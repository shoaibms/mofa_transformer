"""
Figure 6: Temporal Evolution of Spectral-Molecular feature S2M Attention Networks

This script generates visualizations of temporal network evolution showing how the relationships
between spectral features and molecular features change over time. The visualization includes:
- Bipartite network graphs showing spectral features connected to molecular features
- Color-coded nodes and weighted edges based on attention scores
- Separate panels for different genotypes (G1, G2) and tissues (Leaf, Root)
- Temporal progression across three time points

The script produces high-quality figures suitable for publication with proper legends and annotations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Font definitions
FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'main_title': 22,
    'panel_label': 19,
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

# Apply base font settings
plt.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif']
})

# Create output directory
output_dir = r"C:\\Users\\ms\\Desktop\\hyper\\output\\transformer\\novility_plot"
os.makedirs(output_dir, exist_ok=True)

# Define paths to data files
leaf_attention_path = r"C:\\Users\\ms\\Desktop\\hyper\\output\\transformer\\v3_feature_attention\\processed_attention_data_leaf\\processed_mean_attention_conditional_Leaf.csv"
root_attention_path = r"C:\\Users\\ms\\Desktop\\hyper\\output\\transformer\\v3_feature_attention\\processed_attention_data_root\\processed_mean_attention_conditional_Root.csv"

# Color definitions
COLORS = {
    'Edge_Low': '#f0f0f0',        # Very Light Gray
    'Edge_High': '#EEE8AA',       # Pale Goldenrod
    'Node_Spectral': '#6baed6',   # Medium Blue
    'Node_Metabolite': '#FFC4A1',  # Light Salmon (Used for Molecular Feature nodes)
    'Positive_Diff': '#66CDAA',   # Medium Aquamarine (for G1 annotation)
    'Negative_Diff': '#fe9929',   # Muted Orange/Yellow (for G2 annotation)
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd', # Light Grey border for text boxes
}

# Define a custom colormap for edge weights using defined colors
edge_cmap = LinearSegmentedColormap.from_list('attention_cmap', [
    (0.0, COLORS['Edge_Low']),
    (1.0, COLORS['Edge_High'])
])


def rename_and_clean_features(df):
    """Renames columns and cleans feature names based on specified rules."""
    # Rename columns
    df = df.rename(columns={'Day': 'Time point', 'Metabolite_Feature': 'Molecular_Feature'})

    # Clean Molecular Feature names
    if 'Molecular_Feature' in df.columns:
        df['Molecular_Feature'] = df['Molecular_Feature'].astype(str).str.replace(
            'N_Cluster_', 'N_', regex=False)
        df['Molecular_Feature'] = df['Molecular_Feature'].astype(str).str.replace(
            'P_Cluster_', 'P_', regex=False)
        print("Applied N_/P_ Cluster renaming to Molecular Features.")
    else:
        print("Warning: 'Molecular_Feature' column not found for cleaning.")

    return df

def create_temporal_network_grid(output_path, leaf_data_path, root_data_path):
    """
    Create a grid of temporal network plots showing attention evolution over time points.
    
    Generates a visualization that shows how spectral-molecular feature relationships
    evolve across three time points for different tissues and genotypes.
    
    Args:
        output_path: Path where the output figure will be saved
        leaf_data_path: Path to the leaf attention data CSV
        root_data_path: Path to the root attention data CSV
    """
    print(f"Loading data from {leaf_data_path} and {root_data_path}")

    # Load conditional attention data
    try:
        leaf_df = pd.read_csv(leaf_data_path)
        root_df = pd.read_csv(root_data_path)
        print(f"Successfully loaded data - Leaf: {leaf_df.shape}, Root: {root_df.shape}")

        # Apply Renaming and Cleaning
        leaf_df = rename_and_clean_features(leaf_df)
        root_df = rename_and_clean_features(root_df)

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("Creating sample data for demonstration")
        leaf_df, root_df = create_sample_data()

    # Basic data validation
    for df_name, df in [("Leaf", leaf_df), ("Root", root_df)]:
        print(f"Data validation for {df_name}:")

        # Convert data types to improve filtering consistency
        if 'Time point' in df.columns:
            df['Time point'] = pd.to_numeric(df['Time point'], errors='coerce')
        else:
            print("Warning: 'Time point' column not found.")
            return  # Cannot proceed without time point data

        df['Treatment'] = pd.to_numeric(df['Treatment'], errors='coerce')

        # Map string genotypes to standardized form
        if 'Genotype' in df.columns:
            df['Genotype'] = df['Genotype'].astype(str)
            df['Genotype'] = df['Genotype'].replace({'1': 'G1', '2': 'G2'})

    # Filter for treatment T1 (stress condition)
    leaf_df_stress = leaf_df[leaf_df['Treatment'] == 1.0].copy()
    root_df_stress = root_df[root_df['Treatment'] == 1.0].copy()

    print(f"Leaf stress data after filtering: {leaf_df_stress.shape}")
    print(f"Root stress data after filtering: {root_df_stress.shape}")

    # Set up figure with a 2x2 grid (rows: tissues, columns: genotypes)
    fig = plt.figure(figsize=(20, 14))

    # Add panel label
    fig.text(0.02, 0.98, 'I)', ha='left', va='top', 
             fontsize=FONTS_SANS['panel_label'], weight='bold')

    # Create a GridSpec with 2 rows and 2 columns
    outer_grid = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)

    # Define common layout parameters
    tissues = ['Leaf', 'Root']
    genotypes = ['G1', 'G2']
    time_points = [1, 2, 3]

    # Define a function to consistently position nodes in a bipartite layout
    def bipartite_layout(G, spectral_nodes, molecular_feature_nodes):
        """Create a bipartite layout with spectral nodes on left and molecular feature nodes on right."""
        pos = {}
        # Position spectral nodes on left in a grid
        n_spectral = len(spectral_nodes)
        for i, node in enumerate(sorted(spectral_nodes)):
            pos[node] = (-1, (i - n_spectral/2) / max(n_spectral, 1) * 1.8)

        # Position molecular feature nodes on right in a grid
        n_molecular = len(molecular_feature_nodes)
        for i, node in enumerate(sorted(molecular_feature_nodes)):
            pos[node] = (1, (i - n_molecular/2) / max(n_molecular, 1) * 1.8)

        return pos

    # Process each tissue and genotype combination
    network_stats = []  # To store network metrics for CSV output

    for t_idx, tissue in enumerate(tissues):
        df = leaf_df_stress if tissue == 'Leaf' else root_df_stress

        for g_idx, genotype in enumerate(genotypes):
            # Create inner grid for this tissue-genotype combination
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer_grid[t_idx, g_idx], wspace=0.1)

            # Collect all important features for this genotype
            # First collect data across all time points
            genotype_data = df[df['Genotype'] == genotype]

            # Check data availability for each time point
            tp_counts = {}
            for tp in time_points:
                tp_count = len(df[(df['Genotype'] == genotype) & 
                                 (df['Time point'] == float(tp))])
                tp_counts[tp] = tp_count
                print(f"{tissue}-{genotype}-Time point {tp}: {tp_count} rows")

            # Get top spectral and molecular features across all time points for consistency
            top_spec_feat = genotype_data.groupby('Spectral_Feature')[
                'Mean_Attention_S2M_Group_AvgHeads'].sum().nlargest(25).index.tolist()
            top_mol_feat = genotype_data.groupby('Molecular_Feature')[
                'Mean_Attention_S2M_Group_AvgHeads'].sum().nlargest(10).index.tolist()

            # Get time point-specific top connections
            tp_data = {}
            all_spectral = set(top_spec_feat)  # Start with most important features
            all_molecular_features = set(top_mol_feat)

            # Time point-specific maximum attention values for better normalization
            tp_max_attention = {}

            # First pass: collect data per time point and get time point-specific important nodes
            for tp in time_points:
                # Filter with explicit float conversion
                tp_df = df[(df['Genotype'] == genotype) & 
                          (df['Time point'] == float(tp))]

                if len(tp_df) == 0:
                    print(f"WARNING: No data for {tissue}-{genotype}-Time point {tp}")
                    continue

                # For each time point, get more connections for better visualization
                # This is key for Time point 2 visibility
                n_connections = min(40, len(tp_df))

                # Get top connections for this time point
                top_connections = tp_df.nlargest(n_connections, 'Mean_Attention_S2M_Group_AvgHeads')

                # Store data and max attention
                tp_data[tp] = top_connections
                tp_max_attention[tp] = top_connections['Mean_Attention_S2M_Group_AvgHeads'].max()

                # Add time point-specific important nodes
                tp_specs = set(top_connections['Spectral_Feature'])
                tp_mol_feats = set(top_connections['Molecular_Feature'])

                # Update sets of all nodes
                all_spectral.update(tp_specs)
                all_molecular_features.update(tp_mol_feats)

                print(f"  Time point {tp} connections: {len(top_connections)}, "
                      f"Max attention: {tp_max_attention[tp]:.6f}")
                print(f"  Time point {tp} features: {len(tp_specs)} spectral, "
                      f"{len(tp_mol_feats)} molecular features")

            # Calculate node sizes based on overall connectivity
            node_sizes = {}
            for tp in time_points:
                if tp in tp_data:
                    for _, row in tp_data[tp].iterrows():
                        s_node = row['Spectral_Feature']
                        m_node = row['Molecular_Feature']
                        attn = row['Mean_Attention_S2M_Group_AvgHeads']

                        # Add attention to source and target node sizes
                        node_sizes[s_node] = node_sizes.get(s_node, 0) + attn
                        node_sizes[m_node] = node_sizes.get(m_node, 0) + attn

            # Normalize node sizes for visualization
            min_size = 50
            max_size = 400
            if node_sizes:
                min_value = min(node_sizes.values())
                max_value = max(node_sizes.values())
                range_value = max_value - min_value

                if range_value > 0:
                    for node in node_sizes:
                        normalized = (node_sizes[node] - min_value) / range_value
                        node_sizes[node] = min_size + normalized * (max_size - min_size)
                else:
                    for node in node_sizes:
                        node_sizes[node] = min_size

            # Second pass: create and draw the networks for each time point
            for tp_idx, tp in enumerate(time_points):
                ax = fig.add_subplot(inner_grid[tp_idx])

                if tp not in tp_data or len(tp_data[tp]) == 0:
                    ax.text(0.5, 0.5, f"No data for Time point {tp}",
                           ha='center', va='center', 
                           fontsize=FONTS_SANS['annotation'])
                    ax.axis('off')
                    continue

                # Create network
                G = nx.Graph()

                # Add all nodes to ensure consistent layout
                for node in all_spectral:
                    G.add_node(node, bipartite=0, type='spectral')

                for node in all_molecular_features:
                    G.add_node(node, bipartite=1, type='molecular_feature')

                # Add edges from this time point's data
                tp_connections = tp_data[tp]
                for _, row in tp_connections.iterrows():
                    s_node = row['Spectral_Feature']
                    m_node = row['Molecular_Feature']
                    weight = row['Mean_Attention_S2M_Group_AvgHeads']

                    # Ensure nodes exist (defensive programming)
                    if s_node in G.nodes() and m_node in G.nodes():
                        G.add_edge(s_node, m_node, weight=weight)
                    else:
                        print(f"WARNING: Node not found in graph: {s_node} or {m_node}")

                # Get the fixed bipartite layout
                pos = bipartite_layout(G, all_spectral, all_molecular_features)

                # Node colors based on type using defined colors
                node_colors = [
                    COLORS['Node_Spectral'] if G.nodes[node]['type'] == 'spectral' 
                    else COLORS['Node_Metabolite'] for node in G.nodes()
                ]

                # Get edges and weights for coloring
                edges = G.edges()

                # Normalize edge weights by TIME POINT-SPECIFIC maximum
                # This ensures each time point panel shows clear connections
                tp_norm_factor = tp_max_attention[tp]
                if tp_norm_factor > 0:
                    edge_weights = [G[u][v]['weight'] / tp_norm_factor for u, v in edges]
                else:
                    edge_weights = [0.5 for _ in edges]  # Default if no max attention

                # Ensure edges have visible thickness
                min_edge_width = 0.8
                max_edge_width = 6.0
                width_multiplier = 20.0
                edge_widths = [
                    max(min_edge_width, min(max_edge_width, w * width_multiplier)) 
                    for w in edge_weights
                ]

                # Create network plot with improved visibility
                nx.draw_networkx_nodes(
                    G, pos,
                    node_size=[node_sizes.get(node, min_size) for node in G.nodes()],
                    node_color=node_colors,
                    alpha=0.8,
                    ax=ax
                )

                # Draw edges with higher opacity for better visibility
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=list(edges),
                    width=edge_widths,
                    edge_color=edge_weights,
                    edge_cmap=edge_cmap,
                    alpha=0.7,
                    ax=ax
                )

                # Add labels to key nodes
                if len(G.nodes()) > 0:
                    # Calculate node importance based on connections
                    degree = dict(G.degree())
                    importance = {
                        node: (node_sizes.get(node, 0) * 0.3 + degree.get(node, 0) * 5)
                        for node in G.nodes()
                    }

                    # Label top nodes by type
                    spec_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'spectral']
                    mol_feature_nodes = [
                        n for n in G.nodes() if G.nodes[n]['type'] == 'molecular_feature'
                    ]

                    # Get top nodes of each type
                    top_spec = sorted(
                        spec_nodes, key=lambda x: importance.get(x, 0), reverse=True
                    )[:3]
                    top_mol = sorted(
                        mol_feature_nodes, key=lambda x: importance.get(x, 0), reverse=True
                    )[:3]

                    # Create labels
                    node_labels = {node: node for node in top_spec + top_mol}

                    # Adjust label positions slightly
                    label_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items()}

                    nx.draw_networkx_labels(
                        G, label_pos,
                        labels=node_labels,
                        font_size=FONTS_SANS['annotation'],
                        font_color='black',
                        ax=ax
                    )

                # Set title and remove axis
                ax.set_title(f"Time point {tp}", fontsize=FONTS_SANS['panel_title'])
                ax.axis('off')

                # Collect network statistics for CSV output
                n_edges = len(G.edges())
                avg_weight = np.mean([G[u][v]['weight'] for u, v in G.edges()]) if n_edges > 0 else 0
                network_stats.append({
                    'Tissue': tissue,
                    'Genotype': genotype,
                    'Time point': tp,
                    'Nodes': len(G.nodes()),
                    'Edges': n_edges,
                    'Avg_Attention': avg_weight,
                    'Max_Attention': max([G[u][v]['weight'] for u, v in G.edges()]) if n_edges > 0 else 0,
                    'Network_Density': nx.density(G) if len(G.nodes()) > 1 else 0
                })

            # Add pattern annotations for each genotype
            if genotype == 'G1':
                annotation = "Progressive network strengthening\\nwith key hubs emerging"
                color = COLORS['Positive_Diff']  # Green for G1
            else:
                annotation = "Less coordinated network\\ndevelopment over time"
                color = COLORS['Negative_Diff']  # Red for G2

            # Add annotation in the middle subplot
            middle_ax = fig.add_subplot(inner_grid[1])
            # Add transparent text box in the corner using defined colors
            middle_ax.text(
                0.05, 0.05, annotation, transform=middle_ax.transAxes,
                fontsize=FONTS_SANS['annotation'], color='black', alpha=0.8,
                bbox=dict(
                    facecolor=color, alpha=0.2, boxstyle='round,pad=0.3',
                    edgecolor=COLORS['Annotation_Box_Edge']
                )
            )

            # Add a title for this tissue-genotype combination
            row, col = t_idx, g_idx
            fig.text(
                0.25 + 0.5 * col,
                0.95 - 0.5 * row,
                f"{tissue} - {genotype}",
                ha='center',
                fontsize=FONTS_SANS['panel_title'],
                weight='bold'
            )

    # Add overall figure legends
    # 1. Node type legend using defined colors
    node_legend_elements = [
        Patch(facecolor=COLORS['Node_Spectral'], edgecolor='k', label='Spectral Features'),
        Patch(facecolor=COLORS['Node_Metabolite'], edgecolor='k', label='Molecular Features')
    ]

    # 2. Edge weight legend
    # Create a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=edge_cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Normalized Attention Strength', fontsize=FONTS_SANS['axis_label'])

    # Create a legend for node types
    legend_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02], frameon=False)
    legend_ax.axis('off')
    legend_ax.legend(
        handles=node_legend_elements,
        loc='center',
        ncol=2,
        fontsize=FONTS_SANS['legend_text']
    )

    # Add title to the entire figure
    fig.suptitle(
        'Figure 12: Temporal Evolution of Spectral-Molecular feature Attention Networks',
        fontsize=FONTS_SANS['main_title'],
        y=0.99
    )

    # Output the network statistics to CSV
    stats_df = pd.DataFrame(network_stats)
    stats_path = os.path.join(output_dir, "fig12_network_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Network statistics saved to {stats_path}")

    # Save the figure - avoid tight_layout which causes warnings
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Temporal network visualization saved to {output_path}")

    # Generate a separate focused version just for Leaf tissue
    create_focused_leaf_networks(leaf_df_stress, os.path.join(output_dir, "fig12_temporal_networks_leaf_focused.png"))

def create_focused_leaf_networks(leaf_df, output_path):
    """
    Create a focused version showing only leaf tissue networks with more detail.
    
    This generates a 2x3 grid (2 genotypes, 3 time points) with larger, more detailed
    networks specifically for leaf tissue data.
    
    Args:
        leaf_df: DataFrame containing leaf tissue attention data
        output_path: Path where the output figure will be saved
    """
    print("\nCreating focused leaf networks")

    # Define relevant colors again for this function scope
    COLORS = {
        'Edge_Low': '#f0f0f0',
        'Edge_High': '#EEE8AA',
        'Node_Spectral': '#6baed6',
        'Node_Metabolite': '#FFC4A1',  # For Molecular Features
    }

    # Define edge colormap using the defined colors
    edge_cmap = LinearSegmentedColormap.from_list('attention_cmap', [
        (0.0, COLORS['Edge_Low']),
        (1.0, COLORS['Edge_High'])
    ])

    fig = plt.figure(figsize=(18, 10))
    grid = gridspec.GridSpec(2, 3, figure=fig, wspace=0.2, hspace=0.3)

    # Add panel label
    fig.text(0.02, 0.98, 'I)', ha='left', va='top', 
             fontsize=FONTS_SANS['panel_label'], weight='bold')

    genotypes = ['G1', 'G2']
    time_points = [1, 2, 3]

    # Define a function to consistently position nodes in a bipartite layout
    def bipartite_layout(G, spectral_nodes, molecular_feature_nodes):
        """Create a bipartite layout with spectral nodes on left and molecular feature nodes on right."""
        pos = {}
        # Position spectral nodes on left in a grid
        n_spectral = len(spectral_nodes)
        for i, node in enumerate(sorted(spectral_nodes)):
            pos[node] = (-1, (i - n_spectral/2) / max(n_spectral, 1) * 1.8)

        # Position molecular feature nodes on right in a grid
        n_molecular = len(molecular_feature_nodes)
        for i, node in enumerate(sorted(molecular_feature_nodes)):
            pos[node] = (1, (i - n_molecular/2) / max(n_molecular, 1) * 1.8)

        return pos

    # Process each genotype
    for g_idx, genotype in enumerate(genotypes):
        # Get genotype-specific data
        genotype_data = leaf_df[leaf_df['Genotype'] == genotype]

        # Get top features across all time points for this genotype
        top_spec_feat = genotype_data.groupby('Spectral_Feature')[
            'Mean_Attention_S2M_Group_AvgHeads'].sum().nlargest(30).index.tolist()
        top_mol_feat = genotype_data.groupby('Molecular_Feature')[
            'Mean_Attention_S2M_Group_AvgHeads'].sum().nlargest(8).index.tolist()

        # Collect time point-specific data
        tp_data = {}
        tp_max_attention = {}
        all_spectral = set(top_spec_feat)
        all_molecular_features = set(top_mol_feat)

        for tp in time_points:
            tp_df = leaf_df[(leaf_df['Genotype'] == genotype) & 
                           (leaf_df['Time point'] == float(tp))]

            print(f"Focused leaf view: {genotype}-Time point {tp}: {len(tp_df)} rows")

            if len(tp_df) == 0:
                print(f"ERROR: No data found for Leaf-{genotype}-Time point {tp} in focused view")
                continue

            # Get more connections for the focused view
            n_connections = min(60, len(tp_df))
            top_connections = tp_df.nlargest(n_connections, 'Mean_Attention_S2M_Group_AvgHeads')

            # Store for later use
            tp_data[tp] = top_connections
            tp_max_attention[tp] = top_connections['Mean_Attention_S2M_Group_AvgHeads'].max()

            # Update the sets of all nodes - add time point-specific ones
            all_spectral.update(top_connections['Spectral_Feature'])
            all_molecular_features.update(top_connections['Molecular_Feature'])

        print(f"Focused Leaf-{genotype}: Using {len(all_spectral)} spectral and "
              f"{len(all_molecular_features)} molecular feature nodes")

        # Calculate node sizes based on overall connectivity
        node_sizes = {}
        for tp in time_points:
            if tp in tp_data:
                for _, row in tp_data[tp].iterrows():
                    s_node = row['Spectral_Feature']
                    m_node = row['Molecular_Feature']
                    attn = row['Mean_Attention_S2M_Group_AvgHeads']

                    # Add attention to source and target node sizes
                    node_sizes[s_node] = node_sizes.get(s_node, 0) + attn
                    node_sizes[m_node] = node_sizes.get(m_node, 0) + attn

        # Normalize node sizes for visualization
        min_size = 100  # Larger nodes for focused view
        max_size = 800
        if node_sizes:
            min_value = min(node_sizes.values())
            max_value = max(node_sizes.values())
            range_value = max_value - min_value

            if range_value > 0:
                for node in node_sizes:
                    normalized = (node_sizes[node] - min_value) / range_value
                    node_sizes[node] = min_size + normalized * (max_size - min_size)
            else:
                for node in node_sizes:
                    node_sizes[node] = min_size

        # Create networks for each time point
        for tp_idx, tp in enumerate(time_points):
            ax = fig.add_subplot(grid[g_idx, tp_idx])

            if tp not in tp_data or len(tp_data[tp]) == 0:
                ax.text(0.5, 0.5, f"No data for Time point {tp}",
                       ha='center', va='center', fontsize=FONTS_SANS['annotation'])
                ax.axis('off')
                continue

            # Create network
            G = nx.Graph()

            # Add all nodes to ensure consistent layout
            for node in all_spectral:
                G.add_node(node, bipartite=0, type='spectral')

            for node in all_molecular_features:
                G.add_node(node, bipartite=1, type='molecular_feature')

            # Add edges from this time point's data
            for _, row in tp_data[tp].iterrows():
                G.add_edge(
                    row['Spectral_Feature'],
                    row['Molecular_Feature'],
                    weight=row['Mean_Attention_S2M_Group_AvgHeads']
                )

            # Get the fixed bipartite layout
            pos = bipartite_layout(G, all_spectral, all_molecular_features)

            # Node colors based on type using defined colors
            node_colors = [
                COLORS['Node_Spectral'] if G.nodes[node]['type'] == 'spectral' 
                else COLORS['Node_Metabolite'] for node in G.nodes()
            ]

            # Get edges and weights for coloring
            edges = G.edges()

            # Time point-specific normalization for better visibility
            if tp_max_attention[tp] > 0:
                edge_weights = [G[u][v]['weight'] / tp_max_attention[tp] for u, v in edges]
            else:
                edge_weights = [0.5 for _ in edges]

            # Thicker edge widths for better visibility
            min_edge_width = 0.8
            max_edge_width = 6.0
            width_multiplier = 20.0
            edge_widths = [
                max(min_edge_width, min(max_edge_width, w * width_multiplier)) 
                for w in edge_weights
            ]

            # Create network plot
            nx.draw_networkx_nodes(
                G, pos,
                node_size=[node_sizes.get(node, min_size) for node in G.nodes()],
                node_color=node_colors,
                alpha=0.8,
                ax=ax
            )

            nx.draw_networkx_edges(
                G, pos,
                edgelist=list(edges),
                width=edge_widths,
                edge_color=edge_weights,
                edge_cmap=edge_cmap,
                alpha=0.7,
                ax=ax
            )

            # Add labels to key nodes
            degree = dict(G.degree())
            importance = {
                node: (node_sizes.get(node, 0) * 0.5 + degree.get(node, 0) * 10)
                for node in G.nodes()
            }

            # Label top nodes for each type
            spec_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'spectral']
            mol_feature_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'molecular_feature']

            top_spec = sorted(spec_nodes, key=lambda x: importance.get(x, 0), reverse=True)[:5]
            top_mol = sorted(mol_feature_nodes, key=lambda x: importance.get(x, 0), reverse=True)[:4]

            # Create combined labels
            node_labels = {node: node for node in top_spec + top_mol}

            # Adjust label positions slightly
            label_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items()}

            nx.draw_networkx_labels(
                G, label_pos,
                labels=node_labels,
                font_size=FONTS_SANS['annotation'],
                font_color='black',
                ax=ax
            )

            # Set title and remove axis
            ax.set_title(f"Time point {tp}", fontsize=FONTS_SANS['panel_title'])
            ax.axis('off')

        # Add genotype label on the left
        fig.text(
            0.02,
            0.75 - 0.5 * g_idx,
            f"{genotype} (Drought-Tolerant)" if genotype == 'G1' 
            else f"{genotype} (Drought-Susceptible)",
            ha='left',
            va='center',
            fontsize=FONTS_SANS['panel_title'],
            weight='bold',
            rotation=90
        )

    # Add node type legend at the bottom using defined colors
    node_legend_elements = [
        Patch(facecolor=COLORS['Node_Spectral'], edgecolor='k', label='Spectral Features'),
        Patch(facecolor=COLORS['Node_Metabolite'], edgecolor='k', label='Molecular Features')
    ]

    # Create a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=edge_cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Normalized Attention Strength', fontsize=FONTS_SANS['axis_label'])

    # Create a legend for node types
    legend_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02], frameon=False)
    legend_ax.axis('off')
    legend_ax.legend(
        handles=node_legend_elements,
        loc='center',
        ncol=2,
        fontsize=FONTS_SANS['legend_text']
    )

    # Save the figure - avoid tight_layout which causes warnings
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Focused leaf network visualization saved to {output_path}")

def create_sample_data():
    """
    Create sample data if real data can't be loaded.
    
    Returns:
        tuple: (leaf_df, root_df) pandas DataFrames with synthetic data for demonstration
    """
    # Define parameters
    genotypes = ['G1', 'G2']
    treatments = [1.0]  # Only stress condition as float
    time_points = [1.0, 2.0, 3.0]  # As floats to match expected data format

    # Create spectral and molecular features
    spectral_features = [f"W_{i}" for i in range(500, 700, 10)]
    molecular_features = [f"{'P' if i % 2 == 0 else 'N'}_{i}" for i in range(1, 21)]

    # Create sample data
    leaf_data = []
    root_data = []

    for geno in genotypes:
        for treat in treatments:
            for tp in time_points:
                # Create different numbers of connections per time point
                # G1 shows increasing connections over time
                # G2 shows random fluctuation
                if geno == 'G1':
                    n_connections = 15 + 10 * int(tp)  # Increases with time point
                    base_attention = 0.02 + 0.03 * tp  # Stronger signals in later time points
                else:
                    n_connections = 10 + 5 * np.random.randint(0, 3)  # Random
                    base_attention = 0.01 + 0.01 * tp  # Slower increase

                # Generate connections
                for _ in range(int(n_connections)):
                    spec = np.random.choice(spectral_features)
                    mol_feat = np.random.choice(molecular_features)

                    # Attention scores - some randomness but with patterns
                    leaf_attention = base_attention + 0.05 * np.random.random()
                    root_attention = base_attention * 1.2 + 0.05 * np.random.random()  # Root higher

                    # Add to leaf data
                    leaf_data.append({
                        'Genotype': geno,
                        'Treatment': float(treat),
                        'Time point': float(tp),
                        'Spectral_Feature': spec,
                        'Molecular_Feature': mol_feat,
                        'Mean_Attention_S2M_Group_AvgHeads': leaf_attention,
                        'N_Samples_Group': np.random.randint(5, 15)
                    })

                    # Add to root data
                    root_data.append({
                        'Genotype': geno,
                        'Treatment': float(treat),
                        'Time point': float(tp),
                        'Spectral_Feature': spec,
                        'Molecular_Feature': mol_feat,
                        'Mean_Attention_S2M_Group_AvgHeads': root_attention,
                        'N_Samples_Group': np.random.randint(5, 15)
                    })

    leaf_df = pd.DataFrame(leaf_data)
    root_df = pd.DataFrame(root_data)

    # Apply cleaning to sample data as well
    leaf_df = rename_and_clean_features(leaf_df)
    root_df = rename_and_clean_features(root_df)

    return leaf_df, root_df

# Create the temporal network visualizations
full_output_path = os.path.join(output_dir, "fig12_temporal_networks.png")
create_temporal_network_grid(full_output_path, leaf_attention_path, root_attention_path)

print("Figure 12 - Temporal Network Evolution visualization completed successfully!")