"""
Figure 4: Integrated View-Level Attention Statistics Analysis

This script visualizes cross-modal attention statistics across experimental conditions
by analyzing view-level attention data from a transformer model. It generates plots showing
how standard deviation and 95th percentile attention scores change over time across
different genotype-treatment combinations.

The script loads processed attention data for leaf and root tissues, creates visualizations
of attention patterns, and saves the output as a figure.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
import warnings
from colour import COLORS

warnings.filterwarnings('ignore')

# Define output directory
output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
os.makedirs(output_dir, exist_ok=True)

# Define input files
leaf_attention_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_leaf\processed_view_level_attention_Leaf.csv"
root_attention_path = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention\processed_attention_data_root\processed_view_level_attention_Root.csv"
leaf_stats_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.2\leaf\transformer_view_attention_stats_Leaf.csv"
root_stats_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.2\root\transformer_view_attention_stats_Root.csv"


def load_and_process_data():
    """Load and process view-level attention data files."""
    print("Loading view-level attention data...")
    
    # Load files
    try:
        leaf_data = pd.read_csv(leaf_attention_path)
        root_data = pd.read_csv(root_attention_path)
        print(f"Successfully loaded data: Leaf={leaf_data.shape}, Root={root_data.shape}")
    except Exception as e:
        print(f"Error loading attention data: {e}")
        raise
    
    # Load stats files if they exist
    try:
        leaf_stats = pd.read_csv(leaf_stats_path)
        root_stats = pd.read_csv(root_stats_path)
        stats_available = True
        print("Successfully loaded statistical analysis files")
    except Exception as e:
        print(f"Stats files not available or error loading stats: {e}")
        stats_available = False
        leaf_stats = None
        root_stats = None
    
    # Clean up data types
    for df_name, df in [("Leaf", leaf_data), ("Root", root_data)]:
        # Convert Day to numeric, then rename
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        df.rename(columns={'Day': 'Time point'}, inplace=True)
        
        # Ensure Genotype and Treatment are strings
        df['Genotype'] = df['Genotype'].astype(str)
        
        # Fix treatments: First convert to float, then int, then string to remove decimals
        try:
            # Handle numeric treatments with decimals
            df['Treatment'] = df['Treatment'].astype(float).astype(int).astype(str)
        except:
            # If that fails, just use string conversion
            df['Treatment'] = df['Treatment'].astype(str)
        
        # Clean up Genotype/Treatment values (handle numeric encodings)
        df['Genotype'] = df['Genotype'].replace({'1': 'G1', '2': 'G2'})
        df['Treatment'] = df['Treatment'].replace({'0': 'T0', '1': 'T1'})
    
    # Create combined condition column for plotting
    for df in [leaf_data, root_data]:
        df['Condition'] = df['Genotype'] + '-' + df['Treatment']
    
    return leaf_data, root_data, leaf_stats, root_stats, stats_available


def add_statistical_annotations(ax, data, metric_name, stats_df, tissue):
    """Add statistical significance markers to the plot based on stats data."""
    if stats_df is None:
        print(f"No stats data available for {tissue} {metric_name}")
        return
    
    try:
        # Check if required columns exist
        required_cols = ['Variable', 'Comparison', 'P_value']
        if not all(col in stats_df.columns for col in required_cols):
            print(f"Stats data for {tissue} missing required columns")
            return
        
        # Extract significant comparisons 
        sig_col = 'Significant_FDR' if 'Significant_FDR' in stats_df.columns else 'P_value'
        sig_threshold = 0.05 if sig_col == 'P_value' else True
        
        # Find significance based on variable name
        stats_subset = stats_df[stats_df['Variable'] == metric_name]
        
        if len(stats_subset) == 0:
            # Try alternative variable names
            alt_names = {
                'StdAttn_S2M': ['Std_S2M', 'StdDev_S2M', 'StdAttn_Spec_to_Metab'],
                'P95Attn_S2M': ['P95_S2M', 'P95Attn_Spec_to_Metab', '95Pct_S2M'],
                'StdAttn_M2S': ['Std_M2S', 'StdDev_M2S', 'StdAttn_Metab_to_Spec'],
                'P95Attn_M2S': ['P95_M2S', 'P95Attn_Metab_to_Spec', '95Pct_M2S']
            }
            
            if metric_name in alt_names:
                for alt_name in alt_names[metric_name]:
                    if alt_name in stats_df['Variable'].values:
                        stats_subset = stats_df[stats_df['Variable'] == alt_name]
                        print(f"Found alternative metric name: {alt_name} for {metric_name}")
                        break
        
        if len(stats_subset) == 0:
            return
        
        # Extract time point from comparison string
        def extract_time_point(comparison):
            try:
                if '_Time point' in comparison:
                    return int(comparison.split('_Time point')[1][0])
                elif 'Time point' in comparison:
                    return int(''.join(filter(str.isdigit, comparison.split('Time point')[1][0])))
            except:
                pass
            return None
            
        stats_subset['Time point'] = stats_subset['Comparison'].apply(extract_time_point)
        stats_subset.dropna(subset=['Time point'], inplace=True)
        stats_subset['Time point'] = stats_subset['Time point'].astype(int)
        
        # Add significance markers for each time point
        for time_point in [1, 2, 3]:
            time_point_stats = stats_subset[stats_subset['Time point'] == time_point]
            
            significant_comparisons = []
            for _, row in time_point_stats.iterrows():
                is_significant = row[sig_col] < sig_threshold if sig_col == 'P_value' else row[sig_col]
                if is_significant:
                    comparison = row['Comparison']
                    p_value = row['P_value']
                    significant_comparisons.append((comparison, p_value))
            
            if significant_comparisons:
                # Calculate position for significance marker
                time_point_data = data[data['Time point'] == time_point]
                if not time_point_data.empty:
                    y_max = time_point_data[metric_name].max()
                    y_range = time_point_data[metric_name].max() - time_point_data[metric_name].min()
                    
                    # Add asterisk at the top
                    y_pos = y_max + 0.1 * y_range
                    ax.text(time_point, y_pos, '*', ha='center', va='center', fontsize=15, 
                           color=COLORS.get('Significance', 'red'), 
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', 
                                    edgecolor='none'))
                    
                    # Add annotation for the first significant comparison
                    if significant_comparisons:
                        comparison, p_value = significant_comparisons[0]
                        if 'vs' in comparison:
                            groups = comparison.split('vs')
                            group1 = groups[0].strip().split('Time point')[0].strip()
                            group2 = groups[1].strip().split('Time point')[0].strip()
                            annotation = f"{group1} vs {group2}\np={p_value:.4f}"
                            
                            ax.annotate(
                                annotation,
                                xy=(time_point, y_pos),
                                xytext=(time_point + 0.2, y_pos + 0.15 * y_range),
                                arrowprops=dict(arrowstyle='->', lw=1, 
                                               color=COLORS.get('Significance', 'red')),
                                fontsize=15,
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, 
                                         ec=COLORS.get('Significance', 'red'))
                            )
                    
    except Exception as e:
        print(f"Error adding statistical annotations: {e}")


def create_view_level_attention_figure():
    """Create Figure 4: Condition-Dependent Dynamics of View-Level Attention Statistics."""
    # Load and process data
    leaf_data, root_data, leaf_stats, root_stats, stats_available = load_and_process_data()
    
    # Define attention statistics to plot
    primary_metrics = [
        {'name': 'StdAttn_S2M', 'label': 'Std. Dev. of S→M Attention'},
        {'name': 'P95Attn_S2M', 'label': '95th Percentile of S→M Attention'},
    ]
    
    # Check if M2S metrics are available and show meaningful patterns
    m2s_metrics = [
        {'name': 'StdAttn_M2S', 'label': 'Std. Dev. of M→S Attention'},
        {'name': 'P95Attn_M2S', 'label': '95th Percentile of M→S Attention'},
    ]
    
    has_m2s_metrics = all([m['name'] in leaf_data.columns and m['name'] in root_data.columns 
                          for m in m2s_metrics])
    
    if has_m2s_metrics:
        print("M→S attention metrics found - analyzing for meaningful patterns...")
        m2s_meaningful = False
        
        # Check if M2S metrics show substantial differences between genotypes
        for tissue, df in [('Leaf', leaf_data), ('Root', root_data)]:
            for metric in m2s_metrics:
                g1t1 = df[(df['Genotype'] == 'G1') & (df['Treatment'] == 'T1')][metric['name']].mean()
                g2t1 = df[(df['Genotype'] == 'G2') & (df['Treatment'] == 'T1')][metric['name']].mean()
                
                if not (np.isnan(g1t1) or np.isnan(g2t1) or g2t1 == 0):
                    pct_diff = (g1t1 - g2t1) / g2t1 * 100
                    if abs(pct_diff) > 10:  # >10% difference is meaningful
                        m2s_meaningful = True
                        print(f"Meaningful M→S pattern: {metric['name']} in {tissue}")
        
        if m2s_meaningful:
            primary_metrics.extend(m2s_metrics)
            print("Including M→S metrics in visualization")
    
    # Set up figure layout
    metrics_to_plot = primary_metrics[:2]  # Limit to S2M for now
    num_plots = len(metrics_to_plot) * 2  # 2 tissues * 2 metrics
    
    # Create figure with appropriate size for 1x4 layout + legend
    fig = plt.figure(figsize=(22, 6))
    gs = gridspec.GridSpec(1, num_plots, figure=fig)
    
    # Setup visual properties using imported COLORS
    colors = {
        'G1': COLORS.get('G1', '#1E88E5'),
        'G2': COLORS.get('G2', '#D81B60'),
        'T0': COLORS.get('T0', '#4682B4'),
        'T1': COLORS.get('T1', '#BDB76B')
    }
    
    linestyles = {
        'T0': 'dashed',   # Control (dashed)
        'T1': 'solid',    # Stress (solid)
    }
    
    # Marker styles
    markers = {
        'G1-T0': 'o',     # Circle for G1-T0
        'G1-T1': 's',     # Square for G1-T1
        'G2-T0': '^',     # Triangle for G2-T0
        'G2-T1': 'D',     # Diamond for G2-T1
    }
    
    # Create subplots for each metric and tissue
    plot_counter = 0
    axes = []  # Store axes to add legend later
    for metric_idx, metric in enumerate(metrics_to_plot): 
        for tissue_idx, tissue in enumerate(['Leaf', 'Root']):        
            data = leaf_data if tissue == 'Leaf' else root_data
            stats = leaf_stats if tissue == 'Leaf' else root_stats
            
            ax = fig.add_subplot(gs[0, plot_counter])
            
            # Skip if metric doesn't exist in data
            if metric['name'] not in data.columns:
                print(f"Warning: {metric['name']} not found in {tissue} data")
                ax.text(0.5, 0.5, f"No data for {metric['name']}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=15)
                continue
            
            # Plot the data
            for genotype in ['G1', 'G2']:
                for treatment in ['T0', 'T1']:
                    condition = f"{genotype}-{treatment}"
                    mask = (data['Genotype'] == genotype) & (data['Treatment'] == treatment)
                    subset = data[mask]
                    
                    if len(subset) > 0:
                        # Group by time point and calculate mean and standard error
                        grouped = subset.groupby('Time point')[metric['name']].agg(['mean', 'std', 'count'])
                        
                        # Check if we have enough time points for plotting
                        if len(grouped) >= 2:  # Need at least 2 points to draw a line
                            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
                            
                            # Plot the line
                            line = ax.plot(
                                grouped.index, 
                                grouped['mean'], 
                                marker=markers.get(condition, 'o'),
                                markersize=8,
                                linestyle=linestyles[treatment],
                                linewidth=2.5,
                                color=colors[genotype],
                                label=condition
                            )
                            
                            # Add confidence band
                            ax.fill_between(
                                grouped.index,
                                grouped['mean'] - grouped['se'],
                                grouped['mean'] + grouped['se'],
                                color=colors[genotype],
                                alpha=0.2
                            )
                        else:
                            # If only one time point is available, plot it as a single point
                            if len(grouped) == 1:
                                time_point_val = grouped.index[0]
                                ax.scatter(
                                    time_point_val,
                                    grouped['mean'].iloc[0],
                                    marker=markers.get(condition, 'o'),
                                    s=100,  # Larger than line markers
                                    color=colors[genotype],
                                    label=f"{condition} (Time point {int(time_point_val)} only)"
                                )
            
            # Customize plot
            ax.set_xlabel('Time point', fontsize=17)
            ax.set_ylabel(metric['label'], fontsize=17)
            ax.grid(True, linestyle='--', alpha=0.7, color=COLORS.get('Grid', '#d9d9d9'))
            
            # Create panel label (H, I, J, K)
            panel_label = chr(72 + plot_counter)  # Start from ASCII 72 ('H')
            direction = "S→M" if "S2M" in metric['name'] else "M→S"
            metric_type = "Structure" if "Std" in metric['name'] else "Intensity"
            
            ax.set_title(f"{panel_label}) {tissue} - {direction} Attention {metric_type}", 
                        fontsize=17, loc='left')
            
            ax.set_xticks([1, 2, 3])
            ax.set_xlim(0.8, 3.2)  # Add padding
            ax.tick_params(axis='both', labelsize=16)
            
            ax.set_facecolor(COLORS.get('Panel_Background', '#f7f7f7'))
            
            # Check and report if we plotted any data
            if len(ax.get_lines()) == 0:
                print(f"Warning: No lines were plotted for {tissue} {metric['name']}!")
                ax.text(0.5, 0.5, "No valid time series data", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=15, style='italic', color='gray')
            
            plot_counter += 1  # Increment plot counter
            axes.append(ax)  # Store the axis
    
    # Create a legend for the entire figure
    handles = []
    
    # Create entries for each condition
    for genotype, g_color in colors.items():
        for treatment, t_style in linestyles.items():
            condition = f"{genotype}-{treatment}"
            marker = markers.get(condition, 'o')
            
            # Create label with descriptive text
            genotype_desc = "Tolerant" if genotype == "G1" else "Susceptible"
            treatment_desc = "Control" if treatment == "T0" else "Stress"
            label = f"{genotype} ({genotype_desc}) - {treatment} ({treatment_desc})"
            
            handles.append(Line2D([0], [0], color=g_color, linestyle=t_style, 
                                marker=marker, markersize=8, linewidth=2.5, label=label))
    
    # Add legend to the top-right of the last plot
    if axes:  # Check if any plots were created
        last_ax = axes[-1]
        last_ax.legend(handles=handles, loc='upper right', fontsize=16, 
                      title='Conditions', title_fontsize=19, 
                      labelcolor='gray',
                      frameon=False)
    
    # Extract key insights from the data
    insights = []
    
    # Calculate Time point 3 differences between G1 and G2 under stress
    for tissue, df in [('Leaf', leaf_data), ('Root', root_data)]:
        for metric in metrics_to_plot:
            if metric['name'] not in df.columns:
                continue
                
            g1t1_tp3 = df[(df['Genotype'] == 'G1') & (df['Treatment'] == 'T1') & 
                         (df['Time point'] == 3)][metric['name']].mean()
            g2t1_tp3 = df[(df['Genotype'] == 'G2') & (df['Treatment'] == 'T1') & 
                         (df['Time point'] == 3)][metric['name']].mean()
            
            if not (np.isnan(g1t1_tp3) or np.isnan(g2t1_tp3) or g2t1_tp3 == 0):
                pct_diff = (g1t1_tp3 - g2t1_tp3) / g2t1_tp3 * 100
                if abs(pct_diff) > 10:  # Only include meaningful differences
                    direction = "higher" if pct_diff > 0 else "lower"
                    metric_short = metric['name'].replace('Attn_', '')
                    insights.append(f"• {tissue} {metric_short} is {abs(pct_diff):.1f}% {direction} in G1 vs G2 under stress (Time point 3)")
    
    # Calculate temporal changes (Time point 1 to Time point 3) for G1 under stress
    for tissue, df in [('Leaf', leaf_data), ('Root', root_data)]:
        for metric in metrics_to_plot:
            if metric['name'] not in df.columns:
                continue
                
            g1t1_tp1 = df[(df['Genotype'] == 'G1') & (df['Treatment'] == 'T1') & 
                         (df['Time point'] == 1)][metric['name']].mean()
            g1t1_tp3_temporal = df[(df['Genotype'] == 'G1') & (df['Treatment'] == 'T1') & 
                                  (df['Time point'] == 3)][metric['name']].mean()
            
            if not (np.isnan(g1t1_tp1) or np.isnan(g1t1_tp3_temporal) or g1t1_tp1 == 0):
                pct_diff = (g1t1_tp3_temporal - g1t1_tp1) / g1t1_tp1 * 100
                if abs(pct_diff) > 20:  # Only include substantial temporal changes
                    direction = "increases" if pct_diff > 0 else "decreases"
                    metric_short = metric['name'].replace('Attn_', '')
                    insights.append(f"• G1 {tissue} {metric_short} {direction} by {abs(pct_diff):.1f}% from Time point 1-3 under stress")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.92])
    
    fig.patch.set_facecolor(COLORS.get('Background', 'white'))
    
    # Save output
    output_path = os.path.join(output_dir, "fig4_view_level_attention.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {output_path}")


def main():
    print("Creating Figure 4: Integrated View-Level Attention Statistics")
    create_view_level_attention_figure()
    print("Figure 4 generation complete!")


if __name__ == "__main__":
    main()