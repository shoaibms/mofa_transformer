#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Wavelength Analysis for Plant Stress Response
==================================================

This script organizes and visualizes spectral-metabolite interactions
across different wavelengths. It creates multi-wavelength comparison
figures by working with existing plot files to present a cohesive
analysis of plant stress responses.

The script performs the following tasks:
1. Locates existing visualization files in the specified directory
2. Extracts and categorizes wavelength information from filenames
3. Creates multi-panel figures grouped by tissue and wavelength
4. Generates summary visualizations of attention patterns
"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import shutil
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
# Base directory and output directory
BASE_DIR = r"C:/Users/ms/Desktop/hyper/output/transformer/v3_feature_attention"
OUTPUT_DIR = r"C:/Users/ms/Desktop/hyper/output/transformer/v3_feature_attention/plots_attention_advanced"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define physiologically relevant wavelength categories
WAVELENGTH_CATEGORIES = {
    'Water Bands': [1450, 1940, 970, 1200],           # Water absorption bands
    'Pigment Bands': [550, 660, 680, 700, 710],       # Chlorophyll and other pigments
    'NIR Structure': [850, 900, 1000, 1100, 1300],    # Cell structure
    'SWIR Features': [1600, 1700, 2000, 2200, 2300]   # Biochemical features
}

# ===== HELPER FUNCTIONS =====

def find_existing_plots(base_dir, pattern="*.png"):
    """Find all existing plot files in the directory tree.
    
    Args:
        base_dir: The root directory to search
        pattern: File pattern to match (default: "*.png")
        
    Returns:
        List of file paths
    """
    all_plots = []
    for root, dirs, files in os.walk(base_dir):
        for file in glob.glob(os.path.join(root, pattern)):
            all_plots.append(file)
    return all_plots

def extract_wavelength_from_filename(filename):
    """Extract wavelength information from a filename.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Integer wavelength value or None if not found
    """
    # Patterns to look for
    patterns = [
        r'W_(\d+)',                          # W_550
        r'(\d+)nm',                          # 550nm
        r'wavelength[_-](\d+)',              # wavelength_550
        r'wl[_-](\d+)'                       # wl_550
    ]
    
    for pattern in patterns:
        match = re.search(pattern, os.path.basename(filename), re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    
    return None

def categorize_wavelength(wavelength):
    """Categorize a wavelength into a physiological category.
    
    Args:
        wavelength: Wavelength value in nm
        
    Returns:
        String category name
    """
    if wavelength is None:
        return "Unknown"
        
    for category, wave_list in WAVELENGTH_CATEGORIES.items():
        # Find the closest wavelength in the category
        closest = min(wave_list, key=lambda x: abs(x - wavelength))
        if abs(closest - wavelength) <= 50:  # Within 50nm is close enough
            return category
    
    return "Other"

def extract_tissue_from_filename(filename):
    """Extract tissue information from a filename.
    
    Args:
        filename: The filename to parse
        
    Returns:
        String tissue name or None if not found
    """
    basename = os.path.basename(filename).lower()
    if "leaf" in basename:
        return "Leaf"
    elif "root" in basename:
        return "Root"
    else:
        return None

def find_comprehensive_temporal_plots(all_plots):
    """Find the comprehensive temporal plots with 4-panel visualizations.
    
    Args:
        all_plots: List of plot file paths
        
    Returns:
        List of dictionaries with plot metadata
    """
    # Look for plots that match the pattern of comprehensive analyses
    comp_plots = []
    
    patterns = [
        r'comprehensive.*analysis',
        r'temporal.*analysis',
        r'fig.*(\d+).*temporal', 
        r'fig.*(\d+).*attention',
        r'attention.*temporal',
        r'metabolome.*temporal'
    ]
    
    for plot in all_plots:
        basename = os.path.basename(plot).lower()
        for pattern in patterns:
            if re.search(pattern, basename):
                # Extract wavelength if possible
                wavelength = extract_wavelength_from_filename(plot)
                tissue = extract_tissue_from_filename(plot)
                
                if wavelength or tissue:
                    comp_plots.append({
                        'path': plot,
                        'wavelength': wavelength,
                        'wavelength_category': categorize_wavelength(wavelength),
                        'tissue': tissue
                    })
                    break
    
    return comp_plots

def find_attention_plots(all_plots):
    """Find plots related to attention between features.
    
    Args:
        all_plots: List of plot file paths
        
    Returns:
        List of dictionaries with plot metadata
    """
    attention_plots = []
    
    patterns = [
        r'attention.*heatmap',
        r'attention.*network',
        r'cross.*modal',
        r'feature.*attention',
        r'temporal.*profile'
    ]
    
    for plot in all_plots:
        basename = os.path.basename(plot).lower()
        for pattern in patterns:
            if re.search(pattern, basename):
                # Extract wavelength if possible
                wavelength = extract_wavelength_from_filename(plot)
                tissue = extract_tissue_from_filename(plot)
                
                attention_plots.append({
                    'path': plot,
                    'wavelength': wavelength,
                    'wavelength_category': categorize_wavelength(wavelength),
                    'tissue': tissue
                })
                break
    
    return attention_plots

def create_multi_wavelength_figure(comp_plots, output_dir):
    """Create a multi-panel figure showing multiple wavelengths and tissues.
    
    Args:
        comp_plots: List of dictionaries with plot metadata
        output_dir: Directory to save output figures
    """
    # Group plots by tissue
    tissue_plots = defaultdict(list)
    for plot in comp_plots:
        if plot['tissue']:
            tissue_plots[plot['tissue']].append(plot)
    
    # Sort each tissue's plots by wavelength category for biological relevance
    category_order = [
        'Water Bands', 'Pigment Bands', 'NIR Structure', 
        'SWIR Features', 'Other', 'Unknown'
    ]
    
    for tissue, plots in tissue_plots.items():
        # Sort plots by category order
        plots.sort(key=lambda x: (
            category_order.index(x['wavelength_category']) 
            if x['wavelength_category'] in category_order else 999,
            x['wavelength'] if x['wavelength'] else 9999
        ))
        
        # Take up to 3 most relevant plots for each tissue
        selected_plots = plots[:3]
        
        if len(selected_plots) == 0:
            continue
            
        try:
            # Create a multi-panel figure
            fig = plt.figure(figsize=(16, 10 * len(selected_plots)))
            gs = GridSpec(len(selected_plots), 1, figure=fig)
            
            # Add each plot as a panel
            for i, plot_info in enumerate(selected_plots):
                # Load the image
                img = mpimg.imread(plot_info['path'])
                
                # Create subplot
                ax = fig.add_subplot(gs[i, 0])
                
                # Display the image
                ax.imshow(img)
                ax.axis('off')
                
                # Add panel label
                wavelength_str = (f"W_{plot_info['wavelength']}" 
                                 if plot_info['wavelength'] else "Unknown")
                category_str = (f" ({plot_info['wavelength_category']})" 
                               if plot_info['wavelength_category'] != "Unknown" else "")
                ax.set_title(f"{chr(65+i)}) {wavelength_str}{category_str}", 
                             fontsize=14, loc='left')
            
            # Add figure title
            fig.suptitle(
                f"Figure 12: Multi-Wavelength Temporal Dynamics in {tissue} Tissue", 
                fontsize=16, y=0.99
            )
            
            # Add caption
            caption = (
                f"Figure 12. Comprehensive analysis of wavelength-metabolite "
                f"interactions across time in {tissue} tissue. "
                "Each panel shows a different physiologically relevant wavelength "
                "with four sub-panels: "
                "(A) Heatmaps comparing temporal attention patterns between genotypes, "
                "(B) Differential attention heatmap (G1-G2) highlighting metabolites "
                "with genotype-specific temporal patterns, "
                "(C) Temporal profiles for key metabolites, showing coordination "
                "patterns over time, "
                "(D) Day-specific comparisons revealing the evolution of "
                "differential attention. "
                "This multi-wavelength perspective reveals both conserved response "
                "mechanisms across spectral regions "
                "and wavelength-specific interactions associated with different "
                "physiological processes."
            )
            
            # Add the caption at the bottom
            fig.text(0.5, 0.01, caption, wrap=True, 
                     horizontalalignment='center', fontsize=12)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            
            # Save figure
            output_path = os.path.join(
                output_dir, f"Figure12_Multi_Wavelength_Temporal_{tissue}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created multi-wavelength figure for {tissue} tissue: {output_path}")
            
            # Also copy the individual plots to the output directory for reference
            for plot_info in selected_plots:
                basename = os.path.basename(plot_info['path'])
                shutil.copy2(plot_info['path'], os.path.join(output_dir, basename))
                print(f"Copied {basename} to output directory")
                
        except Exception as e:
            print(f"Error creating multi-wavelength figure for {tissue}: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

def create_attention_summary_figure(attention_plots, output_dir):
    """Create a summary figure of key attention visualizations.
    
    Args:
        attention_plots: List of dictionaries with plot metadata
        output_dir: Directory to save output figures
    """
    # Filter for the most informative attention plots
    selected_plots = []
    
    # Selection criteria - prioritize certain types
    priority_patterns = [
        (r'network.*comparison', 'Network Comparison'),
        (r'attention.*heatmap', 'Attention Heatmap'),
        (r'temporal.*profile', 'Temporal Profile'),
        (r'cross.*modal', 'Cross-Modal Analysis')
    ]
    
    # Match plots to priority patterns
    plot_matches = []
    for plot in attention_plots:
        basename = os.path.basename(plot['path']).lower()
        for pattern, label in priority_patterns:
            if re.search(pattern, basename):
                plot_matches.append(
                    (plot, label, priority_patterns.index((pattern, label)))
                )
                break
    
    # Sort by priority
    plot_matches.sort(key=lambda x: x[2])
    
    # Take up to 4 plots, prioritizing different types and tissues
    selected_types = set()
    selected_tissues = set()
    
    for plot, label, _ in plot_matches:
        # Ensure diversity by selecting different types and tissues
        if len(selected_plots) >= 4:
            break
            
        if label not in selected_types or plot['tissue'] not in selected_tissues:
            selected_plots.append((plot, label))
            selected_types.add(label)
            if plot['tissue']:
                selected_tissues.add(plot['tissue'])
    
    if len(selected_plots) == 0:
        print("No suitable attention plots found for summary figure")
        return
        
    try:
        # Create a 2x2 grid figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        # Add each plot to a panel
        for i, (plot_info, label) in enumerate(selected_plots):
            if i >= len(axes):
                break
                
            # Load the image
            img = mpimg.imread(plot_info['path'])
            
            # Display the image
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Add panel label
            tissue_str = f" ({plot_info['tissue']})" if plot_info['tissue'] else ""
            axes[i].set_title(f"{chr(65+i)}) {label}{tissue_str}", 
                              fontsize=14, loc='left')
        
        # Hide any unused axes
        for i in range(len(selected_plots), len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
        
        # Add figure title
        fig.suptitle("Summary of Cross-Modal Attention Analyses", 
                     fontsize=16, y=0.98)
        
        # Add caption (generic)
        caption = (
            "Summary of key visualizations from the cross-modal attention analysis. "
            "These visualizations highlight the interaction between spectral and "
            "metabolite features across different tissues, genotypes, and time points, "
            "revealing the underlying mechanisms of plant stress response coordination."
        )
        
        # Add the caption at the bottom
        fig.text(0.5, 0.02, caption, wrap=True, 
                 horizontalalignment='center', fontsize=12)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = os.path.join(output_dir, "Attention_Analysis_Summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created attention summary figure: {output_path}")
        
    except Exception as e:
        print(f"Error creating attention summary figure: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

# ===== MAIN FUNCTION =====

def main():
    """Main execution function."""
    print("Starting improved multi-wavelength analysis...")
    
    # Find all existing plot files
    all_plots = find_existing_plots(BASE_DIR)
    print(f"Found {len(all_plots)} plot files in the directory tree")
    
    # Find plots that have comprehensive temporal analysis
    comp_plots = find_comprehensive_temporal_plots(all_plots)
    print(f"Found {len(comp_plots)} comprehensive temporal analysis plots")
    
    # Print what we found for debugging
    for i, plot in enumerate(comp_plots[:10]):  # Print first 10 for brevity
        print(f"  {i+1}. {os.path.basename(plot['path'])}")
        print(f"     Wavelength: {plot['wavelength']}, Category: "
              f"{plot['wavelength_category']}, Tissue: {plot['tissue']}")
    
    # Find attention-related plots
    attention_plots = find_attention_plots(all_plots)
    print(f"Found {len(attention_plots)} attention-related plots")
    
    # Create multi-wavelength figure
    if comp_plots:
        create_multi_wavelength_figure(comp_plots, OUTPUT_DIR)
    else:
        print("No comprehensive temporal plots found to create multi-wavelength figure")
    
    # Create attention summary figure
    if attention_plots:
        create_attention_summary_figure(attention_plots, OUTPUT_DIR)
    else:
        print("No attention plots found to create summary figure")
    
    print("\nImproved multi-wavelength analysis completed!")
    
    # If we didn't find any plots, also check for and copy specific plots
    example_filenames = ["W_550", "W_1047", "1450nm", "660nm", "550nm", "970nm"]
    found_examples = []
    
    for example in example_filenames:
        for plot in all_plots:
            if example in os.path.basename(plot):
                found_examples.append(plot)
                dest_path = os.path.join(OUTPUT_DIR, os.path.basename(plot))
                try:
                    shutil.copy2(plot, dest_path)
                    print(f"Copied example plot: {os.path.basename(plot)}")
                except Exception as e:
                    print(f"Error copying {plot}: {e}")
    
    if not found_examples:
        print("\nNo example plots found to copy directly.")
        
        # Create a placeholder to explain how to use the existing plots
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5, 0.5, 
                "No existing plots could be found for multi-wavelength analysis.\n\n"
                "To create Figure 12, combine the plots you already have into a "
                "multi-panel figure. These plots already contain the 4-panel layout "
                "with:\n"
                "- Genotype-specific attention heatmaps\n"
                "- Differential (G1-G2) heatmaps\n"
                "- Temporal profiles\n"
                "- G1 vs G2 scatter comparisons",
                ha='center', va='center', fontsize=14, wrap=True
            )
            ax.axis('off')
            plt.savefig(
                os.path.join(OUTPUT_DIR, "Figure12_Instructions.png"), 
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        except Exception as e:
            print(f"Error creating instruction image: {e}")

if __name__ == "__main__":
    main()