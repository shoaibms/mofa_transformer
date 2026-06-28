#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master figure generator for comparing MOFA+ and SHAP features.

This script generates a 10-panel figure (a-j) that provides a comprehensive
comparison between features identified by MOFA+ and SHAP. The figure includes
analyses of feature overlap, composition, spectral characteristics, and
importance correlation.

The script is designed to be run from the command line and will save the
final figure as both PNG and SVG files in the specified output directory.

Usage:
    python Figure_6.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle
from datetime import datetime
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# --- Configuration ---

# Style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')

# Color and Font Palettes
COLORS = {
    # Panels a, b, g, h
    'Overlap': "#CFF198",
    'MOFA_Only': "#A6E9C4",
    'SHAP_Only': "#98B59B",

    # Panel c
    'MOFA_Spectral': "#AADDAD",
    'MOFA_Metabolite': "#7ECE7E",
    'SHAP_Spectral': '#87CEEB',
    'SHAP_Metabolite': "#99F387",

    # Panels d, e
    'G1_Tolerant': '#00FA9A',
    'G2_Susceptible': '#48D1CC',
    'Spectra_Highlight': 'yellow',
    'Difference_Line': '#555555',

    # Panel f
    'MOFA_Importance': "#99F05F",
    'SHAP_Importance': "#0FAFC8",

    # Panels i, j
    'Leaf_Spectral': '#00FF7F',
    'Leaf_Molecular': '#9ACD32',
    'Root_Spectral': '#40E0D0',
    'Root_Molecular': '#20B2AA',

    # General Elements
    'Edge_Color': '#252525',
    'Text_Dark': '#252525',
}


FONTS_SANS = {
    'family': 'sans-serif', 'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'main_title': 20, 'panel_label': 22, 'panel_title': 18, 'axis_label': 16,
    'tick_label': 14, 'legend_title': 16, 'legend_text': 14, 'annotation': 14,
}

# Apply global font settings
mpl.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'font.size': FONTS_SANS['tick_label'], 'axes.labelsize': FONTS_SANS['axis_label'],
    'axes.titlesize': FONTS_SANS['panel_title'], 'xtick.labelsize': FONTS_SANS['tick_label'],
    'ytick.labelsize': FONTS_SANS['tick_label'], 'legend.fontsize': FONTS_SANS['legend_text'],
    'legend.title_fontsize': FONTS_SANS['legend_title'], 'figure.titlesize': FONTS_SANS['main_title']
})

# --- Paths and Data ---
BASE_DIR = r"C:\Users\ms\Desktop\hyper"
DATA_DIR = os.path.join(BASE_DIR, "data")
MOFA_DIR = os.path.join(BASE_DIR, "output", "mofa")
SHAP_DIR = os.path.join(BASE_DIR, "output", "transformer", "shap_analysis_ggl", "importance_data")
CONTRACT_PATH = os.path.join(BASE_DIR, "output", "robustness", "robustness_contract.json")
ASSOC_PATH = os.path.join(MOFA_DIR, "mofa_factor_metadata_associations_spearman.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "figure")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_SUFFIXES = ['_leaf_spectral', '_root_spectral', '_leaf_metabolite', '_root_metabolite']
TASKS = ['Genotype', 'Treatment', 'Day']
TISSUES = ['Leaf', 'Root']


# --- Data Loading and Helper Functions ---

def clean_feature_name(name):
    """Remove tissue and data type suffixes from a feature name."""
    for s in FEATURE_SUFFIXES:
        name = name.replace(s, '')
    return name

def load_correlation_and_overlap_data():
    """Load and merge MOFA+ and SHAP data for correlation analysis."""
    results = {}
    for tissue in ['leaf', 'root']:
        mofa_s = pd.read_csv(os.path.join(MOFA_DIR, f"mofa_feature_weights_{tissue}_spectral_active.csv"), index_col=0)
        mofa_m = pd.read_csv(os.path.join(MOFA_DIR, f"mofa_feature_weights_{tissue}_metabolite_active.csv"), index_col=0)
        shap_data = pd.read_csv(os.path.join(SHAP_DIR, f"shap_importance_{tissue.title()}_Genotype.csv"))

        mofa_s_clean = pd.DataFrame({'Feature': [clean_feature_name(f) for f in mofa_s.index], 'MOFA_Weight': mofa_s['Factor9'].abs()})
        mofa_m_clean = pd.DataFrame({'Feature': [clean_feature_name(f) for f in mofa_m.index], 'MOFA_Weight': mofa_m['Factor9'].abs()})
        mofa_combined = pd.concat([mofa_s_clean, mofa_m_clean])

        merged = pd.merge(shap_data, mofa_combined, on='Feature', how='inner')
        results[tissue] = {'mofa_combined': mofa_combined, 'shap_data': shap_data, 'merged': merged}
    return results

def load_spectral_data_with_stats(data_dir):
    """Load hyperspectral data and compute per-genotype reflectance statistics."""
    raw_data = pd.read_csv(os.path.join(data_dir, "hyper_full_w.csv"))
    s_cols = [c for c in raw_data if c.startswith('W_')]
    res = pd.DataFrame({'wavelength': [int(c.split('_')[1]) for c in s_cols], 'feature': s_cols})
    g1 = raw_data[raw_data['Genotype'] == 'G1']
    g2 = raw_data[raw_data['Genotype'] == 'G2']
    res['G1_mean'] = g1[s_cols].mean().values
    res['G2_mean'] = g2[s_cols].mean().values
    res['G1_std'] = g1[s_cols].std().values
    res['G2_std'] = g2[s_cols].std().values
    res['diff'] = res['G1_mean'] - res['G2_mean']
    res['percent_diff'] = 100 * (res['diff'] / ((res['G1_mean'] + res['G2_mean']) / 2))
    return res

def load_robustness_contract():
    """Load robustness_contract.json — the authoritative source for Fig. 6 overlap data."""
    with open(CONTRACT_PATH, 'r') as f:
        return json.load(f)

def get_task_factor_map():
    """Map each task to its strongest-associated MOFA factor (by |Spearman R|)."""
    df = pd.read_csv(ASSOC_PATH)
    mapping = {}
    for task in TASKS:
        rows = df[df['Metadata'] == task].copy()
        if rows.empty:
            continue
        rows['abs_R'] = rows['Correlation'].abs()
        mapping[task] = rows.loc[rows['abs_R'].idxmax(), 'Factor']
    return mapping

def _load_mofa_spectral(tissue):
    """Load and clean MOFA spectral weights for one tissue."""
    path = os.path.join(MOFA_DIR, f"mofa_feature_weights_{tissue.lower()}_spectral_active.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index().rename(columns={df.index.name or 'index': 'Feature'})
    df.columns = ['Feature'] + list(df.columns[1:])
    df['Feature'] = df['Feature'].apply(clean_feature_name)
    return df

def compute_task_metrics(threshold_pct=0.05):
    """Compute top-N MOFA vs SHAP overlap stats per (tissue, task) using contract methodology.

    MOFA pool: {tissue}_spectral weights, ranked by |loading| of the task-associated factor.
    SHAP pool: full {Tissue}_{Task} importance file.
    Top-N is `threshold_pct` of each pool.
    """
    task_factors = get_task_factor_map()
    metrics = []
    for tissue in TISSUES:
        for task in TASKS:
            factor = task_factors.get(task)
            if factor is None:
                continue
            mofa_df = _load_mofa_spectral(tissue)
            mofa_df['abs_weight'] = mofa_df[factor].abs()
            shap_path = os.path.join(SHAP_DIR, f"shap_importance_{tissue}_{task}.csv")
            shap_df = pd.read_csv(shap_path)

            n_mofa = max(1, int(len(mofa_df) * threshold_pct))
            n_shap = max(1, int(len(shap_df) * threshold_pct))
            top_mofa_df = mofa_df.nlargest(n_mofa, 'abs_weight')
            top_shap_df = shap_df.nlargest(n_shap, 'MeanAbsoluteShap')
            top_mofa = set(top_mofa_df['Feature'])
            top_shap = set(top_shap_df['Feature'])
            overlap = top_mofa & top_shap
            union = top_mofa | top_shap
            jaccard = len(overlap) / len(union) if union else 0.0

            mofa_spec = sum(1 for f in top_mofa if str(f).startswith('W_'))
            mofa_meta = n_mofa - mofa_spec
            shap_types = top_shap_df['FeatureType'].astype(str).str.lower()
            shap_spec = int((shap_types == 'spectral').sum())
            shap_meta = int((shap_types == 'metabolite').sum())

            metrics.append({
                'name': f"{tissue}-{task}",
                'tissue': tissue, 'task': task, 'factor': factor,
                'mofa': n_mofa, 'shap': n_shap, 'overlap': len(overlap),
                'jaccard': jaccard,
                'mofa_spec_pct': 100 * mofa_spec / n_mofa if n_mofa else 0,
                'mofa_meta_pct': 100 * mofa_meta / n_mofa if n_mofa else 0,
                'shap_spec_pct': 100 * shap_spec / n_shap if n_shap else 0,
                'shap_meta_pct': 100 * shap_meta / n_shap if n_shap else 0,
            })
    return metrics

def load_overlap_feature_weights(overlap_features, factor):
    """Load real MOFA+ Factor loadings and SHAP importances for the contract overlap features."""
    mofa_df = _load_mofa_spectral('Leaf').set_index('Feature')
    shap_df = pd.read_csv(os.path.join(SHAP_DIR, "shap_importance_Leaf_Genotype.csv"))
    shap_lookup = shap_df.set_index('Feature')['MeanAbsoluteShap']
    mofa_w = pd.Series({f: mofa_df.loc[f, factor] if f in mofa_df.index else np.nan for f in overlap_features})
    shap_v = pd.Series({f: shap_lookup.get(f, np.nan) for f in overlap_features})
    return mofa_w, shap_v

# --- Panel Plotting Functions ---

def plot_panel_A(ax, metrics):
    """Plot Jaccard index of feature overlap (Panel a)."""
    df = pd.DataFrame(metrics)
    bars = ax.bar(df['name'], df['jaccard'], color=[COLORS['Overlap'] if v > 0 else '#DDDDDD' for v in df['jaccard']])
    ax.set_ylabel('Jaccard Index')
    ax.set_title('Feature Overlap (Top 5%)', loc='center', fontweight='bold')
    ax.text(-0.15, 1.18, 'A', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    ymax = max(0.1, df['jaccard'].max() * 1.4)
    ax.set_ylim(0, ymax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for bar, val in zip(bars, df['jaccard']):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + ymax*0.02,
                    f'{val:.4f}', ha='center')

def plot_panel_B(ax, metrics):
    """Plot feature set composition (MOFA+ only, SHAP only, Overlap) (Panel b)."""
    df = pd.DataFrame(metrics)
    df['mofa_only'] = df['mofa'] - df['overlap']
    df['shap_only'] = df['shap'] - df['overlap']
    ax.bar(df['name'], df['mofa_only'], label='MOFA+ Only', color=COLORS['MOFA_Only'], edgecolor=COLORS['Edge_Color'])
    ax.bar(df['name'], df['shap_only'], bottom=df['mofa_only'], label='SHAP Only', color=COLORS['SHAP_Only'], edgecolor=COLORS['Edge_Color'])
    ax.bar(df['name'], df['overlap'], bottom=df['mofa_only'] + df['shap_only'], label='Overlap', color=COLORS['Overlap'], edgecolor=COLORS['Edge_Color'])
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Set Composition (Top 5%)', loc='center', fontweight='bold')
    ax.text(-0.15, 1.18, 'B', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    ax.legend(loc='upper right')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def plot_panel_C(ax, metrics):
    """Plot feature type distribution for each method and task (Panel c).

    MOFA pool is spectral-only by contract methodology, so MOFA bars are 100% spectral.
    SHAP feature types come from the FeatureType column of each importance file.
    """
    df = pd.DataFrame(metrics)
    tasks = df['name'].tolist()
    x = np.arange(len(tasks))
    width = 0.4
    ax.bar(x-width/2, df['mofa_spec_pct'], width, label='MOFA+ Spectral', color=COLORS['MOFA_Spectral'], edgecolor=COLORS['Edge_Color'])
    ax.bar(x-width/2, df['mofa_meta_pct'], width, bottom=df['mofa_spec_pct'], label='MOFA+ Metabolite', color=COLORS['MOFA_Metabolite'], edgecolor=COLORS['Edge_Color'])
    ax.bar(x+width/2, df['shap_spec_pct'], width, label='SHAP Spectral', color=COLORS['SHAP_Spectral'], edgecolor=COLORS['Edge_Color'])
    ax.bar(x+width/2, df['shap_meta_pct'], width, bottom=df['shap_spec_pct'], label='SHAP Metabolite', color=COLORS['SHAP_Metabolite'], edgecolor=COLORS['Edge_Color'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Feature Type Distribution', loc='center', fontweight='bold')
    ax.text(-0.15, 1.18, 'C', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc='best', ncol=2)

def plot_panel_D(ax, data):
    """Plot full reflectance spectra for G1 and G2 genotypes (Panel d)."""
    ax.plot(data['wavelength'], data['G1_mean'], color=COLORS['G1_Tolerant'], label='G1 (Tolerant)')
    ax.fill_between(data['wavelength'], data['G1_mean']-data['G1_std'], data['G1_mean']+data['G1_std'], color=COLORS['G1_Tolerant'], alpha=0.3)
    ax.plot(data['wavelength'], data['G2_mean'], color=COLORS['G2_Susceptible'], label='G2 (Susceptible)')
    ax.fill_between(data['wavelength'], data['G2_mean']-data['G2_std'], data['G2_mean']+data['G2_std'], color=COLORS['G2_Susceptible'], alpha=0.3)
    y_min, y_max = ax.get_ylim()
    ax.add_patch(Rectangle((546, y_min), 635-546, y_max-y_min, facecolor=COLORS['Spectra_Highlight'], alpha=0.3, zorder=0))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title('Full Reflectance Spectra', loc='center', fontweight='bold')
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, facecolor=COLORS['Spectra_Highlight'], alpha=0.3, edgecolor='none'))
    labels.append('Context window (546-635 nm)')
    ax.legend(handles, labels, loc='upper right')

def plot_panel_E(ax, data):
    """Plot detailed view of reflectance spectra (546-635nm) (Panel e)."""
    roi = data[(data['wavelength'] >= 536) & (data['wavelength'] <= 645)].copy()
    ax.plot(roi['wavelength'], roi['G1_mean'], color=COLORS['G1_Tolerant'], label='G1 Reflectance')
    ax.fill_between(roi['wavelength'], roi['G1_mean']-roi['G1_std'], roi['G1_mean']+roi['G1_std'], color=COLORS['G1_Tolerant'], alpha=0.3)
    ax.plot(roi['wavelength'], roi['G2_mean'], color=COLORS['G2_Susceptible'], label='G2 Reflectance')
    ax.fill_between(roi['wavelength'], roi['G2_mean']-roi['G2_std'], roi['G2_mean']+roi['G2_std'], color=COLORS['G2_Susceptible'], alpha=0.3)
    y_min, y_max = ax.get_ylim()
    ax.add_patch(Rectangle((546, y_min), 635-546, y_max-y_min, facecolor=COLORS['Spectra_Highlight'], alpha=0.3, zorder=0))
    ax2 = ax.twinx()
    ax2.plot(roi['wavelength'], roi['percent_diff'], color=COLORS['Difference_Line'], linestyle='--', label='G1-G2 Diff (%)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax2.set_ylabel('Percent Difference (%)', color=COLORS['Difference_Line'])
    ax.set_title('Detailed View (546-635nm)', loc='center', fontweight='bold')
    ax.text(-0.1, 1.1, 'E', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    l1.append(Rectangle((0, 0), 1, 1, facecolor=COLORS['Spectra_Highlight'], alpha=0.3, edgecolor='none'))
    la1.append('Context window (546-635 nm)')
    ax.legend(l1+l2, la1+la2, loc='upper left')

def plot_panel_F(ax, overlap_features, mofa_w, shap_v):
    """Plot real MOFA+ |loading| and SHAP value for the contract overlap features (Panel f)."""
    features = sorted(overlap_features, key=lambda x: int(x.split('_')[1]))
    mofa_abs = mofa_w.reindex(features).abs()
    shap_abs = shap_v.reindex(features).abs()
    mofa_norm = mofa_abs / mofa_abs.max() if mofa_abs.max() else mofa_abs
    shap_norm = shap_abs / shap_abs.max() if shap_abs.max() else shap_abs
    df = pd.DataFrame({'Wavelength': [int(f.split('_')[1]) for f in features],
                       'MOFA+ |Weight|': mofa_norm.values,
                       'SHAP Value': shap_norm.values}).set_index('Wavelength')
    df.plot(kind='bar', ax=ax, color=[COLORS['MOFA_Importance'], COLORS['SHAP_Importance']], width=0.8)
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Overlapping Feature Importance (Top 5%)', loc='center', fontweight='bold')
    ax.text(-0.1, 1.1, 'F', transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    step = max(1, len(df.index) // 10)
    ax.set_xticks(range(0, len(df.index), step))
    ax.set_xticklabels(df.index[::step], rotation=45, ha='right')

def plot_correlation_panel(ax, data, tissue, panel_label):
    """Plot correlation between MOFA+ weights and SHAP importance (Panels g, h)."""
    if data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return
    r, p = pearsonr(data['MOFA_Weight'], data['MeanAbsoluteShap'])
    ax.scatter(data['MOFA_Weight'], data['MeanAbsoluteShap'], alpha=0.9, s=50, color=COLORS['Overlap'], edgecolor=COLORS['Edge_Color'], linewidth=0.7)
    z = np.polyfit(data['MOFA_Weight'], data['MeanAbsoluteShap'], 1)
    ax.plot(data['MOFA_Weight'], np.poly1d(z)(data['MOFA_Weight']), "--", color=COLORS['Text_Dark'], linewidth=2)
    ax.set_xlabel('MOFA+ LF9 Abs. Weight')
    ax.set_ylabel('SHAP Mean Abs. Importance')
    title = f'{tissue.title()}: r={r:.2f}, p={p:.1e}'
    ax.set_title(title, loc='center', fontweight='bold')
    ax.text(-0.15, 1.18, panel_label.upper(), transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')

    if panel_label.upper() == 'G':
        ax.set_ylim(top=0.4)
    elif panel_label.upper() == 'H':
        ax.set_ylim(top=0.15)


def plot_overlap_bars_panels(ax_I, ax_J, data, top_n=100):
    """Plot feature overlap as counts and percentages (Panels i, j)."""
    if not data:
        ax_I.text(0.5, 0.5, 'No Data')
        ax_J.text(0.5, 0.5, 'No Data')
        return

    def count_feature_types(features):
        """Count spectral (W_) vs. molecular (N_, P_, F) features."""
        spectral = sum(1 for f in features if 'W_' in f)
        molecular = sum(1 for f in features if 'N_' in f or 'P_' in f or 'F' in f)
        return spectral, molecular

    d = []
    for t, td in data.items():
        mofa = set(td['mofa_combined'].nlargest(top_n, 'MOFA_Weight')['Feature'])
        shap = set(td['shap_data'].nlargest(top_n, 'MeanAbsoluteShap')['Feature'])
        mofa_only_s, mofa_only_m = count_feature_types(mofa - shap)
        both_s, both_m = count_feature_types(mofa & shap)
        shap_only_s, shap_only_m = count_feature_types(shap - mofa)
        d.append({
            'tissue': t, 'mofa_s': mofa_only_s, 'mofa_m': mofa_only_m,
            'both_s': both_s, 'both_m': both_m,
            'shap_s': shap_only_s, 'shap_m': shap_only_m
        })

    leaf = next((i for i in d if i["tissue"] == "leaf"), None)
    root = next((i for i in d if i["tissue"] == "root"), None)
    methods = ['MOFA+\nOnly', 'Both', 'SHAP\nOnly']
    x = np.arange(len(methods))
    width = 0.4

    for ax, mode in [(ax_I, 'count'), (ax_J, 'percent')]:
        ax.cla()
        if leaf:
            s = [leaf['mofa_s'], leaf['both_s'], leaf['shap_s']]
            m = [leaf['mofa_m'], leaf['both_m'], leaf['shap_m']]
            if mode == 'percent':
                totals = [a+b for a, b in zip(s, m)]
                s = [i/j*100 if j > 0 else 0 for i, j in zip(s, totals)]
                m = [i/j*100 if j > 0 else 0 for i, j in zip(m, totals)]
            ax.bar(x - width/2, s, width, label='Leaf Spectral', color=COLORS['Leaf_Spectral'], edgecolor=COLORS['Edge_Color'])
            ax.bar(x - width/2, m, width, bottom=s, label='Leaf Molecular', color=COLORS['Leaf_Molecular'], edgecolor=COLORS['Edge_Color'])
        if root:
            s = [root['mofa_s'], root['both_s'], root['shap_s']]
            m = [root['mofa_m'], root['both_m'], root['shap_m']]
            if mode == 'percent':
                totals = [a+b for a, b in zip(s, m)]
                s = [i/j*100 if j > 0 else 0 for i, j in zip(s, totals)]
                m = [i/j*100 if j > 0 else 0 for i, j in zip(m, totals)]
            ax.bar(x + width/2, s, width, label='Root Spectral', color=COLORS['Root_Spectral'], edgecolor=COLORS['Edge_Color'])
            ax.bar(x + width/2, m, width, bottom=s, label='Root Molecular', color=COLORS['Root_Molecular'], edgecolor=COLORS['Edge_Color'])
        ax.set_xticks(x)
        ax.set_xticklabels(methods)

    ax_I.set_ylabel('Number of Features')
    ax_I.set_title('Top 100 Feature Counts', loc='center', fontweight='bold')
    ax_I.text(-0.15, 1.18, 'I', transform=ax_I.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')
    ax_I.legend(title="Feature Type")

    ax_J.set_ylim(0, 100)
    ax_J.set_ylabel('Percentage of Features (%)')
    ax_J.set_title('Top 100 Feature Distribution', loc='center', fontweight='bold')
    ax_J.text(-0.15, 1.18, 'J', transform=ax_J.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')


# --- Main Orchestration ---

def create_master_figure():
    """Load data, create figure layout, plot all panels, and save the output."""
    print("Generating master figure...")

    # Authoritative source for Fig. 6 overlap data
    contract = load_robustness_contract()
    overlap_features = contract['primary_overlap']['feature_list']
    selected_factor = contract['selected_factor']['name']
    threshold_pct = contract['primary_overlap']['threshold_used']
    print(f"Loaded contract: factor={selected_factor}, threshold={threshold_pct*100:.1f}%, "
          f"overlap={contract['primary_overlap']['count']} features")

    # Per-task overlap metrics (matches contract methodology for Leaf-Genotype)
    task_metrics = compute_task_metrics(threshold_pct=threshold_pct)

    # Pin Leaf-Genotype to contract values — the contract is the authoritative source for it
    primary_jaccard = next((s['jaccard'] for s in contract['robustness_sweep']
                            if abs(s['threshold_pct'] - threshold_pct) < 1e-9), None)
    for m in task_metrics:
        if m['name'] == 'Leaf-Genotype':
            m['overlap'] = contract['primary_overlap']['count']
            if primary_jaccard is not None:
                m['jaccard'] = primary_jaccard

    # Real MOFA+ loadings and SHAP values for the contract's overlap features
    mofa_w, shap_v = load_overlap_feature_weights(overlap_features, selected_factor)

    correlation_data = load_correlation_and_overlap_data()
    spectral_data = load_spectral_data_with_stats(DATA_DIR)

    # Create Figure Layout
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(5, 2, figure=fig, hspace=0.6, wspace=0.25,
                  height_ratios=[1, 1.5, 1.5, 1.2, 1.2])

    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.3, width_ratios=[0.8, 1, 1.2])
    axA = fig.add_subplot(gs_top[0])
    axB = fig.add_subplot(gs_top[1])
    axC = fig.add_subplot(gs_top[2])

    axD = fig.add_subplot(gs[1, :])

    gs_mid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[3, 3.4], wspace=0.3)
    axE = fig.add_subplot(gs_mid[0])
    axF = fig.add_subplot(gs_mid[1])

    gs_corr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :], wspace=0.25)
    axG = fig.add_subplot(gs_corr[0])
    axH = fig.add_subplot(gs_corr[1])

    gs_bar = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4, :], wspace=0.25)
    axI = fig.add_subplot(gs_bar[0])
    axJ = fig.add_subplot(gs_bar[1])

    # Populate Panels
    plot_panel_A(axA, task_metrics)
    plot_panel_B(axB, task_metrics)
    plot_panel_C(axC, task_metrics)
    if not spectral_data.empty:
        plot_panel_D(axD, spectral_data)
        plot_panel_E(axE, spectral_data)
    plot_panel_F(axF, overlap_features, mofa_w, shap_v)
    if correlation_data:
        plot_correlation_panel(axG, correlation_data.get('leaf', {}).get('merged', pd.DataFrame()), 'Leaf', 'g')
        plot_correlation_panel(axH, correlation_data.get('root', {}).get('merged', pd.DataFrame()), 'Root', 'h')
        plot_overlap_bars_panels(axI, axJ, correlation_data)

    # Final Touches
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    path_png = os.path.join(OUTPUT_DIR, "fig_6.png")
    path_svg = os.path.join(OUTPUT_DIR, "fig_6.svg")

    plt.savefig(path_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path_svg, bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)
    print(f"Figure saved to {path_png} and {path_svg}")
    return path_png

if __name__ == "__main__":
    create_master_figure()












