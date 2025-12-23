import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ==========================================
# Configuration and Constants
# ==========================================

# Output directory
OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"

# Input file paths
TRANSFORMER_LEAF_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_class_performance_Leaf.csv"
TRANSFORMER_ROOT_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_class_performance_Root.csv"
BASELINE_LEAF_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_baseline_comparison_Leaf.csv"
BASELINE_ROOT_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_baseline_comparison_Root.csv"

LEAF_PRED_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_test_predictions_metadata_Leaf.csv"
ROOT_PRED_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_test_predictions_metadata_Root.csv"

# Visualization Settings
FIG_SIZE = (18, 12)
COLOR_MAP = 'BuGn'
DPI = 300

# Font Sizes
FONT_SIZE_TITLE = 24
FONT_SIZE_SUBTITLE = 18
FONT_SIZE_AXIS_LABEL = 16
FONT_SIZE_TICK_LABEL = 14
FONT_SIZE_CELL_TEXT = 13
FONT_SIZE_ERROR = 16

TASKS = ['Genotype', 'Treatment', 'Day']


def ensure_directories():
    """Ensures the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix(ax, df, task, tissue):
    """
    Generates and plots a confusion matrix for a specific task and tissue on the provided axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df (pandas.DataFrame): The DataFrame containing prediction data.
        task (str): The name of the task (e.g., 'Genotype').
        tissue (str): The tissue type (e.g., 'Leaf').
    """
    # Identify true and predicted columns
    true_cols = [f"{task}_true", f"{task.lower()}_true", f"{task}_true_encoded"]
    pred_cols = [f"{task}_pred", f"{task.lower()}_pred", f"{task}_pred_encoded"]
    
    true_col = next((col for col in true_cols if col in df.columns), None)
    pred_col = next((col for col in pred_cols if col in df.columns), None)
    
    if not true_col or not pred_col:
        ax.text(0.5, 0.5, f"Missing data for {task}", ha='center', va='center', fontsize=FONT_SIZE_TICK_LABEL)
        ax.axis('off')
        return
    
    # Prepare data subset and unique classes
    df_subset = df[[true_col, pred_col]].dropna()
    true_classes = sorted(df_subset[true_col].astype(str).unique())
    pred_classes = sorted(df_subset[pred_col].astype(str).unique())
    classes = sorted(list(set(true_classes) | set(pred_classes)))
    
    # Compute confusion matrix
    cm = confusion_matrix(df_subset[true_col].astype(str), df_subset[pred_col].astype(str), labels=classes)
    
    # Normalize confusion matrix (Recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        mask = row_sums > 0
        # Use broadcasting for division
        cm_normalized[mask] = cm[mask] / row_sums[mask, np.newaxis] if mask.any() else 0
    
    # Plot heatmap
    sns.heatmap(cm_normalized, cmap=COLOR_MAP, 
                xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABEL)
    
    # Annotate heatmap with counts and percentages
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
                # Adjust text color for visibility
                text_color = 'white' if cm_normalized[i, j] > 0.6 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color=text_color, fontsize=FONT_SIZE_CELL_TEXT)
    
    # Set labels and title
    ax.set_xlabel('Predicted', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('True', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title(f'{tissue} - {task}', fontsize=FONT_SIZE_SUBTITLE)


def main():
    """Main execution function for generating Supplementary Figure S4."""
    ensure_directories()

    try:
        print("Generating Confusion Matrices...")
        
        # Load prediction data
        leaf_pred = pd.read_csv(LEAF_PRED_PATH)
        root_pred = pd.read_csv(ROOT_PRED_PATH)
        
        # Initialize figure
        fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE)
        fig.suptitle('Confusion Matrices for Classification Tasks by Tissue', fontsize=FONT_SIZE_TITLE)
        
        datasets = [(leaf_pred, 'Leaf'), (root_pred, 'Root')]
        
        # Generate plots for each tissue and task
        for i, (df, tissue) in enumerate(datasets):
            for j, task in enumerate(TASKS):
                plot_confusion_matrix(axes[i, j], df, task, tissue)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        
        output_path = os.path.join(OUTPUT_DIR, 'Confusion_Matrices.png')
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        
        print(f"Successfully generated: {output_path}")

    except Exception as e:
        print(f"Error generating Confusion Matrices: {e}")
        
        # Create an error message plot
        plt.figure(figsize=(14, 8))
        plt.text(0.5, 0.5, f"Error generating confusion matrices: {str(e)}", 
                 ha='center', va='center', fontsize=FONT_SIZE_ERROR, wrap=True)
        plt.axis('off')
        
        # Save error plot to the same filename to indicate failure in the artifact
        plt.savefig(os.path.join(OUTPUT_DIR, 'Confusion_Matrices.png'), dpi=DPI)
        plt.close()


if __name__ == "__main__":
    main()
