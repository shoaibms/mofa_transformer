import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from matplotlib.patches import Patch

# Set output directory
output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
os.makedirs(output_dir, exist_ok=True)

# Input file paths
transformer_leaf_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_class_performance_Leaf.csv"
transformer_root_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_class_performance_Root.csv"
baseline_leaf_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_baseline_comparison_Leaf.csv"
baseline_root_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_baseline_comparison_Root.csv"

leaf_pred_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\leaf\transformer_test_predictions_metadata_Leaf.csv"
root_pred_path = r"C:\Users\ms\Desktop\hyper\output\transformer\phase1.1\root\transformer_test_predictions_metadata_Root.csv"

# SUPPLEMENTARY FIGURE S4: Confusion Matrices
try:
    print("Generating Confusion Matrices...")
    # Load prediction data
    leaf_pred = pd.read_csv(leaf_pred_path)
    root_pred = pd.read_csv(root_pred_path)

    # Define tasks
    tasks = ['Genotype', 'Treatment', 'Day']

    # Create a 2x3 subplot grid for confusion matrices (2 tissues x 3 tasks)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices for Classification Tasks by Tissue', fontsize=24)

    # Helper function to create confusion matrix for a specific task and tissue
    def plot_confusion_matrix(ax, df, task, tissue):
        # Get true and predicted values column names
        true_cols = [f"{task}_true", f"{task.lower()}_true", f"{task}_true_encoded"]
        pred_cols = [f"{task}_pred", f"{task.lower()}_pred", f"{task}_pred_encoded"]
        
        # Find the first matching column names
        true_col = next((col for col in true_cols if col in df.columns), None)
        pred_col = next((col for col in pred_cols if col in df.columns), None)
        
        if not true_col or not pred_col:
            ax.text(0.5, 0.5, f"Missing data for {task}", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Get unique classes and ensure they're strings
        df_subset = df[[true_col, pred_col]].dropna()
        true_classes = sorted(df_subset[true_col].astype(str).unique())
        pred_classes = sorted(df_subset[pred_col].astype(str).unique())
        classes = sorted(list(set(true_classes) | set(pred_classes)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(df_subset[true_col].astype(str), df_subset[pred_col].astype(str), labels=classes)
        
        # Normalize by row (true labels) to get recall
        row_sums = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                cm_normalized[i] = cm[i] / row_sums[i]
        
        # Create heatmap with both counts and percentages - changed colormap to BuGn as requested
        sns.heatmap(cm_normalized, cmap='BuGn', 
                    xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
        
        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add both counts and percentages as annotations with increased font size
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
                    ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                            color='white' if cm_normalized[i, j] > 0.6 else 'black',
                            fontsize=13)  # Increased font size for cell text
        
        # Add labels and title with increased font sizes
        ax.set_xlabel('Predicted', fontsize=16)
        ax.set_ylabel('True', fontsize=16)
        ax.set_title(f'{tissue} - {task}', fontsize=18)

    # Plot confusion matrices for each task and tissue
    for i, tissue_data in enumerate([(leaf_pred, 'Leaf'), (root_pred, 'Root')]):
        df, tissue = tissue_data
        for j, task in enumerate(tasks):
            plot_confusion_matrix(axes[i, j], df, task, tissue)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'Confusion_Matrices.png'), dpi=300)
    plt.close()
    print(f"Successfully generated Confusion_Matrices.png")

except Exception as e:
    print(f"Error generating Confusion Matrices: {e}")
    # Create an error message plot
    plt.figure(figsize=(14, 8))
    plt.text(0.5, 0.5, f"Error generating confusion matrices: {str(e)}", 
            ha='center', va='center', fontsize=16, wrap=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'Confusion_Matrices.png'), dpi=300)
    plt.close()

print("Confusion matrix generation complete.")