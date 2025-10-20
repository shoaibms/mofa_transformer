# --- manuscript_style.py Color Definitions (Hex Only - Muted Blue/Green/Yellow/Grey Focus) ---

COLORS = {
    # ==========================================================================
    # == Core Experimental Variables ==
    # ==========================================================================
    # Using distinct core families: Blues for Genotypes, Greens for Treatments
    'G1': '#3182bd',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#2ca25f',             # Susceptible Genotype (Medium Teal)

    'T0': '#74c476',             # Control Treatment (Medium Green)
    'T1': '#fe9929',             # Stress Treatment (Muted Orange/Yellow)

    'Leaf': '#006d2c',            # Leaf Tissue (Darkest Green)
    'Root': '#08519c',            # Root Tissue (Darkest Blue)

    # --- Days (Subtle Yellow-Green sequence) ---
    'Day1': '#ffffcc',            # Very Light Yellow-Green
    'Day2': '#c2e699',            # Light Yellow-Green
    'Day3': '#78c679',            # Medium Yellow-Green

    # ==========================================================================
    # == Data Types / Omics / Features ==
    # ==========================================================================
    # Using distinct Blue/Green families for general types
    'Spectral': '#6baed6',        # General Spectral (Medium Blue)
    'Metabolite': '#41ab5d',       # General Metabolite (Medium-Dark Yellow-Green)
    'UnknownFeature': '#969696',  # Medium Grey for fallback

    # --- Specific Spectral Categories --- (Using blues, teals, greens, greys)
    'Spectral_Water': '#3182bd',     # Medium-Dark Blue
    'Spectral_Pigment': '#238b45',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#08519c',         # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',      # Medium Grey

    # --- Specific Metabolite Categories --- (Using Yellow/Greens)
    'Metabolite_PCluster': '#006837', # Darkest Yellow-Green
    'Metabolite_NCluster': '#ffffd4', # Very Light Yellow
    'Metabolite_Other': '#bdbdbd',     # Light Grey

    # ==========================================================================
    # == Methods & Model Comparison ==
    # ==========================================================================
    # Using distinct shades for clarity
    'MOFA': '#08519c',            # Dark Blue
    'SHAP': '#006d2c',            # Dark Green
    'Overlap': '#41ab5d',         # Medium-Dark Yellow-Green

    'Transformer': '#6baed6',     # Medium Blue
    'RandomForest': '#74c476',    # Medium Green
    'KNN': '#7fcdbb',             # Medium Teal

    # ==========================================================================
    # == Network Visualization Elements ==
    # ==========================================================================
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#08519c',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Metabolite': '#41ab5d',   # Default Metabolite Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # ==========================================================================
    # == Statistical & Difference Indicators ==
    # ==========================================================================
    # Using Green for positive, muted Yellow for negative, Dark Blue for significance
    'Positive_Diff': '#238b45',     # Medium-Dark Green
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