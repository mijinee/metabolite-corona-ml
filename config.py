"""
Input/output paths and shared parameters.
"""

from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_BASE = ROOT / "output"

# Raw inputs (place in DATA_DIR)
CD_EXPORTS = {                                   # Compound Discoverer exports
    "HILIC-": DATA_DIR / "HILIC- (all).xlsx",
    "HILIC+": DATA_DIR / "HILIC+ (all).xlsx",
    "RP-":    DATA_DIR / "RP- (all).xlsx",
    "RP+":    DATA_DIR / "RP+ (all).xlsx",
}
CURATION_PATH     = DATA_DIR / "Other_Unknown_Manual_Curation.xlsx"
LIPIDMAPS_MAPPING = DATA_DIR / "LipidMaps_Feature_Mapping.xlsx"
REPRO_RUN_DIR     = DATA_DIR / "reproducibility_run"   # CD exports of the repeated (AT)15-pH run

# Output folders
OUT_PREPROCESS    = OUT_BASE / "00_preprocessing"
OUT_CLASSIFY      = OUT_BASE / "01_classification"
OUT_ML_EVAL       = OUT_BASE / "02_ml_evaluation"
OUT_LOG2FC        = OUT_BASE / "03_log2fc"
OUT_BATCH         = OUT_BASE / "04_batch_reproducibility"
OUT_FIGURES       = OUT_BASE / "05_figures"
OUT_MUMMICHOG     = OUT_BASE / "06_pathway_analysis"
OUT_SUPPLEMENTARY = OUT_BASE / "07_supplementary"

# Files passed between steps
COMBINED_PATH  = OUT_PREPROCESS / "01_Global_Analysis_Combined.xlsx"
PROCESSED_PATH = OUT_PREPROCESS / "Master_Processed_Conservative_Data.xlsx"
PRED_PATH      = OUT_CLASSIFY / "metabolite_class_predictions.xlsx"
HILIC_LOG2FC   = OUT_LOG2FC / "HILIC_ALL_Log2FC.xlsx"
RP_LOG2FC      = OUT_LOG2FC / "RP_ALL_Log2FC.xlsx"

# Experimental design
POLYMERS     = ["AT", "CTT", "DMG", "DSPE", "GT"]
DEFECTS      = ["3F", "C", "DN", "pH", "R"]      # R = pristine (no quantum defect)
PRISTINE     = "R"
QWD_DEFECTS  = [d for d in DEFECTS if d != PRISTINE]
DNA_POLYMERS = ["AT", "CTT", "GT"]
PEG_POLYMERS = ["DMG", "DSPE"]
DEFECT_LABELS = {"3F": "Trifluoro", "C": "Carboxyl", "DN": "Dinitro", "pH": "Amino"}

# Metabolite classes analysed
RP_CLASSES   = ["Phospholipids", "Complex Lipids", "Lysophospholipids",
                "Acylcarnitines", "Fatty Acids"]
HILIC_TRACK1 = ["Phospholipids", "Lysophospholipids", "Organic Acids",
                "Fatty Acids & Amides", "Secondary & Xenobiotic"]
HILIC_TRACK2 = ["Acylcarnitines", "Complex Lipids", "Benzenoids",
                "Amino Acids & Peptides"]

# Significance thresholds
FC_THRESHOLD     = 1.0    # |log2FC|, corona vs plasma and ssDNA vs PEG
QWD_FC_THRESHOLD = 0.5    # |log2FC|, QWD vs pristine
Q_THRESHOLD      = 0.05   # Benjamini-Hochberg FDR
