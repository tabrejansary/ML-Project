"""
config.py — Central configuration for the entire project.
All paths, constants, and settings live here.
Change things here, it propagates everywhere.
"""

from pathlib import Path

# ─── Root Paths ───────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATASET_DIR   = ROOT_DIR / "dataset" / "BreaKHis_v1" / "histology_slides" / "breast"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUTS_DIR   = ROOT_DIR / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
MODELS_DIR    = OUTPUTS_DIR / "models"
RESULTS_DIR   = OUTPUTS_DIR / "results"
SRC_DIR       = ROOT_DIR / "src"

# ─── Key Files ────────────────────────────────────────────────────────────────
METADATA_CSV        = ROOT_DIR / "metadata.csv"
SPLITS_DIR          = ROOT_DIR / "splits"
FEATURES_GLCM_CSV   = ROOT_DIR / "features_glcm.csv"
FEATURES_LBP_CSV    = ROOT_DIR / "features_lbp.csv"
FEATURES_FUSED_CSV  = ROOT_DIR / "features_fused.csv"

# ─── Dataset Constants ────────────────────────────────────────────────────────
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

BENIGN_SUBTYPES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]

MALIGNANT_SUBTYPES = [
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]

CLASS_MAP = {"benign": 0, "malignant": 1}
CLASS_NAMES = ["Benign", "Malignant"]

# ─── Preprocessing ────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # Resize target (H, W)

# ─── GLCM Settings ────────────────────────────────────────────────────────────
GLCM_DISTANCES  = [1, 2, 3]                   # Pixel distances
GLCM_ANGLES     = [0, 0.785, 1.571, 2.356]   # 0°, 45°, 90°, 135° in radians
GLCM_LEVELS     = 64                           # Gray levels (64 = industry standard, denser matrix = better stats)
GLCM_PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "ASM",
    "correlation",
]

# ─── LBP Settings ─────────────────────────────────────────────────────────────
LBP_SCALES = [
    {"P": 8,  "R": 1},   # Fine texture
    {"P": 16, "R": 2},   # Medium texture
    {"P": 24, "R": 3},   # Coarse texture
]
LBP_METHOD = "uniform"

# ─── Data Split ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ─── Cross Validation ─────────────────────────────────────────────────────────
CV_FOLDS = 5

# ─── Ensure output dirs exist ─────────────────────────────────────────────────
for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
