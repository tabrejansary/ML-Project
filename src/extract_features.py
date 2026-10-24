"""
Fast GLCM + LBP extraction script.
- Uses 64 gray levels (industry standard)
- Parallel processing with joblib
- Saves features_glcm.csv and features_lbp.csv
Run: .venv\Scripts\python src\extract_features.py
"""

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time

from src.config import (
    METADATA_CSV, FEATURES_GLCM_CSV, FEATURES_LBP_CSV,
    GLCM_LEVELS
)
from src.preprocessing import ImagePreprocessor
from src.glcm_features import GLCMExtractor
from src.lbp_features import LBPExtractor

# ── Init ───────────────────────────────────────────────────────────────────────
prep = ImagePreprocessor()
glcm = GLCMExtractor()
lbp  = LBPExtractor()

metadata = pd.read_csv(METADATA_CSV)
print(f"Loaded metadata: {len(metadata)} images")
print(f"GLCM: {glcm.n_features} features | LBP: {lbp.n_features} features")
print(f"GLCM gray levels: {GLCM_LEVELS}")
print()


# ── Per-image extraction function ─────────────────────────────────────────────
def extract_one(row):
    """Extract GLCM + LBP features for one image row."""
    meta = {
        'image_path'   : row['image_path'],
        'patient_id'   : row['patient_id'],
        'class_name'   : row['class_name'],
        'label'        : row['label'],
        'subtype'      : row['subtype'],
        'magnification': row['magnification'],
        'split'        : row['split'],
    }

    # Preprocess
    img_uint8 = prep.preprocess_to_uint8(row['image_path'])
    img_float = prep.preprocess(row['image_path'])

    if img_uint8 is None or img_float is None:
        g_feats = np.zeros(glcm.n_features, dtype=np.float32)
        l_feats = np.zeros(lbp.n_features,  dtype=np.float32)
    else:
        # Quantize to GLCM_LEVELS gray levels (critical for speed + quality)
        img_q = (img_uint8 / 255.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
        g_feats = glcm.extract(img_q)
        l_feats = lbp.extract(img_float)

    glcm_record = {**meta, **dict(zip(glcm.get_feature_names(), g_feats))}
    lbp_record  = {**meta, **dict(zip(lbp.get_feature_names(),  l_feats))}
    return glcm_record, lbp_record


# ── Run parallel extraction ────────────────────────────────────────────────────
print("Starting feature extraction (parallel, 4 jobs)...")
print("This will take ~5-8 minutes.\n")

rows = [row for _, row in metadata.iterrows()]
start = time.time()

results = Parallel(n_jobs=4, prefer='threads')(
    delayed(extract_one)(row)
    for row in tqdm(rows, desc="Extracting features", unit="img")
)

elapsed = time.time() - start
print(f"\nDone in {elapsed/60:.1f} minutes")

# ── Save ───────────────────────────────────────────────────────────────────────
glcm_records = [r[0] for r in results]
lbp_records  = [r[1] for r in results]

glcm_df = pd.DataFrame(glcm_records)
lbp_df  = pd.DataFrame(lbp_records)

glcm_df.to_csv(FEATURES_GLCM_CSV, index=False)
lbp_df.to_csv( FEATURES_LBP_CSV,  index=False)

print(f"\nSaved:")
print(f"  features_glcm.csv  shape={glcm_df.shape}  -> {FEATURES_GLCM_CSV}")
print(f"  features_lbp.csv   shape={lbp_df.shape}   -> {FEATURES_LBP_CSV}")
print()
print("FEATURE EXTRACTION COMPLETE!")
print(f"  GLCM features per image : {glcm.n_features}")
print(f"  LBP  features per image : {lbp.n_features}")
print(f"  Total fused features    : {glcm.n_features + lbp.n_features}")
