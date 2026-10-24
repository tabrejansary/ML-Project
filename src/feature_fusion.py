"""
feature_fusion.py — Feature fusion strategies and feature selection.

5 Strategies compared:
  A: GLCM only
  B: LBP only
  C: GLCM + LBP concatenated  (basic fusion — our main contribution)
  D: GLCM + LBP + PCA reduction
  E: GLCM + LBP + SelectKBest (ANOVA F-test)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES_GLCM_CSV, FEATURES_LBP_CSV, FEATURES_FUSED_CSV
from src.utils import get_logger

logger = get_logger(__name__)


class FeatureFusion:
    """
    Loads GLCM and LBP feature CSVs and produces fused feature matrices
    for all 5 fusion strategies.
    """

    def __init__(self):
        self.glcm_df: Optional[pd.DataFrame] = None
        self.lbp_df:  Optional[pd.DataFrame] = None
        self.meta_cols = ["image_path", "patient_id", "label",
                          "class_name", "subtype", "magnification"]

    def load(self,
             glcm_csv: Path = FEATURES_GLCM_CSV,
             lbp_csv:  Path = FEATURES_LBP_CSV) -> None:
        """Load GLCM and LBP feature DataFrames from CSV."""
        self.glcm_df = pd.read_csv(glcm_csv)
        self.lbp_df  = pd.read_csv(lbp_csv)
        logger.info(f"Loaded GLCM features: {self.glcm_df.shape}")
        logger.info(f"Loaded LBP  features: {self.lbp_df.shape}")

    def _glcm_feat_cols(self) -> List[str]:
        return [c for c in self.glcm_df.columns if c.startswith("glcm_")]

    def _lbp_feat_cols(self) -> List[str]:
        return [c for c in self.lbp_df.columns if c.startswith("lbp_")]

    def get_strategy(
        self,
        strategy: str,
        split_col: str = "split",
        pca_components: int = 30,
        k_best: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        for the requested strategy.

        strategy : 'A' | 'B' | 'C' | 'D' | 'E'
        """
        assert self.glcm_df is not None, "Call .load() first"

        # Merge on image_path
        merged = self.glcm_df.merge(
            self.lbp_df[["image_path"] + self._lbp_feat_cols()],
            on="image_path"
        )

        glcm_cols = self._glcm_feat_cols()
        lbp_cols  = self._lbp_feat_cols()

        if strategy == "A":
            feat_cols = glcm_cols
        elif strategy == "B":
            feat_cols = lbp_cols
        elif strategy in ("C", "D", "E"):
            feat_cols = glcm_cols + lbp_cols
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use A/B/C/D/E")

        train = merged[merged[split_col] == "train"]
        val   = merged[merged[split_col] == "val"]
        test  = merged[merged[split_col] == "test"]

        X_train = train[feat_cols].values.astype(np.float32)
        X_val   = val[feat_cols].values.astype(np.float32)
        X_test  = test[feat_cols].values.astype(np.float32)
        y_train = train["label"].values
        y_val   = val["label"].values
        y_test  = test["label"].values
        feat_names = list(feat_cols)

        # Strategy D: PCA reduction
        if strategy == "D":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)
            n_comp = min(pca_components, X_train.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_val   = pca.transform(X_val)
            X_test  = pca.transform(X_test)
            feat_names = [f"PC{i+1}" for i in range(n_comp)]

        # Strategy E: SelectKBest
        if strategy == "E":
            k = min(k_best, X_train.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_train = selector.fit_transform(X_train, y_train)
            X_val   = selector.transform(X_val)
            X_test  = selector.transform(X_test)
            selected_idx = selector.get_support(indices=True)
            feat_names = [feat_cols[i] for i in selected_idx]

        logger.info(f"Strategy {strategy}: X_train={X_train.shape}, "
                    f"X_val={X_val.shape}, X_test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, feat_names

    def save_fused(self, output_path: Path = FEATURES_FUSED_CSV) -> None:
        """Save the concatenated (Strategy C) features to CSV."""
        assert self.glcm_df is not None, "Call .load() first"
        merged = self.glcm_df.merge(
            self.lbp_df[["image_path"] + self._lbp_feat_cols()],
            on="image_path"
        )
        merged.to_csv(output_path, index=False)
        logger.info(f"Saved fused features → {output_path}  shape={merged.shape}")
