"""
lbp_features.py — Multi-scale LBP (Local Binary Pattern) feature extractor.

3 scales:
  Fine:   P=8,  R=1  → captures micro-texture
  Medium: P=16, R=2  → captures mid-level texture
  Coarse: P=24, R=3  → captures macro-texture

Each scale produces a normalized histogram.
All 3 histograms are concatenated → final feature vector.
"""

import numpy as np
from skimage.feature import local_binary_pattern
from typing import List

from src.config import LBP_SCALES, LBP_METHOD
from src.utils import get_logger

logger = get_logger(__name__)


class LBPExtractor:
    """
    Multi-scale LBP extractor.

    For each scale (P, R):
        - Compute LBP image
        - Build normalized histogram (n_bins = P + 2 for 'uniform')
        - Concatenate all histograms

    Total features = sum of (P + 2) for each scale
        = (8+2) + (16+2) + (24+2) = 10 + 18 + 26 = 54 features
    """

    def __init__(self, scales: List[dict] = LBP_SCALES, method: str = LBP_METHOD):
        self.scales = scales
        self.method = method

    def _n_bins(self, P: int) -> int:
        """Number of histogram bins for uniform LBP."""
        return P + 2   # uniform patterns + 1 non-uniform bin

    def extract_one_scale(self, img_float: np.ndarray, P: int, R: float) -> np.ndarray:
        """
        Compute LBP histogram for one (P, R) scale.

        Parameters
        ----------
        img_float : np.ndarray  float [0,1] grayscale
        P         : int         number of neighbors
        R         : float       radius

        Returns
        -------
        np.ndarray  normalized histogram of shape (P+2,)
        """
        lbp_img = local_binary_pattern(img_float, P=P, R=R, method=self.method)
        n_bins   = self._n_bins(P)
        hist, _  = np.histogram(lbp_img.ravel(), bins=n_bins,
                                range=(0, n_bins), density=True)
        return hist.astype(np.float32)

    def extract(self, img_float: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale LBP feature vector.

        Parameters
        ----------
        img_float : np.ndarray  float [0,1] grayscale image

        Returns
        -------
        np.ndarray of shape (54,)  — concatenated multi-scale histograms
        """
        try:
            histograms = []
            for scale in self.scales:
                hist = self.extract_one_scale(img_float, scale["P"], scale["R"])
                histograms.append(hist)
            return np.concatenate(histograms)
        except Exception as e:
            logger.error(f"LBP extraction failed: {e}")
            return np.zeros(self.n_features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return feature names matching the output vector."""
        names = []
        for scale in self.scales:
            P, R = scale["P"], scale["R"]
            n_bins = self._n_bins(P)
            for i in range(n_bins):
                names.append(f"lbp_P{P}R{R}_bin{i}")
        return names

    @property
    def n_features(self) -> int:
        return sum(self._n_bins(s["P"]) for s in self.scales)
