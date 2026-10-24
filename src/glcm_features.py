"""
glcm_features.py — Multi-angle, multi-distance GLCM feature extractor.

For each image:
  → 4 angles × 3 distances = 12 GLCM matrices
  → 6 properties per matrix = 72 raw values
  → Average across angles  = 18 final features (3 distances × 6 props)

Industry-level: captures texture at multiple scales and orientations.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List

from src.config import GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS, GLCM_PROPERTIES
from src.utils import get_logger

logger = get_logger(__name__)


class GLCMExtractor:
    """
    Extracts GLCM-based texture features from a grayscale uint8 image.

    Features extracted per distance (averaged over all angles):
        contrast, dissimilarity, homogeneity, energy, ASM, correlation

    With 3 distances → 3 × 6 = 18 features total.
    Feature names are like: glcm_d1_contrast, glcm_d2_energy, etc.
    """

    def __init__(
        self,
        distances: List[int]   = GLCM_DISTANCES,
        angles:    List[float] = GLCM_ANGLES,
        levels:    int         = GLCM_LEVELS,
    ):
        self.distances = distances
        self.angles    = angles
        self.levels    = levels

    def extract(self, img_uint8: np.ndarray) -> np.ndarray:
        """
        Extract GLCM feature vector from a uint8 grayscale image.

        Parameters
        ----------
        img_uint8 : np.ndarray  uint8 grayscale image (H, W)

        Returns
        -------
        np.ndarray of shape (n_distances × n_properties,)
            i.e. 3 × 6 = 18 features, averaged over angles.
        """
        try:
            # graycomatrix returns shape (levels, levels, n_dist, n_angles)
            glcm = graycomatrix(
                img_uint8,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=True,
                normed=True,
            )

            feature_vector = []
            for d_idx in range(len(self.distances)):
                for prop in GLCM_PROPERTIES:
                    # graycoprops returns shape (n_dist, n_angles)
                    vals = graycoprops(glcm, prop)[d_idx, :]   # values for all angles at distance d
                    feature_vector.append(float(vals.mean()))   # average over angles

            return np.array(feature_vector, dtype=np.float32)

        except Exception as e:
            logger.error(f"GLCM extraction failed: {e}")
            n_features = len(self.distances) * len(GLCM_PROPERTIES)
            return np.zeros(n_features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return feature names matching the output vector order."""
        names = []
        for d in self.distances:
            for prop in GLCM_PROPERTIES:
                names.append(f"glcm_d{d}_{prop}")
        return names

    @property
    def n_features(self) -> int:
        return len(self.distances) * len(GLCM_PROPERTIES)
