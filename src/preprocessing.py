"""
preprocessing.py — Image loading and preprocessing pipeline.

Pipeline per image:
    Raw PNG  →  Read (BGR)  →  Grayscale  →  CLAHE  →  Resize  →  Normalize [0,1]

CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local
contrast making features invariant to staining differences between patients.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.config import IMAGE_SIZE
from src.utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Handles loading and preprocessing of histopathology images.

    Pipeline: BGR → Grayscale → CLAHE → Resize → Normalize

    CLAHE ensures GLCM/LBP features are patient-invariant by normalizing
    local contrast across all images regardless of staining variation.

    Parameters
    ----------
    image_size : tuple (H, W)
        Target size after resizing. Default: (224, 224)
    clahe_clip : float
        Clip limit for CLAHE. Higher = more contrast enhancement.
    clahe_grid : tuple
        Tile grid size for CLAHE.
    """

    def __init__(self,
                 image_size: Tuple[int, int] = IMAGE_SIZE,
                 clahe_clip: float = 2.0,
                 clahe_grid: Tuple[int, int] = (8, 8)):
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid
        )

    def load_image(self, path: str | Path) -> Optional[np.ndarray]:
        """Load an image from disk. Returns BGR array or None on failure."""
        path = str(path)
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Could not read image: {path}")
        return img

    def to_grayscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to normalize local contrast (patient-invariant)."""
        return self.clahe.apply(gray)

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size using area interpolation."""
        return cv2.resize(img, (self.image_size[1], self.image_size[0]),
                          interpolation=cv2.INTER_AREA)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1] float32."""
        return img.astype(np.float32) / 255.0

    def preprocess(self, path: str | Path) -> Optional[np.ndarray]:
        """
        Full pipeline: load → grayscale → CLAHE → resize → normalize.

        Returns float32 array (H, W) with values in [0, 1].
        """
        img = self.load_image(path)
        if img is None:
            return None
        gray    = self.to_grayscale(img)
        equalized = self.apply_clahe(gray)
        resized = self.resize(equalized)
        normed  = self.normalize(resized)
        return normed

    def preprocess_to_uint8(self, path: str | Path) -> Optional[np.ndarray]:
        """
        Load → grayscale → CLAHE → resize (keeps uint8 for GLCM).
        """
        img = self.load_image(path)
        if img is None:
            return None
        gray      = self.to_grayscale(img)
        equalized = self.apply_clahe(gray)
        resized   = self.resize(equalized)
        return resized  # uint8, [0, 255]
