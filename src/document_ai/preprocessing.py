from __future__ import annotations
import cv2
import numpy as np


def preprocess_scan(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    den = cv2.fastNlMeansDenoising(clahe, None, 8, 7, 21)
    bw = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    return gray, clahe, bw


def ocr_variants(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Generate conservative OCR variants; the OCR stage selects the best by confidence."""
    gray, clahe, bw = preprocess_scan(rgb)
    h, w = gray.shape
    scale = 2.0 if max(h, w) < 2600 else 1.35

    def up(img):
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    up_gray = up(gray)
    up_clahe = up(clahe)
    # Unsharp mask helps the intentionally degraded synthetic scan.
    blur = cv2.GaussianBlur(up_clahe, (0, 0), 1.1)
    sharp = cv2.addWeighted(up_clahe, 1.7, blur, -0.7, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 11)

    return {
        'original_rgb': rgb,
        'upscaled_gray': cv2.cvtColor(up_gray, cv2.COLOR_GRAY2RGB),
        'clahe_sharpen': cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB),
        'otsu': cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB),
        'adaptive': cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB),
    }
