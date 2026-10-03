from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _watershed_class(binary: np.ndarray, min_distance: int, min_area: int) -> list[np.ndarray]:
    if not binary.any():
        return []
    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        dist,
        min_distance=max(2, int(min_distance)),
        labels=binary,
        exclude_border=False,
        threshold_abs=2,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    if markers.max() == 0:
        markers, _ = ndi.label(binary)
    labels = watershed(-dist, markers, mask=binary)
    return [(labels == lab) for lab in range(1, int(labels.max()) + 1) if int((labels == lab).sum()) >= min_area]


def _components(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    labels, n = ndi.label(binary)
    return [(labels == lab) for lab in range(1, n + 1) if int((labels == lab).sum()) >= min_area]


def semantic_to_instances(sem: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """Convert semantic classes to meaningful instances with class-specific rules.

    Grains/pores use marker-controlled watershed; fractures/cement use connected
    components. Area thresholds scale with image size to suppress texture speckle.
    """
    h, w = sem.shape
    px = h * w
    base = min_area or max(25, int(px * 0.00005))
    inst = np.zeros_like(sem, dtype=np.int32)
    oid = 1

    class_rules = {
        1: ('watershed', max(8, int(min(h, w) * 0.025)), max(base, int(px * 0.00020))),
        2: ('watershed', max(4, int(min(h, w) * 0.010)), max(18, int(px * 0.000035))),
        3: ('components', 0, max(15, int(px * 0.000025))),
        4: ('components', 0, max(50, int(px * 0.00010))),
    }

    for cls, (method, distance, area) in class_rules.items():
        binary = sem == cls
        masks = _watershed_class(binary, distance, area) if method == 'watershed' else _components(binary, area)
        for m in masks:
            inst[m] = oid
            oid += 1
    return inst
