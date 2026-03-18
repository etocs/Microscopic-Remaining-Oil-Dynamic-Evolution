from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F


def static_image_preprocess(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (256, 256, 64),
    clamp_std: Optional[float] = 3.0,
) -> Tensor:
    """
    Normalize and resize a 3D grayscale volume to (C=1, D, H, W).

    Steps: grayscale standardization -> outlier clipping -> trilinear resize.
    target_shape is provided as (H, W, D); the output tensor is (1, D, H, W).
    Global mean/std are used; swap in slice-wise normalization if required.
    """
    if volume.ndim != 3:
        raise ValueError("Expected volume with shape (H, W, D)")

    vol = torch.as_tensor(volume, dtype=torch.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    if clamp_std is not None:
        vol = torch.clamp(vol, -clamp_std, clamp_std)

    # Add channel and batch dimensions: (1, 1, D, H, W). PyTorch expects depth-first
    # ordering for volumetric ops, so we permute from (H, W, D) -> (D, H, W).
    vol = vol.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    target_h, target_w, target_d = target_shape
    vol = F.interpolate(
        vol,
        size=(target_d, target_h, target_w),
        mode="trilinear",
        align_corners=False,
    )
    return vol.squeeze(0)  # (1, D, H, W)


def dynamic_time_series_preprocess(
    sequence: Sequence[Sequence[float]],
    target_steps: int = 30,
) -> Tensor:
    """
    Interpolate missing values, align to target_steps, and apply min-max scaling.

    Accepts list/array or pandas DataFrame with shape (steps, features).
    """
    df = pd.DataFrame(sequence).interpolate(method="linear", limit_direction="both")
    data = df.to_numpy(dtype=np.float32)
    original_steps = data.shape[0]
    if original_steps != target_steps:
        x_orig = np.linspace(0, original_steps - 1, original_steps, dtype=np.float32)
        x_new = np.linspace(0, original_steps - 1, target_steps, dtype=np.float32)
        # Map target_steps uniformly across the original timeline to preserve temporal spacing.
        # x_orig/x_new are computed once and reused for every feature column.
        data_interp = np.empty((target_steps, data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            data_interp[:, i] = np.interp(x_new, x_orig, data[:, i])
        data = data_interp

    min_vals = data.min(axis=0, keepdims=True)
    max_vals = data.max(axis=0, keepdims=True)
    scaled = (data - min_vals) / (max_vals - min_vals + 1e-6)
    return torch.from_numpy(scaled)


def pair_modalities(
    static_features: Iterable[np.ndarray],
    dynamic_features: Iterable[np.ndarray],
    similarity_matrix: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Pair static and dynamic samples. If a similarity matrix is provided
    (shape: n_static x n_dynamic), each static sample is matched to the most
    similar dynamic sample to approximate pore-structure alignment. Ties are
    resolved by the first maximal entry returned by `argmax`; many static samples
    may map to the same dynamic sample when that best preserves pore similarity.
    """
    static_list = list(static_features)
    dynamic_list = list(dynamic_features)
    if similarity_matrix is None:
        return list(zip(static_list, dynamic_list))

    similarity_matrix = np.asarray(similarity_matrix)
    if similarity_matrix.shape[0] != len(static_list):
        raise ValueError("similarity_matrix row count must equal number of static items")
    if similarity_matrix.shape[1] != len(dynamic_list):
        raise ValueError("similarity_matrix column count must equal number of dynamic items")
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, row in enumerate(similarity_matrix):
        j = int(np.argmax(row))
        pairs.append((static_list[i], dynamic_list[j]))
    return pairs
