"""
Microscopic Remaining Oil Dynamic Evolution package.

Provides a CNN + LSTM fusion model with Darcy-law physical constraints for
predicting microscopic remaining oil saturation and dynamics.
"""

from .model import FusionModel, PhysicalConstraintLoss  # noqa: F401
from .preprocessing import (  # noqa: F401
    dynamic_time_series_preprocess,
    pair_modalities,
    static_image_preprocess,
)

__all__ = [
    "FusionModel",
    "PhysicalConstraintLoss",
    "static_image_preprocess",
    "dynamic_time_series_preprocess",
    "pair_modalities",
]
