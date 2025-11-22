"""
Backend module for AI Model Bias Detector.

This package contains modules for data processing, model training, and fairness analysis.
"""

from . import data_processing
from . import model_train
from . import fairness

__all__ = ['data_processing', 'model_train', 'fairness']
