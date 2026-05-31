"""
LightViT-AD: Lightweight ViT Teacher-Student Anomaly Detection for UAV Imagery.

Modules:
    models  — ViTTeacher, ViTStudent, CombinedModel, anomaly_score
    dataset — DataLoader, compute_mean_and_std
    train   — train_epoch, validate, evaluate_plot, save/load_checkpoint
    utils   — set_seed, compute_flops_and_params, track_peak_rss
"""

from .models  import ViTTeacher, ViTStudent, CombinedModel, anomaly_score
from .dataset import DataLoader, compute_mean_and_std
from .train   import (
    train_epoch, validate, evaluate_plot,
    save_checkpoint, load_checkpoint,
)
from .utils   import set_seed, compute_flops_and_params, track_peak_rss

__version__ = '1.0.0'
__all__ = [
    'ViTTeacher', 'ViTStudent', 'CombinedModel', 'anomaly_score',
    'DataLoader', 'compute_mean_and_std',
    'train_epoch', 'validate', 'evaluate_plot',
    'save_checkpoint', 'load_checkpoint',
    'set_seed', 'compute_flops_and_params', 'track_peak_rss',
]
