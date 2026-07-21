"""Metrics with the Pre-Reg §7/§12 pinned machinery.

Primary endpoint: per-condition-threshold recall@1%FPR (pure discrimination).
Frozen-threshold recall + realized-FPR co-statistic is descriptive / H4.
Inference: joint two-way (cluster × seed) percentile bootstrap only.
"""
from .bootstrap import (
    PREREG_MIN_B,
    BootstrapResult,
    draw_replicate,
    joint_bootstrap_delta,
)
from .core import MetricResult, auroc, balanced_accuracy, recall_at_fpr
from .thresholds import (
    GRANULARITY_WINDOW,
    achievable_fprs,
    frozen_threshold,
    frozen_threshold_hash,
    granularity_gate_passes,
    interpolated_recall_at_fpr,
    roc_points,
)

__all__ = [
    "PREREG_MIN_B",
    "BootstrapResult",
    "GRANULARITY_WINDOW",
    "MetricResult",
    "achievable_fprs",
    "auroc",
    "balanced_accuracy",
    "draw_replicate",
    "frozen_threshold",
    "frozen_threshold_hash",
    "granularity_gate_passes",
    "interpolated_recall_at_fpr",
    "joint_bootstrap_delta",
    "recall_at_fpr",
    "roc_points",
]
