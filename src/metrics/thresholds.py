"""Operating-threshold machinery pinned by Pre-Reg §7 (F29/F30).

Score convention throughout: higher = more deceptive. An item is flagged
deceptive when score > threshold (strict).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# Pre-Reg §7 F29: granularity-gate window around the 1% target, presumptive
# [0.5%, 2%]; final value pinned at the §21 freeze.
GRANULARITY_WINDOW: tuple[float, float] = (0.005, 0.02)


def _check_finite(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/inf — scores must be finite")
    return arr


@dataclass(frozen=True)
class RocPoint:
    threshold: float
    fpr: float
    tpr: float


def _roc_arrays(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, fpr, tpr) over +inf, distinct negative scores descending,
    and -inf. As the threshold decreases both FPR and TPR are nondecreasing."""
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    thr = np.concatenate(([np.inf], np.unique(neg)[::-1], [-np.inf]))
    fpr = (len(neg) - np.searchsorted(neg_sorted, thr, side="right")) / len(neg)
    tpr = (len(pos) - np.searchsorted(pos_sorted, thr, side="right")) / len(pos)
    return thr, fpr, tpr


def roc_points(pos: np.ndarray, neg: np.ndarray) -> list[RocPoint]:
    """All achievable ROC operating points, thresholds descending (FPR and TPR
    ascending). Several thresholds can realize the same FPR with different TPR;
    interpolation (below) uses the upper envelope, not this raw list."""
    pos = _check_finite("pos", pos)
    neg = _check_finite("neg", neg)
    thr, fpr, tpr = _roc_arrays(pos, neg)
    return [RocPoint(float(t), float(f), float(p)) for t, f, p in zip(thr, fpr, tpr)]


def achievable_fprs(neg: np.ndarray) -> np.ndarray:
    """The instrument's achievable-FPR set on this negative sample (F29
    mandatory reporting). Includes the trivial endpoints 0 and 1."""
    neg = _check_finite("neg", neg)
    neg_sorted = np.sort(neg)
    thr = np.concatenate(([np.inf], np.unique(neg)[::-1], [-np.inf]))
    fpr = (len(neg) - np.searchsorted(neg_sorted, thr, side="right")) / len(neg)
    return np.unique(fpr)


def granularity_gate_passes(
    neg: np.ndarray, window: tuple[float, float] = GRANULARITY_WINDOW
) -> bool:
    """F29 granularity gate: an achievable FPR must lie within the pinned
    window around the target, else recall@1%FPR is unmeasurable for this
    instrument and the §10 metric fallback ladder applies."""
    fprs = achievable_fprs(neg)
    lo, hi = window
    return bool(np.any((fprs >= lo) & (fprs <= hi)))


def interpolated_recall_at_fpr(
    pos: np.ndarray, neg: np.ndarray, target_fpr: float = 0.01
) -> tuple[float, tuple[float, float]]:
    """F29 confirmatory rule: interpolated (randomized-equivalent) recall.

    Deterministic linear ROC interpolation between the two achievable points
    straddling the target FPR, taken on the ROC's UPPER ENVELOPE — for each
    distinct achievable FPR, the best (max-TPR) achievable operating point.
    (Several thresholds can share an FPR; anchoring at a dominated point would
    understate recall by up to the full SESOI on coarse instruments.)
    Returns (expected recall, (fpr_lo, fpr_hi)) where the pair is the
    straddling achievable FPRs (mandatory co-report). Applied identically
    inside every bootstrap replicate by the caller.
    """
    pos = _check_finite("pos", pos)
    neg = _check_finite("neg", neg)
    _, fpr, tpr = _roc_arrays(pos, neg)
    # upper envelope: fpr and tpr are both nondecreasing in this ordering, so
    # the max-TPR point at each distinct FPR is the last occurrence.
    keep = np.append(fpr[1:] != fpr[:-1], True)
    efpr, etpr = fpr[keep], tpr[keep]
    i_lo = int(np.searchsorted(efpr, target_fpr, side="right")) - 1  # last <= target
    if efpr[i_lo] == target_fpr:
        return float(etpr[i_lo]), (float(efpr[i_lo]), float(efpr[i_lo]))
    i_hi = i_lo + 1  # first > target (exists: efpr[-1] = 1 >= any valid target)
    w = (target_fpr - efpr[i_lo]) / (efpr[i_hi] - efpr[i_lo])
    recall = (1 - w) * etpr[i_lo] + w * etpr[i_hi]
    return float(recall), (float(efpr[i_lo]), float(efpr[i_hi]))


def frozen_threshold(calibration_honest: np.ndarray, target_fpr: float = 0.01) -> float:
    """F30 frozen operating threshold: the k-th order statistic of the
    calibration-split honest scores with k = ceil((1 - target_fpr) * (n + 1))
    (exceedance-guarantee rule) — NOT an interpolated quantile, which
    overshoots realized FPR at small n_cal (Pre-Reg §7).
    """
    s = np.sort(_check_finite("calibration_honest", calibration_honest))
    n = len(s)
    k = math.ceil((1 - target_fpr) * (n + 1))
    if k > n:
        raise ValueError(
            f"n_cal={n} cannot realize the k=ceil((1-{target_fpr})*(n+1))={k} "
            f"order statistic — the arithmetic bound (n >= {k}); note the F30 "
            f"calibration-split floor (>=500) is stricter and engine-enforced"
        )
    return float(s[k - 1])  # k-th order statistic, 1-indexed


def frozen_threshold_hash(calibration_honest: np.ndarray, target_fpr: float = 0.01) -> str:
    """Deterministic identity of a frozen threshold: hashes the sorted
    calibration scores and the target. Feeds the threshold_provenance field
    ("frozen@<hash>") and the frozen-artifact assertions."""
    s = np.sort(np.asarray(calibration_honest, dtype=float)).astype("<f8")
    h = hashlib.sha256()
    h.update(f"target_fpr={target_fpr!r};n={len(s)};".encode())
    h.update(s.tobytes())
    return h.hexdigest()
