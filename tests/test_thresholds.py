"""Known-answer tests for the Pre-Reg §7 threshold machinery (F29/F30)."""
import numpy as np
import pytest

from metrics import (
    achievable_fprs,
    frozen_threshold,
    frozen_threshold_hash,
    granularity_gate_passes,
    interpolated_recall_at_fpr,
)


class TestFrozenThreshold:
    def test_order_statistic_exact(self):
        # n=199: k = ceil(0.99 * 200) = 198 -> 198th smallest value
        cal = np.arange(1.0, 200.0)  # 1..199
        assert frozen_threshold(cal) == 198.0

    def test_exceedance_guarantee_on_average(self):
        # E[FPR] = (n+1-k)/(n+1) <= target by construction
        rng = np.random.default_rng(0)
        realized = []
        for _ in range(300):
            cal = rng.normal(size=499)
            t = frozen_threshold(cal)
            fresh = rng.normal(size=2000)
            realized.append(np.mean(fresh > t))
        assert np.mean(realized) <= 0.0105  # small slack over the 1% target

    def test_calibration_floor_raises(self):
        # k > n whenever n < 99 at the 1% target
        with pytest.raises(ValueError, match="cannot realize"):
            frozen_threshold(np.arange(50.0))

    def test_hash_stable_and_sensitive(self):
        cal = np.arange(1.0, 200.0)
        h1 = frozen_threshold_hash(cal)
        h2 = frozen_threshold_hash(cal.copy())
        assert h1 == h2 and len(h1) == 64
        assert frozen_threshold_hash(cal + 1.0) != h1

    def test_hash_order_invariant(self):
        rng = np.random.default_rng(1)
        cal = rng.normal(size=200)
        shuffled = rng.permutation(cal)
        assert frozen_threshold_hash(cal) == frozen_threshold_hash(shuffled)


class TestInterpolatedRecall:
    def test_upper_envelope_at_fpr_zero(self):
        # Coarse negatives: 90 at 0.0, 10 at 1.0; all positives at 2.0.
        # The BEST achievable point at FPR=0 is t=1.0 with TPR=1.0 (the
        # dominated t=+inf point has TPR=0 and must not anchor the mixture).
        # Envelope: (0, 1.0), (0.10, 1.0), (1, 1) -> recall at 0.01 = 1.0.
        neg = np.array([0.0] * 90 + [1.0] * 10)
        pos = np.full(50, 2.0)
        recall, straddle = interpolated_recall_at_fpr(pos, neg, 0.01)
        assert recall == pytest.approx(1.0)
        assert straddle == (0.0, pytest.approx(0.10))

    def test_hand_computed_mixture(self):
        # neg: 197 at 0.0, 3 at 5.0 -> achievable FPRs {0, 0.015, 1}
        # (granularity gate passes: 0.015 is in [0.005, 0.02]).
        # pos: 6 at 10.0, 4 at 1.0 -> TPR(t=5)=0.6, TPR(t=0)=1.0.
        # Target 0.01: w = 0.01/0.015 = 2/3 -> recall = (1/3)(0.6) + (2/3)(1.0).
        neg = np.array([0.0] * 197 + [5.0] * 3)
        pos = np.array([10.0] * 6 + [1.0] * 4)
        recall, straddle = interpolated_recall_at_fpr(pos, neg, 0.01)
        assert recall == pytest.approx(13 / 15)
        assert straddle == (0.0, pytest.approx(0.015))
        assert granularity_gate_passes(neg)

    def test_exact_operating_point_no_interpolation(self):
        # 100 distinct negatives -> FPR = 0.01 exactly achievable.
        neg = np.arange(100.0)
        pos = np.full(10, 200.0)
        recall, straddle = interpolated_recall_at_fpr(pos, neg, 0.01)
        assert recall == 1.0
        assert straddle[0] == straddle[1] == pytest.approx(0.01)

    def test_uniform_shift_invariance(self):
        # §7: the per-condition policy is invariant to a uniform score shift.
        rng = np.random.default_rng(2)
        neg = rng.normal(size=500)
        pos = rng.normal(loc=2.0, size=200)
        base, _ = interpolated_recall_at_fpr(pos, neg, 0.01)
        shifted, _ = interpolated_recall_at_fpr(pos + 3.7, neg + 3.7, 0.01)
        assert shifted == pytest.approx(base)


class TestGranularityGate:
    def test_fires_on_ultra_coarse_instrument(self):
        # One negative value -> achievable FPRs {0, 1}: nothing in [0.5%, 2%].
        neg = np.zeros(1000)
        assert not granularity_gate_passes(neg)

    def test_passes_on_continuous_instrument(self):
        neg = np.random.default_rng(3).normal(size=1000)
        assert granularity_gate_passes(neg)

    def test_achievable_fprs_reported(self):
        neg = np.array([0.0] * 90 + [1.0] * 10)
        fprs = achievable_fprs(neg)
        assert set(np.round(fprs, 6)) == {0.0, 0.1, 1.0}

    def test_nan_scores_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            interpolated_recall_at_fpr(
                np.array([1.0, np.nan]), np.array([0.0, 0.5]), 0.01
            )
