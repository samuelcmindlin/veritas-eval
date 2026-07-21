"""Known-answer + provenance regression tests for §7 discrimination metrics.

The provenance tests are the regression suite Pre-Reg §7 mandates: known-answer
tests cannot disambiguate the two threshold-policy semantics; a provenance test
can.
"""
import numpy as np
import pytest

from metrics import auroc, balanced_accuracy, recall_at_fpr


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0

    def test_reversed(self):
        assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0

    def test_hand_computed_with_ties(self):
        # pos = [1, 2], neg = [1, 0]: pairs -> (1,1)=.5, (1,0)=1, (2,1)=1, (2,0)=1
        assert auroc(np.array([1.0, 2.0]), np.array([1.0, 0.0])) == pytest.approx(3.5 / 4)

    def test_null_distribution_near_half(self):
        rng = np.random.default_rng(4)
        a = auroc(rng.normal(size=4000), rng.normal(size=4000))
        assert abs(a - 0.5) < 0.02


class TestThresholdProvenance:
    """Pre-Reg §7: threshold provenance is an explicit output field."""

    def setup_method(self):
        rng = np.random.default_rng(5)
        self.neg = rng.normal(size=400)
        self.pos = rng.normal(loc=2.0, size=150)
        self.cal = rng.normal(size=400)

    def test_per_condition_provenance(self):
        r = recall_at_fpr(self.pos, self.neg, threshold_policy="per_condition")
        assert r.meta["threshold_provenance"] == "per_condition"
        assert "realized_fpr" not in r.meta
        assert "achievable_fpr_straddle" in r.meta

    def test_frozen_provenance(self):
        r = recall_at_fpr(
            self.pos, self.neg, threshold_policy="frozen", calibration_honest=self.cal
        )
        prov = r.meta["threshold_provenance"]
        assert prov.startswith("frozen@") and len(prov) == len("frozen@") + 64
        assert "realized_fpr" in r.meta  # mandatory co-statistic

    def test_frozen_requires_calibration(self):
        with pytest.raises(ValueError, match="calibration"):
            recall_at_fpr(self.pos, self.neg, threshold_policy="frozen")

    def test_policies_diverge_under_uniform_shift(self):
        """The semantic difference the two policies exist to capture (§7):
        a uniform shift leaves per-condition recall unchanged but moves the
        frozen policy's realized FPR — the H4 story."""
        shift = 1.5
        pc_base = recall_at_fpr(self.pos, self.neg, threshold_policy="per_condition")
        pc_shift = recall_at_fpr(
            self.pos + shift, self.neg + shift, threshold_policy="per_condition"
        )
        assert pc_shift.point == pytest.approx(pc_base.point)

        fr_base = recall_at_fpr(
            self.pos, self.neg, threshold_policy="frozen", calibration_honest=self.cal
        )
        fr_shift = recall_at_fpr(
            self.pos + shift,
            self.neg + shift,
            threshold_policy="frozen",
            calibration_honest=self.cal,
        )
        assert fr_shift.meta["realized_fpr"] > fr_base.meta["realized_fpr"] + 0.05

    def test_balanced_accuracy_carries_provenance(self):
        r = balanced_accuracy(self.pos, self.neg, threshold_policy="per_condition")
        assert r.meta["threshold_provenance"] == "per_condition"
        assert r.meta["tnr_convention"] == "randomized_equivalent_expectation"

    def test_per_condition_reports_achievable_fpr_summary(self):
        r = recall_at_fpr(self.pos, self.neg, threshold_policy="per_condition")
        assert r.meta["achievable_fpr_count"] > 0
        assert isinstance(r.meta["granularity_gate_passes"], bool)


class TestPointKnownAnswers:
    def test_frozen_recall_point(self):
        # cal = 1..199 -> k = ceil(0.99 * 200) = 198 -> threshold = 198.0.
        # pos = [199, 200, 100]: flagged (score > 198) are 199 and 200 -> 2/3.
        cal = np.arange(1.0, 200.0)
        r = recall_at_fpr(
            np.array([199.0, 200.0, 100.0]),
            np.zeros(10),
            threshold_policy="frozen",
            calibration_honest=cal,
        )
        assert r.point == pytest.approx(2 / 3)
        assert r.meta["threshold"] == 198.0
        assert r.meta["realized_fpr"] == 0.0

    def test_balanced_accuracy_point(self):
        # 100 distinct negatives, all positives above them: recall@1%FPR = 1.0,
        # TNR convention = 1 - 0.01 -> BA = (1.0 + 0.99) / 2.
        r = balanced_accuracy(np.full(10, 200.0), np.arange(100.0))
        assert r.point == pytest.approx(0.995)
