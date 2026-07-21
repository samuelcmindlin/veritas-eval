"""Tests for the joint two-way (cluster × seed) percentile bootstrap (§12).

Includes the §8 build-risk check that the cluster bootstrap demonstrably
diverges from a naive i.i.d. bootstrap on clustered data.
"""
import numpy as np
import pytest

from metrics import PREREG_MIN_B, draw_replicate, joint_bootstrap_delta


def _make_data(
    rng,
    n_clusters=25,
    items_per_cluster=8,
    n_seeds=5,
    cluster_sd=0.0,
    treated_shift=0.0,
):
    """Long-format synthetic scores for two conditions, both classes, with a
    per-cluster treatment-effect random component (cluster_sd) — a common
    additive cluster shift would cancel in shift-invariant statistics, so the
    cluster effect must live in the effect itself to exercise clustering.
    Higher score = more deceptive."""
    rows = {"lineage_id": [], "condition": [], "label": [], "score": [], "seed": []}
    for c in range(n_clusters):
        u_treat = rng.normal(scale=cluster_sd)  # cluster-varying treatment effect
        for cond in ("control", "treated"):
            for lab, loc in (("honest", 0.0), ("deceptive", 2.0)):
                # treatment degrades separation (pulls deceptive toward honest)
                eff = loc
                if cond == "treated" and lab == "deceptive":
                    eff = loc - treated_shift - u_treat
                for s in range(n_seeds):
                    for _ in range(items_per_cluster):
                        rows["lineage_id"].append(f"c{c}")
                        rows["condition"].append(cond)
                        rows["label"].append(lab)
                        rows["score"].append(rng.normal(loc=eff))
                        rows["seed"].append(s)
    return {k: np.asarray(v) for k, v in rows.items()}


class TestMechanics:
    def test_seed_draw_is_global_not_per_cluster(self):
        """F27/F28: ONE seed-index set per outer replicate, applied across all
        clusters — draw_replicate returns a single global seed draw."""
        rng = np.random.default_rng(0)
        clusters = np.array([f"c{i}" for i in range(30)])
        seeds = np.arange(10)
        cluster_draw, seed_draw = draw_replicate(rng, clusters, seeds)
        assert cluster_draw.shape == (30,)
        assert seed_draw.shape == (10,)  # one global set, not per-cluster

    def test_reproducible_with_rng_seed(self):
        data = _make_data(np.random.default_rng(1))
        kw = dict(**data, cond_a="control", cond_b="treated", B=50, rng_seed=7)
        r1 = joint_bootstrap_delta(**kw)
        r2 = joint_bootstrap_delta(**kw)
        assert r1.ci_low == r2.ci_low and r1.ci_high == r2.ci_high

    def test_percentile_construction(self):
        data = _make_data(np.random.default_rng(2))
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=200, rng_seed=0
        )
        lo, hi = np.quantile(r.deltas, [0.05, 0.95])
        assert r.ci_low == pytest.approx(lo)
        assert r.ci_high == pytest.approx(hi)
        assert r.to_metric_result("delta").meta["ci_construction"] == "percentile"

    def test_p_value_correction_floor(self):
        # With every delta on one side, p = 1/(B_eff + 1), never 0.
        data = _make_data(np.random.default_rng(3), treated_shift=1.8)
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=99, rng_seed=0
        )
        assert r.p_one_sided_greater == pytest.approx(1 / (r.B_effective + 1))

    def test_prereg_floor_constant(self):
        assert PREREG_MIN_B == 2000

    def test_provenance_emitted_for_recall_statistic(self):
        """§7/ARCHITECTURE: recall results carry threshold_provenance even at
        the bootstrap layer — the field exists so the semantic choice can't
        travel unlabeled."""
        data = _make_data(np.random.default_rng(10))
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=20, rng_seed=0
        )
        meta = r.to_metric_result("delta_recall").meta
        assert meta["threshold_provenance"] == "per_condition"
        assert meta["target_fpr"] == 0.01
        assert meta["statistic"] == "recall_at_fpr"

    def test_no_threshold_provenance_for_auroc(self):
        data = _make_data(np.random.default_rng(11))
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", statistic="auroc",
            B=20, rng_seed=0,
        )
        assert "threshold_provenance" not in r.to_metric_result("delta_auroc").meta

    def test_multiplicity_factorizes_cluster_times_global_seed(self):
        """The strong F27/F28 structure check: every replicate's row weights
        must factor as (cluster draw count) x (seed draw count) with ONE seed
        vector shared by all clusters and one cluster vector shared by both
        conditions and classes — the §12-banned schemes (fixed seeds,
        per-cluster seed redraws, per-condition cluster resampling) all break
        this. Row identity is smuggled through unique score values."""
        G, S = 3, 2
        rows = {"lineage_id": [], "condition": [], "label": [], "score": [], "seed": []}
        for c in range(G):
            for s in range(S):
                for cond_off, cond in ((0, "control"), (1, "treated")):
                    for lab_off, lab in ((0, "honest"), (2, "deceptive")):
                        rows["lineage_id"].append(f"c{c}")
                        rows["seed"].append(s)
                        rows["condition"].append(cond)
                        rows["label"].append(lab)
                        # unique, decodable identity: c*100 + s*10 + offsets
                        rows["score"].append(float(c * 100 + s * 10 + cond_off + lab_off))
        data = {k: np.asarray(v) for k, v in rows.items()}

        calls = []

        def recording_stat(pos, neg):
            calls.append((pos.copy(), neg.copy()))
            return 0.0

        B = 20
        joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated",
            statistic=recording_stat, B=B, rng_seed=42,
        )
        # calls: 2 for the point estimate, then 2 per replicate (cond_a, cond_b)
        assert len(calls) == 2 + 2 * B

        def counts(arr):
            m = np.zeros((G, S), dtype=int)
            for v in arr:
                base = int(v) - (int(v) % 10 % 5)  # strip cond/label offsets
                m[base // 100, (base % 100) // 10] += 1
            return m

        seed_vectors = []
        for b in range(B):
            a_pos, a_neg = calls[2 + 2 * b]
            b_pos, b_neg = calls[3 + 2 * b]
            mats = [counts(x) for x in (a_pos, a_neg, b_pos, b_neg)]
            # identical (cluster x seed) weights across conditions AND classes
            for m in mats[1:]:
                assert np.array_equal(mats[0], m)
            # exact outer-product factorization: C = outer(m_c, k_s)
            C = mats[0]
            m_c = C.sum(axis=1) // S  # row sums = m_c * sum(k_s) = m_c * S
            k_s = C.sum(axis=0) // G
            assert np.array_equal(C, np.outer(m_c, k_s))
            assert m_c.sum() == G and k_s.sum() == S
            seed_vectors.append(tuple(k_s))
        # seeds are REDRAWN across replicates (banned: held fixed)
        assert len(set(seed_vectors)) > 1


class TestBehavior:
    def test_null_ci_covers_zero(self):
        data = _make_data(np.random.default_rng(4), treated_shift=0.0)
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=300, rng_seed=1
        )
        assert r.ci_low < 0.0 < r.ci_high

    def test_true_degradation_detected(self):
        data = _make_data(np.random.default_rng(5), treated_shift=1.8, n_clusters=40)
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=300, rng_seed=1
        )
        assert r.point > 0.2  # large manufactured separation loss
        assert r.ci_low > 0.0  # one-sided significance via the CI lower bound

    def test_cluster_ci_wider_than_naive_iid(self):
        """§8 build risk: the cluster bootstrap must demonstrably diverge from
        a naive i.i.d. bootstrap under strong clustering. We emulate i.i.d. by
        relabeling every row as its own cluster."""
        data = _make_data(
            np.random.default_rng(6),
            cluster_sd=1.0,
            treated_shift=0.5,
            n_clusters=20,
            items_per_cluster=10,
        )
        clustered = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", statistic="auroc", B=200, rng_seed=2
        )
        iid = dict(data)
        iid["lineage_id"] = np.arange(len(data["lineage_id"])).astype(str)
        naive = joint_bootstrap_delta(
            **iid, cond_a="control", cond_b="treated", statistic="auroc", B=200, rng_seed=2
        )
        width_clustered = clustered.ci_high - clustered.ci_low
        width_naive = naive.ci_high - naive.ci_low
        assert width_clustered > 1.3 * width_naive

    def test_degenerate_replicates_counted(self):
        # Tiny negative class: many replicates cannot realize the 1% operating
        # point once resampling thins the negatives -> drop-and-count (§12).
        rng = np.random.default_rng(7)
        data = _make_data(rng, n_clusters=6, items_per_cluster=2, n_seeds=2)
        # keep only a sliver of honest rows to force degeneracy
        honest = np.flatnonzero(data["label"] == "honest")
        drop = honest[40:]
        keep = np.setdiff1d(np.arange(len(data["label"])), drop)
        thin = {k: v[keep] for k, v in data.items()}
        r = joint_bootstrap_delta(
            **thin, cond_a="control", cond_b="treated", B=100, rng_seed=3, target_fpr=0.05
        )
        assert r.B_degenerate > 0  # the fixture genuinely produces degeneracy
        assert r.B_effective + r.B_degenerate == r.B_requested == 100
        assert r.degenerate_fraction == pytest.approx(r.B_degenerate / 100)

    def test_unrealizable_full_data_raises(self):
        rng = np.random.default_rng(8)
        data = _make_data(rng, n_clusters=3, items_per_cluster=1, n_seeds=1)
        with pytest.raises(ValueError, match="metric ladder"):
            joint_bootstrap_delta(
                **data, cond_a="control", cond_b="treated", B=10, rng_seed=0,
                target_fpr=0.01,  # needs >= 100 negatives per condition
            )

    def test_seedless_data_allowed(self):
        # 0a scoring-level data has no seed dimension; the bootstrap must
        # degrade gracefully to a pure cluster bootstrap.
        data = _make_data(np.random.default_rng(9), n_seeds=1)
        del data["seed"]
        r = joint_bootstrap_delta(
            **data, cond_a="control", cond_b="treated", B=50, rng_seed=0
        )
        assert r.n_seeds == 1
