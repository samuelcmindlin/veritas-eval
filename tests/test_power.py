"""Smoke tests for the F17 power simulation machinery."""
import numpy as np
import pytest

from analysis.power import (
    PRIORS,
    _oracle_calibrate_sep,
    make_dataset,
    oracle_delta,
    run_config,
)


def test_oracle_calibration_hits_target_recall():
    p = dict(PRIORS)
    rng = np.random.default_rng(0)
    sep = _oracle_calibrate_sep(0.65, p, rng)
    # verify on a fresh oracle sample (mirror every PRIORS variance component)
    e = rng.normal(scale=p["sd_item"], size=200_000)
    honest = (e + rng.normal(scale=p["sd_cluster_honest"], size=200_000)
              + rng.normal(scale=p["sd_noise"], size=200_000))
    t = np.quantile(honest, 0.99)
    dec = (
        rng.normal(scale=p["sd_item"], size=200_000)
        + rng.normal(scale=p["sd_cluster"], size=200_000)
        + rng.normal(scale=p["sd_seed"], size=200_000)
        + rng.normal(scale=p["sd_seed_uncancelled"], size=200_000)
        + rng.normal(scale=p["sd_noise"], size=200_000)
        + sep
    )
    assert np.mean(dec > t) == pytest.approx(0.65, abs=0.02)


def test_oracle_delta_zero_when_separations_equal():
    p = dict(PRIORS)
    rng = np.random.default_rng(1)
    assert oracle_delta(2.5, 2.5, p, rng) == 0.0


def test_dataset_structure_and_pairing():
    p = dict(PRIORS)
    data = make_dataset(np.random.default_rng(2), G=4, n_pos=8, n_neg=16, S=2,
                        sep_control=2.5, sep_treated=2.0, p=p)
    n = len(data["score"])
    # G * (pos_pc + neg_pc) * 2 conditions * S seeds
    assert n == 4 * (2 + 4) * 2 * 2
    assert set(data["condition"]) == {"control", "treated"}
    assert set(data["label"]) == {"honest", "deceptive"}
    assert len(set(data["lineage_id"])) == 4


def test_run_config_smoke():
    p = dict(PRIORS)
    r = run_config("smoke", G=8, n_pos=40, n_neg=240, S=2, n_sims=3, B=60,
                   p=p, master_seed=3)
    for key in ("superiority_power_P(L>0|SESOI)", "equivalence_power_P(U<SESOI|0)",
                "type_I_P(L>0|0)", "coverage_at_null", "design_effect"):
        assert key in r
    assert 0.0 <= r["superiority_power_P(L>0|SESOI)"] <= 1.0
    assert r["design_effect"] >= 1.0
    assert r["calibration"]["oracle_delta"] == pytest.approx(0.10, abs=0.03)
