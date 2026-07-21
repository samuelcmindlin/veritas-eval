"""Smoke test for the Stage 0a reproduction (skips without the external clone)."""
from pathlib import Path

import pytest

EXP = Path("external/deception-detection/example_results/roleplaying")


@pytest.mark.skipif(not EXP.exists(), reason="external Apollo clone not present")
def test_stage0a_reproduces_shipped_table():
    from analysis.stage0a import run

    report = run(EXP)
    assert report["reproduction"]["pass"]
    assert report["reproduction"]["max_abs_diff_vs_shipped"] < 1e-6
    # the known lossy-label recovery: '0.2%' is really FPR 0.0025 in this run
    assert report["reproduction"]["fpr_inference"]["0.2%"]["inferred_fpr"] == 0.0025
    # threshold provenance must be emitted for the F30 frozen threshold
    assert report["discrepancy_panel"]["frozen_threshold_provenance"].startswith("frozen@")
