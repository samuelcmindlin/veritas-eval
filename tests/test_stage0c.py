"""Smoke test for the Stage 0c anchor qualification (skips without the clone)."""
from pathlib import Path

import pytest

REPO = Path("external/LLM-LieDetector")


@pytest.mark.skipif(not REPO.exists(), reason="external LLM-LieDetector clone not present")
def test_stage0c_reproduces_published_aucs():
    from analysis.stage0c import run

    report = run()
    assert report["qualification"]["pass"]
    assert report["qualification"]["worst_abs_diff"] < 2e-3
    # every configuration clears the Pre-Reg §8 qualification floor
    assert all(r["above_qualification_floor"] for r in report["rows"])
    # the headline 48-probe subset is present and essentially exact
    union = [r for r in report["rows"]
             if r["probe_group"] == "subsets_union" and r["feature_type"] == "logprobs"]
    assert union and union[0]["abs_diff"] < 1e-4
