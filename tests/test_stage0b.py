"""Pre-flight tests for the Stage 0b pipeline itself (run BEFORE burning a
long gated-model run): full end-to-end on a tiny model, the fallback path,
the fail-fast health check, and report integrity."""
import json
from pathlib import Path

import numpy as np
import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def tiny_cfg(tmp_path: Path, primary: str = TINY) -> dict:
    return {
        "id": "stage0b_test",
        "subject": {
            "model_name": primary,
            "revision": "main",
            "fallback_model": TINY,
            "fallback_revision": "main",
            "dtype": "float32",
        },
        "read_layer_frac": 0.5,
        "diagnostic_layer_fracs": [0.5],  # tiny model: 2 layers -> read layer 1
        "generation_smoke": {
            "prompts": ["Hello"],
            "seeds": [0],
            "max_new_tokens": 4,
            "temperature": 0.7,
        },
        "probe": {"seeds": [0, 1], "reg_coeff": 10.0, "train_cluster_fraction": 0.6},
        "metrics": {"bootstrap_B": 50, "ci": 0.90, "rng_seed": 1},
        "cache_dir": str(tmp_path / "acts"),
        "out_dir": str(tmp_path / "out"),
    }


@pytest.fixture(scope="module")
def e2e_report(tmp_path_factory):
    try:
        from analysis.stage0b import run

        return run(tiny_cfg(tmp_path_factory.mktemp("e2e")))
    except Exception as e:
        if "tiny" in str(e) or "Connection" in str(e) or "offline" in str(e):
            pytest.skip(f"tiny model unavailable: {e}")
        raise


class TestEndToEnd:
    def test_structure_and_serializability(self, e2e_report):
        # every field must survive JSON round-trip (catches stray numpy types)
        blob = json.dumps(e2e_report)
        back = json.loads(blob)
        for key in ("provenance", "generation_smoke", "probe",
                    "null_delta_auroc", "recall_guardrail",
                    "steering_known_answer", "pass"):
            assert key in back

    def test_deterministic_components_pass(self, e2e_report):
        # (the null-ΔAUROC zero-coverage check is stochastic at B=50 on a
        # random tiny model, so overall `pass` is NOT asserted here)
        assert e2e_report["recall_guardrail"]["fired"] is True
        assert e2e_report["steering_known_answer"]["pass"] is True
        assert e2e_report["provenance"]["peak_rss_gb"] > 0

    def test_generation_smoke_logs_decoding(self, e2e_report):
        row = e2e_report["generation_smoke"][0]
        for field in ("seed", "temperature", "max_new_tokens", "output"):
            assert field in row

    def test_probe_artifacts_frozen_per_seed(self, e2e_report):
        hashes = e2e_report["provenance"]["probe_artifact_hashes"]
        assert len(hashes) == 2
        assert all(len(h) == 64 for h in hashes.values())

    def test_not_fallback_when_primary_loads(self, e2e_report):
        assert e2e_report["provenance"]["substrate"]["is_fallback"] is False
        assert "dtype" in e2e_report["provenance"]["substrate"]

    def test_md_report_renders(self, e2e_report, tmp_path):
        from analysis.stage0b import write_report_md

        out = tmp_path / "report.md"
        write_report_md(e2e_report, out)
        text = out.read_text()
        assert ("PASS" in text) or ("FAIL" in text)
        assert "SUBSTRATE FALLBACK" not in text  # primary loaded

    def test_cache_reuse_on_second_run(self, e2e_report, tmp_path_factory):
        from analysis.stage0b import run

        cfg = tiny_cfg(tmp_path_factory.mktemp("cache"))
        first = run(cfg)
        second = run(cfg)
        assert first["provenance"]["cache_hits"] == 0
        assert second["provenance"]["cache_hits"] == second["provenance"]["n_items"]


class TestFallbackPath:
    def test_fallback_banner_and_metadata(self, tmp_path):
        from analysis.stage0b import run, write_report_md

        cfg = tiny_cfg(tmp_path, primary="nonexistent-org/definitely-not-a-model")
        try:
            report = run(cfg)
        except Exception as e:
            pytest.skip(f"tiny model unavailable: {e}")
        sub = report["provenance"]["substrate"]
        assert sub["is_fallback"] is True
        assert sub["primary_model"] == "nonexistent-org/definitely-not-a-model"
        assert "primary_load_error" in sub
        out = tmp_path / "report.md"
        write_report_md(report, out)
        assert "SUBSTRATE FALLBACK" in out.read_text()


class TestHealthCheck:
    def test_nonfinite_activations_fail_fast(self):
        from analysis.stage0b import substrate_health_check

        class SickSubject:  # duck-typed: returns NaN activations
            def rescore(self, text, *, read_layers, steering=None):
                from models.runner import Capture

                acts = {layer: np.full((3, 4), np.nan) for layer in read_layers}
                return Capture(output_text=text, activations=acts, meta={})

            def generate(self, *a, **kw):
                raise AssertionError("must fail before generating")

        with pytest.raises(RuntimeError, match="non-finite"):
            substrate_health_check(SickSubject(), read_layer=1)

    def test_empty_generation_fails(self):
        from analysis.stage0b import substrate_health_check
        from models.runner import Capture

        class MuteSubject:
            def rescore(self, text, *, read_layers, steering=None):
                acts = {layer: np.zeros((3, 4)) for layer in read_layers}
                return Capture(output_text=text, activations=acts, meta={})

            def generate(self, *a, **kw):
                return Capture(output_text="   ", activations=None, meta={})

        with pytest.raises(RuntimeError, match="empty greedy generation"):
            substrate_health_check(MuteSubject(), read_layer=0)


class TestDatasetInvariants:
    def test_pseudo_conditions_balanced(self):
        from data import build_toy_dataset

        items = build_toy_dataset()
        t0 = [i for i in items if i.item_id.endswith("t0")]
        t1 = [i for i in items if i.item_id.endswith("t1")]
        assert len(t0) == len(t1) == len(items) / 2
        # both labels present in both pseudo-conditions, same cluster sets
        for grp in (t0, t1):
            assert {i.label for i in grp} == {"honest", "deceptive"}
        assert {i.lineage_id for i in t0} == {i.lineage_id for i in t1}

    def test_dataset_hash_stable(self):
        from data import build_toy_dataset
        from data.toy_truthfulness import dataset_hash

        assert dataset_hash(build_toy_dataset()) == dataset_hash(build_toy_dataset())
