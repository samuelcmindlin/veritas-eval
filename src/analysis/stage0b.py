"""Stage 0b — PIPELINE VALIDATION at small-model scale (DEV plan §4/§9).

Chains one config-driven run: generation (decoding params + seed logged) →
streaming residual-stream extraction (bounded RAM) → per-seed frozen toy probe
→ metrics with the joint two-way bootstrap → provenance-tagged report. Also
runs the two §8 steering known-answer tests on the loaded subject.

This validates the STACK. It is not a baseline reproduction, and the toy
true/false dataset is not a deception organism. Known-answer expectations:
- The two template pseudo-conditions differ by nothing real → the ΔAUROC
  joint-bootstrap CI should cover 0.
- recall@1%FPR is unrealizable at toy scale (~24 negatives/condition) → the
  §10 guardrails (metric-ladder error / granularity gate) must FIRE.
- Same-site constant steering shifts a mean-agg linear score by exactly
  λ·(w·v); injection strictly above the read layer leaves read-layer
  activations bit-identical (Pre-Reg §4 [FIX2-1] mechanics).

Run: PYTHONPATH=src .venv/bin/python -m analysis.stage0b
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from data import build_toy_dataset
from data.toy_truthfulness import dataset_hash
from detectors import MeanAggLogisticProbe
from metrics import auroc, joint_bootstrap_delta
from models import HFSubject
from models.runner import ActivationCache, SteeringHook
from provenance import config_hash, git_provenance


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS


def resolve_revision(model_name: str, revision: str) -> str:
    """Pin `main` to a commit SHA for provenance (public metadata endpoint)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_name, revision=revision).sha or revision
    except Exception:
        return f"{revision} (unresolved — metadata unavailable)"


def load_subject_with_fallback(scfg: dict) -> tuple[HFSubject, dict]:
    import torch

    dtype = getattr(torch, scfg.get("dtype", "float32"))
    try:
        rev = resolve_revision(scfg["model_name"], scfg["revision"])
        subj = HFSubject(scfg["model_name"], rev, dtype=dtype)
        return subj, {"substrate": scfg["model_name"], "revision": rev,
                      "dtype": str(dtype), "is_fallback": False}
    except Exception as e:
        rev = resolve_revision(scfg["fallback_model"], scfg["fallback_revision"])
        subj = HFSubject(scfg["fallback_model"], rev, dtype=dtype)
        return subj, {
            "substrate": scfg["fallback_model"], "revision": rev,
            "dtype": str(dtype), "is_fallback": True,
            "primary_model": scfg["model_name"],
            "primary_load_error": f"{type(e).__name__}: {str(e)[:300]}",
        }


# ---------- known-answer steering tests (Pre-Reg §4/§8, DEV §8) ----------

def steering_known_answer_tests(subject: HFSubject, read_layer: int,
                                rng: np.random.Generator) -> dict:
    import torch

    text = "The quick brown fox jumps over the lazy dog near the river bank."
    d = subject.model.config.hidden_size
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)

    clean = subject.rescore(text, read_layers=(read_layer,))
    clean_act = clean.activations[read_layer]
    clean_score = float(clean_act.mean(0) @ w)

    # Scale v to the residual stream's own magnitude so the predicted shift
    # dominates reduced-precision rounding (in fp16/bf16 a small-norm shift
    # would drown in per-element rounding of the large hidden values).
    hidden_scale = float(np.linalg.norm(clean_act, axis=1).mean())
    v = rng.normal(size=d).astype(np.float32)
    v *= hidden_scale / np.linalg.norm(v)

    model_dtype = next(subject.model.parameters()).dtype
    # rel tolerance by compute precision: fp32 is tight; half precisions
    # accumulate per-element rounding through the addition itself.
    rel_tol = 1e-3 if model_dtype == torch.float32 else 2e-2

    out: dict = {"model_dtype": str(model_dtype), "same_site_rel_tol": rel_tol}

    # (1) same-site nulling arithmetic: score shift == λ·(w·v_cast), where
    # v_cast is the vector as the hook actually adds it (cast to model dtype)
    lam = 0.5
    steered = subject.rescore(
        text, read_layers=(read_layer,),
        steering=[SteeringHook(layer=read_layer, vector=v, coefficient=lam)],
    )
    shift = float(steered.activations[read_layer].mean(0) @ w) - clean_score
    v_cast = torch.as_tensor(v, dtype=model_dtype).to(torch.float64).numpy()
    predicted = lam * float(w @ v_cast)
    rel_err = abs(shift - predicted) / max(abs(predicted), 1e-9)
    out["same_site_shift"] = {
        "observed": shift, "predicted_lambda_w_dot_v": predicted,
        "rel_err": rel_err,
        "pass": bool(rel_err < rel_tol),
    }

    # (2) injection strictly ABOVE the read layer: read-layer activations must
    # be bit-identical to the unsteered pass (layer-ordering unit test,
    # Pre-Reg §4 control 5 / [FIX2-1]). Read/inject sites are clamped so the
    # test is well-defined on shallow (test-scale) models too.
    ra = min(read_layer, subject.n_layers - 2)
    clean_ra = subject.rescore(text, read_layers=(ra,)).activations[ra]
    above = subject.rescore(
        text, read_layers=(ra,),
        steering=[SteeringHook(layer=ra + 1, vector=v, coefficient=10.0)],
    )
    identical = bool(np.array_equal(above.activations[ra], clean_ra))
    out["above_read_bit_identity"] = {
        "read_layer": ra, "inject_layer": ra + 1,
        "pass": identical,
        "max_abs_diff": float(np.max(np.abs(above.activations[ra] - clean_ra))),
    }

    # (3) sanity: injection BELOW the read layer must reach it (additive carry)
    rb = max(read_layer, 1)
    clean_rb = subject.rescore(text, read_layers=(rb,)).activations[rb]
    below = subject.rescore(
        text, read_layers=(rb,),
        steering=[SteeringHook(layer=rb - 1, vector=v, coefficient=0.5)],
    )
    out["below_read_carry"] = {
        "read_layer": rb, "inject_layer": rb - 1,
        "pass": bool(not np.array_equal(below.activations[rb], clean_rb)),
    }
    out["pass"] = all(t["pass"] for t in out.values() if isinstance(t, dict))
    return out


# ---------- the pipeline run ----------

def substrate_health_check(subject: HFSubject, read_layer: int) -> None:
    """Fail FAST (before any long loop) if the substrate is numerically sick —
    e.g. Gemma-2's logit soft-capping is known to overflow in float16. Checks
    one teacher-forced pass for finite activations and one greedy generation
    for non-empty output."""
    cap = subject.rescore("Paris is the capital of France.", read_layers=(read_layer,))
    acts = cap.activations[read_layer]
    if not np.all(np.isfinite(acts)):
        raise RuntimeError(
            f"substrate health check FAILED: non-finite activations at layer "
            f"{read_layer} ({np.isnan(acts).sum()} NaN / {np.isinf(acts).sum()} inf) "
            f"— likely a dtype issue (Gemma-2 requires bfloat16/float32, not float16)"
        )
    gen = subject.generate("The opposite of hot is", seed=0, max_new_tokens=4,
                           do_sample=False)
    if not gen.output_text.strip():
        raise RuntimeError("substrate health check FAILED: empty greedy generation")


def run(cfg: dict) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    subject, substrate = load_subject_with_fallback(cfg["subject"])
    n_layers = subject.n_layers
    read_layer = round(cfg["read_layer_frac"] * n_layers)
    diag_layers = [round(f * n_layers) for f in cfg["diagnostic_layer_fracs"]]
    layers = tuple(sorted({read_layer, *diag_layers}))

    # 0) fail-fast substrate health check (one item, before any long loop)
    substrate_health_check(subject, read_layer)

    # 1) generation smoke: decoding params + seed logged (DEV §9 item 2)
    gen_rows = []
    g = cfg["generation_smoke"]
    for prompt in g["prompts"]:
        for seed in g["seeds"]:
            cap = subject.generate(prompt, seed=seed,
                                   max_new_tokens=g["max_new_tokens"],
                                   temperature=g["temperature"])
            gen_rows.append({"prompt": prompt, **cap.meta["decoding"],
                             "output": cap.output_text})

    # 2) streaming extraction over the toy dataset (bounded RAM)
    items = build_toy_dataset()
    cache = ActivationCache(Path(cfg["cache_dir"]))
    cache_hits = 0
    mean_acts: dict[str, np.ndarray] = {}
    for it in items:
        key = cache.key(subject.id, it.item_id, "pipeline_validation", layers)
        acts = cache.get(key)
        if acts is None:
            acts = subject.rescore(it.text, read_layers=layers).activations
            if not all(np.all(np.isfinite(a)) for a in acts.values()):
                raise RuntimeError(
                    f"non-finite activations for item {it.item_id!r} — aborting "
                    "extraction (fail fast; check substrate dtype)"
                )
            cache.put(key, acts)
        else:
            cache_hits += 1
        # only the mean at the read layer is held in RAM (invariant #4)
        mean_acts[it.item_id] = acts[read_layer].mean(0)

    # 3) cluster split + per-seed frozen probes
    clusters = sorted({it.lineage_id for it in items})
    rng = np.random.default_rng(cfg["metrics"]["rng_seed"])
    perm = rng.permutation(clusters)
    n_train = int(cfg["probe"]["train_cluster_fraction"] * len(clusters))
    train_c, test_c = set(perm[:n_train]), set(perm[n_train:])

    train_items = [it for it in items if it.lineage_id in train_c]
    test_items = [it for it in items if it.lineage_id in test_c]
    x_train = np.stack([mean_acts[it.item_id] for it in train_items])
    y_train = np.array([1.0 if it.label == "deceptive" else 0.0 for it in train_items])

    probe_impl = MeanAggLogisticProbe(read_layer, cfg["probe"]["reg_coeff"])
    rows = {"lineage_id": [], "condition": [], "label": [], "score": [], "seed": []}
    probe_hashes = {}
    per_seed_auroc = {}
    for seed in cfg["probe"]["seeds"]:
        frozen = probe_impl.fit(x_train, y_train, seed=seed)
        probe_hashes[seed] = frozen.artifact_hash
        scores = {it.item_id: frozen.score(mean_acts[it.item_id]) for it in test_items}
        pos = np.array([scores[i.item_id] for i in test_items if i.label == "deceptive"])
        neg = np.array([scores[i.item_id] for i in test_items if i.label == "honest"])
        per_seed_auroc[seed] = auroc(pos, neg)
        for it in test_items:
            rows["lineage_id"].append(it.lineage_id)
            # pseudo-conditions = the two templates; no real difference exists,
            # so the ΔAUROC CI covering 0 is a known-answer pipeline check
            rows["condition"].append("tpl0" if it.item_id.endswith("t0") else "tpl1")
            rows["label"].append(it.label)
            rows["score"].append(scores[it.item_id])
            rows["seed"].append(seed)
    data = {k: np.asarray(v) for k, v in rows.items()}

    # 4) metrics through the joint two-way bootstrap
    m = cfg["metrics"]
    null_delta = joint_bootstrap_delta(
        **data, cond_a="tpl0", cond_b="tpl1", statistic="auroc",
        B=m["bootstrap_B"], ci=m["ci"], rng_seed=m["rng_seed"],
    )
    # guardrail known-answer: the 1%-FPR operating point must be unrealizable
    try:
        joint_bootstrap_delta(**data, cond_a="tpl0", cond_b="tpl1",
                              statistic="recall_at_fpr", target_fpr=0.01,
                              B=10, rng_seed=0)
        guardrail = {"fired": False}
    except ValueError as e:
        guardrail = {"fired": True, "message": str(e)}

    # 5) steering known-answer tests
    ka = steering_known_answer_tests(subject, read_layer, np.random.default_rng(0))

    report = {
        "provenance": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **git_provenance(),
            "config_hash": config_hash(cfg),
            "config_id": cfg["id"],
            "substrate": substrate,
            "n_layers": n_layers,
            "read_layer": read_layer,
            "diagnostic_layers": diag_layers,
            "dataset_hash": dataset_hash(items),
            "n_items": len(items),
            "probe_artifact_hashes": {str(k): v for k, v in probe_hashes.items()},
            "peak_rss_gb": round(_peak_rss_gb(), 3),
            "cache_dir": cfg["cache_dir"],
            "cache_hits": cache_hits,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "generation_smoke": gen_rows,
        "probe": {
            "per_seed_test_auroc": {str(k): v for k, v in per_seed_auroc.items()},
            "n_train_items": len(train_items),
            "n_test_items": len(test_items),
            "n_train_clusters": len(train_c),
            "n_test_clusters": len(test_c),
        },
        "null_delta_auroc": {
            "point": null_delta.point,
            "ci_low": null_delta.ci_low,
            "ci_high": null_delta.ci_high,
            "covers_zero_pass": bool(null_delta.ci_low <= 0.0 <= null_delta.ci_high),
            "B": null_delta.B_requested,
        },
        "recall_guardrail": {**guardrail, "pass": guardrail["fired"]},
        "steering_known_answer": ka,
    }
    report["pass"] = bool(
        report["null_delta_auroc"]["covers_zero_pass"]
        and report["recall_guardrail"]["pass"]
        and ka["pass"]
    )
    return report


def write_report_md(r: dict, out: Path) -> None:
    p = r["provenance"]
    sub = p["substrate"]
    ka = r["steering_known_answer"]
    fallback_banner = (
        "\n> **SUBSTRATE FALLBACK.** The primary model "
        f"`{sub.get('primary_model')}` could not be loaded "
        f"(`{sub.get('primary_load_error', '')[:120]}…`). This run validates the "
        "stack on the fallback substrate; re-run on Gemma-2-2B after `hf auth "
        "login` before treating 0b as complete.\n"
        if sub["is_fallback"] else ""
    )
    lines = [
        "# Stage 0b — pipeline validation (activation → probe → metrics)",
        "",
        f"Generated {p['finished_utc']} | our SHA `{p['our_git_sha'][:12]}` | "
        f"config `{p['config_hash'][:12]}` | **{'PASS' if r['pass'] else 'FAIL'}**",
        fallback_banner,
        f"**Substrate:** `{sub['substrate']}` @ `{str(sub['revision'])[:12]}` "
        f"({p['n_layers']} layers; probe read layer {p['read_layer']}; "
        f"diagnostics {p['diagnostic_layers']}). Pipeline validation only — "
        "not a baseline reproduction (DEV §4 0b).",
        "",
        f"- Generation smoke: {len(r['generation_smoke'])} generations, decoding "
        "params + seed logged per row (see report.json).",
        f"- Extraction: {p['n_items']} items streamed to `{p['cache_dir']}` "
        f"({p['cache_hits']} cache hits); peak RSS {p['peak_rss_gb']} GB (bounded-RAM check).",
        f"- Toy probe ({r['probe']['n_train_clusters']} train / "
        f"{r['probe']['n_test_clusters']} test clusters), per-seed held-out AUROC: "
        + ", ".join(f"s{k}={v:.3f}" for k, v in r["probe"]["per_seed_test_auroc"].items()),
        f"- Null ΔAUROC between template pseudo-conditions: {r['null_delta_auroc']['point']:+.4f} "
        f"[{r['null_delta_auroc']['ci_low']:+.4f}, {r['null_delta_auroc']['ci_high']:+.4f}] — "
        f"covers 0: {r['null_delta_auroc']['covers_zero_pass']} (known-answer).",
        f"- recall@1%FPR guardrail at toy scale: fired = {r['recall_guardrail']['fired']} "
        "(expected True — §10 metric-ladder trigger).",
        "- Steering known-answer tests: "
        f"same-site shift rel_err={ka['same_site_shift']['rel_err']:.2e} "
        f"(pass={ka['same_site_shift']['pass']}); "
        f"above-read bit-identity pass={ka['above_read_bit_identity']['pass']} "
        f"(max|diff|={ka['above_read_bit_identity']['max_abs_diff']:g}); "
        f"below-read carry pass={ka['below_read_carry']['pass']}.",
        "",
        f"Probe artifact hashes (frozen per seed): "
        + ", ".join(f"s{k}=`{v[:10]}`" for k, v in p["probe_artifact_hashes"].items()),
        f"Dataset hash: `{p['dataset_hash'][:16]}` ({p['n_items']} toy items — "
        "NOT a deception organism).",
        "",
        "Reproduce: `PYTHONPATH=src .venv/bin/python -m analysis.stage0b`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/stage0b.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    report = run(cfg)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    write_report_md(report, out / "report.md")
    print((out / "report.md").read_text())
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
