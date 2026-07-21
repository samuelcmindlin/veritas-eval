"""F17 power simulation — first pass on purely synthetic score distributions
(Pre-Reg §10; DEV §9 item 7). No pilot data: priors below are the pinned
presumptive synthetic values, to be re-estimated from Stage A before freeze.

Simulates the full design structure — clusters (crossed) × seeds × matched
pairs across two conditions — and runs the ACTUAL §12 estimator
(joint two-way cluster × seed percentile bootstrap, per-replicate threshold
re-estimation) inside every Monte Carlo replicate, so measured power/size
reflect the real inference procedure, not a formula.

F17 pins exercised here:
  (1) power targets:  P(L > 0 | Δ = SESOI) ≥ 0.8   (superiority leg)
                      P(U < SESOI | Δ = 0) ≥ 0.8    (equivalence leg)
  (2) generative family + pinned synthetic priors (documented in PRIORS)
  (4) seed-variance component in the simulated estimator
  (5) empirical CI-level check per leg (type-I on L, TOST-leg size on U,
      coverage of the oracle Δ)
  design effect / effective n reported per config.

Run: PYTHONPATH=src .venv/bin/python -m analysis.power
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from metrics import joint_bootstrap_delta
from provenance import config_hash, git_provenance

# Pinned presumptive synthetic priors (F17 pin 2) — re-estimate from Stage A.
# v2 (2026-07-08): first pass under-modeled two variance components and read
# equivalence power 1.00 at base floors, contradicting the stats_validation
# suite (~0.55-0.75) — see results/power_firstpass/first_pass_caveat.md. Added:
# honest-class cluster effects (threshold noise, the §10 equivalence cap) and
# a condition-noncancelling seed component (seed share of Var(Δ̂)).
PRIORS = {
    "baseline_recall_at_1pct": 0.65,  # mid dynamic-range band [0.4, 0.9] (F12)
    "sd_item": 0.5,     # matched-pair shared effect (within-pair correlation ρ)
    "sd_cluster": 0.3,  # cluster effect on deceptive separation (ICC source)
    "sd_cluster_honest": 0.3,  # cluster effect on honest scores -> threshold noise
    "sd_seed": 0.2,     # seed effect shared across conditions (cancels in Δ̂)
    "sd_seed_uncancelled": 0.15,  # per seed×condition effect (does NOT cancel)
    "sd_noise": 1.0,
    "sesoi": 0.10,      # Δrecall@1%FPR (Pre-Reg §10, F14)
    "target_fpr": 0.01,
}


def _oracle_calibrate_sep(target_recall: float, p: dict, rng: np.random.Generator,
                          n: int = 400_000) -> float:
    """Find the deceptive-class separation giving `target_recall` at the
    1%-FPR threshold of the honest distribution, under the full random-effects
    variance (numerically — analytic mapping breaks with unequal variances)."""
    e = rng.normal(scale=p["sd_item"], size=n)
    honest = (e + rng.normal(scale=p["sd_cluster_honest"], size=n)
              + rng.normal(scale=p["sd_noise"], size=n))
    threshold = np.quantile(honest, 1.0 - p["target_fpr"])
    dec_noise = (
        rng.normal(scale=p["sd_item"], size=n)
        + rng.normal(scale=p["sd_cluster"], size=n)
        + rng.normal(scale=p["sd_seed"], size=n)
        + rng.normal(scale=p["sd_seed_uncancelled"], size=n)
        + rng.normal(scale=p["sd_noise"], size=n)
    )
    lo, hi = 0.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean(dec_noise + mid > threshold) < target_recall:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_dataset(rng: np.random.Generator, *, G: int, n_pos: int, n_neg: int,
                 S: int, sep_control: float, sep_treated: float, p: dict) -> dict:
    """One synthetic study: matched pairs across two conditions, cluster and
    seed random effects on the deceptive separation, honest class shared."""
    rows = {k: [] for k in ("lineage_id", "condition", "label", "score", "seed")}
    pos_pc = max(1, n_pos // G)
    neg_pc = max(1, n_neg // G)
    u = rng.normal(scale=p["sd_cluster"], size=G)          # cluster effects (deceptive sep)
    uh = rng.normal(scale=p["sd_cluster_honest"], size=G)  # cluster effects (honest scores)
    w = rng.normal(scale=p["sd_seed"], size=S)             # seed effects (cancel in Δ̂)
    w2 = rng.normal(scale=p["sd_seed_uncancelled"], size=(S, 2))  # seed×condition
    for c in range(G):
        for kind, count, base in (("deceptive", pos_pc, None), ("honest", neg_pc, 0.0)):
            e = rng.normal(scale=p["sd_item"], size=count)  # pair-shared effects
            for ci, (cond, sep) in enumerate(
                (("control", sep_control), ("treated", sep_treated))
            ):
                for s in range(S):
                    if kind == "deceptive":
                        loc = sep + u[c] + w[s] + w2[s, ci] + e
                    else:
                        loc = uh[c] + e
                    eps = rng.normal(scale=p["sd_noise"], size=count)
                    for j in range(count):
                        rows["lineage_id"].append(f"c{c}")
                        rows["condition"].append(cond)
                        rows["label"].append(kind)
                        rows["score"].append(float(loc[j] + eps[j]))
                        rows["seed"].append(s)
    return {k: np.asarray(v) for k, v in rows.items()}


def oracle_delta(sep_control: float, sep_treated: float, p: dict,
                 rng: np.random.Generator, n: int = 400_000) -> float:
    """Population Δrecall between the two separations (per-condition 1%-FPR
    thresholds on the shared honest distribution)."""
    e = rng.normal(scale=p["sd_item"], size=n)
    honest = (e + rng.normal(scale=p["sd_cluster_honest"], size=n)
              + rng.normal(scale=p["sd_noise"], size=n))
    t = np.quantile(honest, 1.0 - p["target_fpr"])
    dec_noise = (
        rng.normal(scale=p["sd_item"], size=n)
        + rng.normal(scale=p["sd_cluster"], size=n)
        + rng.normal(scale=p["sd_seed"], size=n)
        + rng.normal(scale=p["sd_seed_uncancelled"], size=n)
        + rng.normal(scale=p["sd_noise"], size=n)
    )
    return float(np.mean(dec_noise + sep_control > t) - np.mean(dec_noise + sep_treated > t))


def empirical_design_effect(data: dict) -> float:
    """Rough deff = 1 + (m̄ − 1)·ICC from a one-way ANOVA ICC estimate on the
    control-condition deceptive scores (seed 0)."""
    mask = (data["condition"] == "control") & (data["label"] == "deceptive") & (data["seed"] == 0)
    scores, groups = data["score"][mask], data["lineage_id"][mask]
    uniq = np.unique(groups)
    means = np.array([scores[groups == g].mean() for g in uniq])
    m = len(scores) / len(uniq)
    var_between = np.var(means, ddof=1)
    var_within = np.mean([np.var(scores[groups == g], ddof=1) for g in uniq])
    icc = max(0.0, (var_between - var_within / m) / (var_between + (m - 1) * var_within / m + 1e-12))
    return 1.0 + (m - 1) * icc


def run_config(name: str, *, G: int, n_pos: int, n_neg: int, S: int,
               n_sims: int, B: int, p: dict, master_seed: int) -> dict:
    rng = np.random.default_rng(master_seed)
    sep0 = _oracle_calibrate_sep(p["baseline_recall_at_1pct"], p, rng)
    sep1 = _oracle_calibrate_sep(p["baseline_recall_at_1pct"] - p["sesoi"], p, rng)
    delta_true = oracle_delta(sep0, sep1, p, rng)

    legs = {"superiority": (sep0, sep1), "null": (sep0, sep0)}
    counts = {k: {"L_gt_0": 0, "U_lt_sesoi": 0, "covers_oracle": 0, "n": 0,
                  "U_lt_0": 0}
              for k in legs}
    deffs = []
    for leg, (sc, st) in legs.items():
        oracle = delta_true if leg == "superiority" else 0.0
        for i in range(n_sims):
            data = make_dataset(rng, G=G, n_pos=n_pos, n_neg=n_neg, S=S,
                                sep_control=sc, sep_treated=st, p=p)
            if i == 0:
                deffs.append(empirical_design_effect(data))
            r = joint_bootstrap_delta(
                **data, cond_a="control", cond_b="treated",
                statistic="recall_at_fpr", target_fpr=p["target_fpr"],
                B=B, rng_seed=int(rng.integers(2**32)),
            )
            c = counts[leg]
            c["n"] += 1
            c["L_gt_0"] += r.ci_low > 0
            c["U_lt_sesoi"] += r.ci_high < p["sesoi"]
            c["U_lt_0"] += r.ci_high < 0
            c["covers_oracle"] += r.ci_low <= oracle <= r.ci_high

    sup, nul = counts["superiority"], counts["null"]
    deff = float(np.mean(deffs))
    return {
        "config": {"name": name, "G": G, "n_pos": n_pos, "n_neg": n_neg,
                   "S": S, "n_sims": n_sims, "B": B},
        "calibration": {"sep_control": sep0, "sep_treated": sep1,
                        "oracle_delta": delta_true},
        "superiority_power_P(L>0|SESOI)": sup["L_gt_0"] / sup["n"],
        "equivalence_power_P(U<SESOI|0)": nul["U_lt_sesoi"] / nul["n"],
        "type_I_P(L>0|0)": nul["L_gt_0"] / nul["n"],
        "tost_leg_size_P(U<0|0)": nul["U_lt_0"] / nul["n"],
        "coverage_at_SESOI": sup["covers_oracle"] / sup["n"],
        "coverage_at_null": nul["covers_oracle"] / nul["n"],
        "design_effect": deff,
        "effective_n_pos_per_cond": round(n_pos / deff),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/power_firstpass.yaml",
                    type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    p = {**PRIORS, **cfg.get("priors", {})}

    results = []
    for c in cfg["configs"]:
        print(f"[power] running {c['name']} ...", flush=True)
        results.append(run_config(p=p, master_seed=cfg["master_seed"], **c))
        print(json.dumps(results[-1], indent=1), flush=True)

    report = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            **git_provenance(),
            "config_path": str(args.config),
            "config_hash": config_hash(cfg),
            "master_seed": cfg["master_seed"],
            "priors": p,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "note": ("first pass on pinned synthetic priors (F17 pin 2); "
                     "re-estimate priors from Stage A before freeze; B below "
                     "the confirmatory 2000 floor is a first-pass economy — "
                     "rerun at B=2000 for the freeze-deposited computation"),
        },
        "results": results,
    }
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))

    lines = ["# F17 power simulation — first pass (synthetic priors)", "",
             f"Generated {report['provenance']['generated_utc']} | SHA "
             f"`{report['provenance']['our_git_sha'][:12]}` | priors: "
             f"baseline recall {p['baseline_recall_at_1pct']}, SESOI {p['sesoi']}",
             "",
             "| config | G | n_pos | n_neg | S | sup. power | equiv. power | "
             "type-I | TOST size | deff |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["config"]
        lines.append(
            f"| {c['name']} | {c['G']} | {c['n_pos']} | {c['n_neg']} | {c['S']} "
            f"| {r['superiority_power_P(L>0|SESOI)']:.2f} "
            f"| {r['equivalence_power_P(U<SESOI|0)']:.2f} "
            f"| {r['type_I_P(L>0|0)']:.3f} | {r['tost_leg_size_P(U<0|0)']:.3f} "
            f"| {r['design_effect']:.2f} |"
        )
    lines += ["", "Targets (F17 pin 1): superiority ≥ 0.8 at the design point; "
              "equivalence ≥ 0.8 required for the §13 Robust row (expected to "
              "FAIL at base floors per §10 — that expectation is itself under test).",
              "", "Reproduce: `PYTHONPATH=src .venv/bin/python -m analysis.power`"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print((out / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
