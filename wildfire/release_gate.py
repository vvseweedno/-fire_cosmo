"""Machine-checkable engineering proof gate for the competition release."""

from __future__ import annotations

import math
from typing import Any

METRIC_KEYS = ("f1_af", "iou_burn", "miou_severity", "score")


def official_score(metrics: dict[str, Any]) -> float:
    f1 = float(metrics["f1_af"])
    iou = float(metrics["iou_burn"])
    miou = float(metrics["miou_severity"])
    values = (f1, iou, miou)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("official metrics must be finite")
    return 0.35 * f1 + 0.35 * iou + 0.30 * miou


def _metrics_ok(metrics: object, *, score_tolerance: float) -> tuple[bool, str]:
    if not isinstance(metrics, dict):
        return False, "metrics object is missing"
    try:
        computed = official_score(metrics)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid metrics: {exc}"
    reported = metrics.get("score")
    if reported is None:
        return False, "reported score is missing"
    try:
        reported_float = float(reported)
    except (TypeError, ValueError):
        return False, "reported score is not numeric"
    if not math.isfinite(reported_float):
        return False, "reported score is not finite"
    if abs(reported_float - computed) > score_tolerance:
        return (
            False,
            f"score formula mismatch: reported={reported_float:.12f}, "
            f"computed={computed:.12f}",
        )
    return True, "official metric triplet exists and score formula matches"


def evaluate_proven_release(
    evidence: dict[str, Any],
    *,
    score_epsilon: float = 1e-4,
    score_tolerance: float = 1e-10,
    repro_tolerance: float = 1e-8,
    min_bootstrap_replicates: int = 1000,
    min_probability_positive: float = 0.95,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    ci = evidence.get("ci")
    ci_green = bool(isinstance(ci, dict) and ci.get("green") is True)
    checks["ci_green"] = {
        "pass": ci_green,
        "detail": "CI is explicitly green" if ci_green else "CI green evidence missing/false",
    }

    preflight = evidence.get("preflight")
    train_ok = bool(
        isinstance(preflight, dict)
        and isinstance(preflight.get("train"), dict)
        and preflight["train"].get("ok") is True
    )
    test_ok = bool(
        isinstance(preflight, dict)
        and isinstance(preflight.get("test"), dict)
        and preflight["test"].get("ok") is True
    )
    checks["official_preflight"] = {
        "pass": train_ok and test_ok,
        "detail": f"train={train_ok}, test={test_ok}",
    }

    leakage = evidence.get("leakage_audit")
    strict_grouping = bool(
        isinstance(leakage, dict)
        and leakage.get("strict_event_grouping") is True
        and int(leakage.get("chip_fallbacks", -1)) == 0
        and float(leakage.get("event_group_coverage", 0.0)) >= 1.0
    )
    checks["strict_leakage_grouping"] = {
        "pass": strict_grouping,
        "detail": (
            "all chips use organiser event/group ids"
            if strict_grouping
            else f"strict organiser grouping unavailable: {leakage}"
        ),
    }

    full_leakage_pass = bool(
        isinstance(leakage, dict)
        and leakage.get("pass") is True
        and str(leakage.get("status", "")).upper() == "PASS"
    )
    checks["leakage_audit_pass"] = {
        "pass": full_leakage_pass,
        "detail": (
            "explicit leakage audit is PASS"
            if full_leakage_pass
            else f"explicit leakage audit did not pass: {leakage}"
        ),
    }

    crossfit = evidence.get("crossfit")
    baseline = crossfit.get("baseline") if isinstance(crossfit, dict) else None
    final = crossfit.get("final") if isinstance(crossfit, dict) else None
    base_ok, base_detail = _metrics_ok(baseline, score_tolerance=score_tolerance)
    final_ok, final_detail = _metrics_ok(final, score_tolerance=score_tolerance)
    checks["crossfit_metrics_exist"] = {
        "pass": base_ok and final_ok,
        "detail": f"baseline: {base_detail}; final: {final_detail}",
    }

    score_gain_ok = False
    delta_score = None
    if base_ok and final_ok and isinstance(baseline, dict) and isinstance(final, dict):
        baseline_score = official_score(baseline)
        final_score = official_score(final)
        delta_score = final_score - baseline_score
        score_gain_ok = delta_score > score_epsilon
    checks["final_score_above_baseline"] = {
        "pass": score_gain_ok,
        "detail": (
            f"delta_score={delta_score:.12f}, epsilon={score_epsilon:.12f}"
            if delta_score is not None
            else "score delta unavailable"
        ),
    }

    bootstrap = evidence.get("bootstrap")
    bootstrap_ok = False
    bootstrap_detail = "bootstrap evidence missing"
    if isinstance(bootstrap, dict):
        try:
            used = int(bootstrap.get("n_boot_used", 0))
            ci95 = bootstrap.get("bootstrap_95_ci")
            probability = float(bootstrap.get("probability_delta_positive"))
            score_a = float(bootstrap.get("score_a"))
            score_b = float(bootstrap.get("score_b"))
            bootstrap_delta = float(bootstrap.get("delta_score"))
            if not isinstance(ci95, (list, tuple)) or len(ci95) != 2:
                raise ValueError("bootstrap_95_ci must contain two values")
            lo = float(ci95[0])
            hi = float(ci95[1])

            metric_consistency = False
            if (
                base_ok
                and final_ok
                and isinstance(baseline, dict)
                and isinstance(final, dict)
            ):
                expected_final = official_score(final)
                expected_baseline = official_score(baseline)
                expected_delta = expected_final - expected_baseline
                metric_consistency = (
                    abs(score_a - expected_final) <= score_tolerance
                    and abs(score_b - expected_baseline) <= score_tolerance
                    and abs(bootstrap_delta - expected_delta) <= score_tolerance
                )

            bootstrap_ok = (
                used >= min_bootstrap_replicates
                and math.isfinite(lo)
                and math.isfinite(hi)
                and lo > 0.0
                and probability >= min_probability_positive
                and metric_consistency
            )
            bootstrap_detail = (
                f"n={used}, ci95=[{lo:.12f}, {hi:.12f}], "
                f"P(delta>0)={probability:.6f}, metric_consistency={metric_consistency}"
            )
        except (TypeError, ValueError) as exc:
            bootstrap_detail = f"invalid bootstrap evidence: {exc}"
    checks["bootstrap_stability"] = {
        "pass": bootstrap_ok,
        "detail": bootstrap_detail,
    }

    repro = evidence.get("reproducibility")
    repro_ok = False
    repro_detail = "reproducibility evidence missing"
    if isinstance(repro, dict):
        run1 = repro.get("run_1")
        run2 = repro.get("run_2")
        left_ok, _ = _metrics_ok(run1, score_tolerance=score_tolerance)
        right_ok, _ = _metrics_ok(run2, score_tolerance=score_tolerance)
        signature_equal = repro.get("deployment_signature_equal") is True
        if left_ok and right_ok and isinstance(run1, dict) and isinstance(run2, dict):
            deltas = {
                key: abs(float(run1[key]) - float(run2[key]))
                for key in METRIC_KEYS
            }
            repro_ok = signature_equal and all(
                value <= repro_tolerance for value in deltas.values()
            )
            repro_detail = (
                f"metric_deltas={deltas}, tolerance={repro_tolerance}, "
                f"deployment_signature_equal={signature_equal}"
            )
    checks["reproducibility"] = {
        "pass": repro_ok,
        "detail": repro_detail,
    }

    submission = evidence.get("submission")
    submission_ok = False
    submission_detail = "submission evidence missing"
    if isinstance(submission, dict):
        run1 = submission.get("run_1")
        run2 = submission.get("run_2")
        if isinstance(run1, dict) and isinstance(run2, dict):
            valid1 = run1.get("valid") is True
            valid2 = run2.get("valid") is True
            sha1 = str(run1.get("sha256") or "")
            sha2 = str(run2.get("sha256") or "")
            no_errors = not run1.get("errors") and not run2.get("errors")
            submission_ok = valid1 and valid2 and no_errors and bool(sha1) and sha1 == sha2
            submission_detail = (
                f"run1_valid={valid1}, run2_valid={valid2}, "
                f"sha256_equal={bool(sha1) and sha1 == sha2}"
            )
    checks["submission_validator_and_sha256"] = {
        "pass": submission_ok,
        "detail": submission_detail,
    }

    proven = all(bool(item["pass"]) for item in checks.values())
    return {
        "status": "PASS" if proven else "FAIL",
        "proven": proven,
        "definition": (
            "PROVEN requires green CI, official train/test preflight, strict organiser "
            "event/group leakage separation, a full explicit leakage-audit PASS, valid "
            "cross-fit metrics, final Score > baseline Score + epsilon, positive "
            "event-level bootstrap stability, reproducible final runs, strict submission "
            "validation, and byte-identical submission SHA256."
        ),
        "policy": {
            "score_epsilon": score_epsilon,
            "score_tolerance": score_tolerance,
            "repro_tolerance": repro_tolerance,
            "min_bootstrap_replicates": min_bootstrap_replicates,
            "min_probability_positive": min_probability_positive,
            "bootstrap_ci_lower_must_be_positive": True,
            "bootstrap_scores_must_match_crossfit": True,
        },
        "delta_score": delta_score,
        "checks": checks,
    }
