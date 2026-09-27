"""
Model registry stage.

MLflow deprecated registry STAGES in 2.9 and will remove them. This module uses
ALIASES instead — a named pointer to one version, repointed atomically.

    champion    what the serving layer loads
    challenger  a candidate that passed the gate and is waiting for a decision

TODO: Complete find_best_run, register_model, set_alias, passes_quality_gate
      and promote_model.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MAX_FAIRNESS_GAP,
    MIN_PR_AUC,
    MIN_ROC_AUC,
    MLFLOW_EXPERIMENT_NAME,
    PRIMARY_METRIC,
    REGISTERED_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# TODO 1: Implement find_best_run
# =============================================================================
# Return the best run in an experiment by a single metric, as:
#   {"run_id": str, "metrics": dict, "params": dict, "artifact_uri": str}
#
# Requirements:
#   - raise ValueError if the experiment does not exist, or has no matching runs
#   - order by the metric, ASC when ascending is True, DESC otherwise
#   - max_results=1
#

def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    ascending: bool = False,
) -> Dict[str, Any]:
    """Best run in an experiment by a single metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist")

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "artifact_uri": run.info.artifact_uri,
    }


# =============================================================================
# TODO 2: Implement register_model
# =============================================================================
# Register a run's model artifact and return the new version number as a string.
#
def register_model(
    run_id: str, model_name: str = REGISTERED_MODEL_NAME, artifact_path: str = "model"
) -> str:
    """Register a run's model artifact and return the new version number."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    result = mlflow.register_model(model_uri, model_name)
    version = result.version
    logger.info("Registered %s v%s from run %s", model_name, version, run_id)
    return str(version)


# =============================================================================
# TODO 3: Implement set_alias
# =============================================================================
# Point an alias at a version. This replaces the deprecated
# client.transition_model_version_stage() — do not use that one.
#

def set_alias(model_name: str, version: str, alias: str) -> None:
    """Point an alias at a version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)
    logger.info("Set %s@%s -> v%s", model_name, alias, version)


def get_model_version_by_alias(
    model_name: str = REGISTERED_MODEL_NAME, alias: str = CHAMPION_ALIAS
) -> Optional[Dict[str, Any]]:
    """Which version does an alias currently point at? None if it is unset."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001 - alias or model may simply not exist yet
        return None
    return {
        "name": mv.name,
        "version": mv.version,
        "run_id": mv.run_id,
        "aliases": list(mv.aliases),
        "model_uri": f"models:/{model_name}@{alias}",
    }


# =============================================================================
# =============================================================================
# TODO 4: Implement passes_quality_gate
# =============================================================================
# Three independent checks; a model must clear all three:
#     roc_auc      >= MIN_ROC_AUC
#     pr_auc       >= MIN_PR_AUC
#     fairness_gap <= MAX_FAIRNESS_GAP     (note the direction)
#
# Return:
#   {"passed": bool,
#    "failed_checks": [names of the checks that failed],
#    "detail": {name: {"passed": bool, "rule": "human-readable rule"}}}
#
# Requirement that decides a test: read the metrics with .get(key, DEFAULT) and
# choose the default so a MISSING metric FAILS. Use 0.0 for the two that must be
# large. If a missing metric defaulted to a pass, a bug that stopped computing
# the fairness gap would silently disable the fairness check — and the gate
# would keep reporting green.
#

def passes_quality_gate(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Does this model clear the promotion bar?"""
    checks = {}
    failed_checks = []

    roc_auc = metrics.get("roc_auc", 0.0)
    roc_auc_passed = roc_auc >= MIN_ROC_AUC
    checks["roc_auc"] = {
        "passed": roc_auc_passed,
        "rule": f"roc_auc >= {MIN_ROC_AUC}",
    }
    if not roc_auc_passed:
        failed_checks.append("roc_auc")

    pr_auc = metrics.get("pr_auc", 0.0)
    pr_auc_passed = pr_auc >= MIN_PR_AUC
    checks["pr_auc"] = {
        "passed": pr_auc_passed,
        "rule": f"pr_auc >= {MIN_PR_AUC}",
    }
    if not pr_auc_passed:
        failed_checks.append("pr_auc")

    fairness_gap = metrics.get("fairness_gap", float('inf'))
    fairness_passed = fairness_gap <= MAX_FAIRNESS_GAP
    checks["fairness_gap"] = {
        "passed": fairness_passed,
        "rule": f"fairness_gap <= {MAX_FAIRNESS_GAP}",
    }
    if not fairness_passed:
        failed_checks.append("fairness_gap")

    return {
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "detail": checks,
    }


def beats_champion(
    candidate_metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
    metric: str = PRIMARY_METRIC,
    margin: float = 0.002,
) -> bool:
    """Is the candidate better than what is already live?

    The margin exists so that noise does not trigger a deployment. Shipping a
    model that is 0.0003 AUC better is all risk and no reward.
    """
    current = get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    if current is None:
        logger.info("No champion yet — candidate wins by default")
        return True
    client = MlflowClient()
    run = client.get_run(current["run_id"])
    champion_score = run.data.metrics.get(metric, 0.0)
    candidate_score = candidate_metrics.get(metric, 0.0)
    logger.info("Champion %s=%.4f, candidate %s=%.4f", metric, champion_score, metric, candidate_score)
    return candidate_score >= champion_score + margin


# =============================================================================
# =============================================================================
# TODO 5: Implement promote_model
# =============================================================================
# Register the run, then decide what alias it deserves. Three outcomes, and only
# the first changes what production serves:
#
#   "champion"    passed the gate AND beats_champion(...) is True  -> set CHAMPION_ALIAS
#   "challenger"  passed the gate but does not beat the champion   -> set CHALLENGER_ALIAS
#   "rejected"    failed the gate                                  -> no alias
#
# Requirements:
#   - call passes_quality_gate FIRST, but register the version either way. A
#     rejected model still gets a version and a "quality_gate: failed" tag —
#     that is the audit trail, and it is how you show a regulator that the bad
#     model was caught rather than never produced.
#   - tag the version: client.set_model_version_tag(name, version, "quality_gate", ...)
#   - return {"run_id", "model_name", "version", "outcome", "quality_gate", "metrics"}
#

def promote_model(
    run_id: str,
    metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
) -> Dict[str, Any]:
    """Register a run, then decide what alias it deserves."""
    quality_gate = passes_quality_gate(metrics)
    version = register_model(run_id, model_name)

    client = MlflowClient()

    if quality_gate["passed"]:
        if beats_champion(metrics, model_name):
            set_alias(model_name, version, CHAMPION_ALIAS)
            outcome = "champion"
            logger.info("Promoted to champion: %s v%s", model_name, version)
        else:
            set_alias(model_name, version, CHALLENGER_ALIAS)
            outcome = "challenger"
            logger.info("Promoted to challenger: %s v%s", model_name, version)
    else:
        outcome = "rejected"
        logger.info("Rejected: %s v%s", model_name, version)

    client.set_model_version_tag(model_name, version, "quality_gate", str(quality_gate["passed"]))

    return {
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "outcome": outcome,
        "quality_gate": quality_gate,
        "metrics": metrics,
    }


# =============================================================================
# Helpers (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """Every registered model with its versions and aliases."""
    client = MlflowClient()
    out = []
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        out.append({
            "name": model.name,
            "aliases": dict(model.aliases or {}),
            "versions": [{"version": v.version, "run_id": v.run_id, "aliases": list(v.aliases)}
                         for v in versions],
        })
    return out


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    top_n: int = 10,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Top N runs, for the comparison table in your report."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=top_n,
    )
    return [
        {"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName", ""),
         "metrics": dict(r.data.metrics), "params": dict(r.data.params)}
        for r in runs
    ]
