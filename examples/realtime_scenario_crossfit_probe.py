#!/usr/bin/env python3
"""Test Potage generalization to unseen initial-state scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from potage import SoupPipe
from examples.realtime_race_probe import FEATURE_NAMES, SCENARIO_FEATURE_NAMES
from examples.realtime_scenario_probe import (
    SCENARIOS,
    SCENARIO_FEATURE_MARKERS,
    generate_scenario_dataset,
)
from examples.realtime_stage_selection_probe import build_recipe


INVARIANT_FEATURE_NAMES = SCENARIO_FEATURE_NAMES + [
    "our_time_to_center",
    "opponent_time_to_center",
    "time_to_center_advantage",
    "speed_ratio",
    "separation_per_speed",
]


def add_invariant_features(X: np.ndarray) -> np.ndarray:
    """Add speed/geometry quantities with a more transportable scale."""
    speed_ours = np.maximum(X[:, 20], 1e-6)
    speed_opponent = np.maximum(X[:, 21], 1e-6)
    our_time = X[:, 23] / speed_ours
    opponent_time = X[:, 24] / speed_opponent
    time_advantage = opponent_time - our_time
    speed_ratio = speed_ours / speed_opponent
    separation_per_speed = X[:, 25] / (speed_ours + speed_opponent)
    return np.column_stack([
        X,
        our_time,
        opponent_time,
        time_advantage,
        speed_ratio,
        separation_per_speed,
    ])


def logistic_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000))


def metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def fit_fold_space(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    context: np.ndarray,
    discovery: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
    select: int,
    recipe: str,
    feature_space: str,
) -> Dict[str, object]:
    excluded = {
        "squared", "log", "sqrt", "cube", "tanh", "reciprocal",
        "rank", "am_ratio", "am_logratio", "am_power",
    }
    pipe = SoupPipe(
        X[discovery], y[discovery], list(names),
        feature_types=["numeric"] * X.shape[1],
        X_val=X[validation], y_val=y[validation],
        y_base=baseline_oof, y_val_base=baseline_validation,
        task="regression", exclude_families=excluded,
        max_candidates=100_000,
    )
    final = build_recipe(pipe, recipe, "correlation", select)
    train_design = np.column_stack([
        logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), final.X,
    ])
    validation_design = np.column_stack([
        logit(np.clip(baseline_validation, 1e-6, 1 - 1e-6)),
        pipe.transform(X[validation]),
    ])
    test_design = np.column_stack([
        logit(np.clip(baseline_test, 1e-6, 1 - 1e-6)),
        pipe.transform(X[test]),
    ])
    model = logistic_model().fit(train_design, y[discovery])
    test_probability = model.predict_proba(test_design)[:, 1]
    selected_scenario_features = [
        name for name in final.names
        if any(marker in name for marker in SCENARIO_FEATURE_MARKERS)
    ]
    return {
        "feature_space": feature_space,
        "test": metrics(y[test], test_probability),
        "selected_features": list(final.names),
        "selected_scenario_features": selected_scenario_features,
    }


def run_probe(games: int, seed: int, select: int, recipe: str) -> Dict[str, object]:
    X, y, context, groups, scenario_rows, generation = generate_scenario_dataset(games, seed)
    X_invariant = add_invariant_features(X)
    folds = []
    for fold_index, heldout_name in enumerate(SCENARIOS):
        test = np.flatnonzero(scenario_rows == heldout_name)
        train = np.flatnonzero(scenario_rows != heldout_name)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + fold_index + 1)
        discovery_local, validation_local = next(
            inner.split(X[train], y[train], groups[train])
        )
        discovery = train[discovery_local]
        validation = train[validation_local]
        baseline_oof = grouped_oof(context[discovery], y[discovery], groups[discovery])
        baseline = logistic_model().fit(context[discovery], y[discovery])
        baseline_validation = baseline.predict_proba(context[validation])[:, 1]
        baseline_test = baseline.predict_proba(context[test])[:, 1]
        fold_result = {
            "heldout_scenario": heldout_name,
            "rows": int(len(test)),
            "baseline": metrics(y[test], baseline_test),
            "results": [],
        }
        for X_space, names, feature_space in (
            (X[:, :len(FEATURE_NAMES)], FEATURE_NAMES, "core"),
            (X, SCENARIO_FEATURE_NAMES, "scenario"),
            (X_invariant, INVARIANT_FEATURE_NAMES, "invariant"),
        ):
            row = fit_fold_space(
                X_space, names, y, context, discovery, validation, test,
                baseline_oof, baseline_validation, baseline_test, select, recipe,
                feature_space,
            )
            fold_result["results"].append(row)
            print(
                f"held out {heldout_name:12s} {row['feature_space']:8s}: "
                f"Brier={row['test']['brier']:.4f}, "
                f"scenario_terms={row['selected_scenario_features']}"
            )
        folds.append(fold_result)
    aggregate = {}
    for feature_space in ("core", "scenario", "invariant"):
        rows = [
            next(row for row in fold["results"] if row["feature_space"] == feature_space)
            for fold in folds
        ]
        aggregate[feature_space] = {
            "mean_brier": float(np.mean([row["test"]["brier"] for row in rows])),
            "mean_auc": float(np.mean([row["test"]["roc_auc"] for row in rows])),
            "selected_scenario_feature_frequency": {
                marker: int(sum(
                    any(marker in name for name in row["selected_features"])
                    for row in rows
                ))
                for marker in SCENARIO_FEATURE_MARKERS
            },
        }
    return {
        "generation": generation,
        "seed": seed,
        "select": select,
        "recipe": recipe,
        "folds": folds,
        "aggregate": aggregate,
    }


def print_summary(result: Dict[str, object]) -> None:
    print("Scenario-held-out aggregate:")
    for space, row in result["aggregate"].items():
        print(f"  {space:8s}: mean Brier={row['mean_brier']:.4f}, mean AUC={row['mean_auc']:.4f}")
        print(f"    marker frequency: {row['selected_scenario_feature_frequency']}")


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    scenarios = [fold["heldout_scenario"] for fold in result["folds"]]
    core = [next(row for row in fold["results"] if row["feature_space"] == "core")["test"]["brier"] for fold in result["folds"]]
    scenario = [next(row for row in fold["results"] if row["feature_space"] == "scenario")["test"]["brier"] for fold in result["folds"]]
    invariant = [next(row for row in fold["results"] if row["feature_space"] == "invariant")["test"]["brier"] for fold in result["folds"]]
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, core, width, label="core")
    ax.bar(x, scenario, width, label="scenario features")
    ax.bar(x + width / 2, invariant, width, label="invariant features")
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("test Brier (lower is better)")
    ax.set_title("Generalization to unseen initial-state scenarios")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=800)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.games, args.seed, args.select, args.recipe)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
