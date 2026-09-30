#!/usr/bin/env python3
"""Probe whether initial geometry and agent speed enter Potage's factors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from potage import SoupPipe
from examples.realtime_race_probe import (
    FEATURE_NAMES,
    SCENARIO_FEATURE_NAMES,
    play_game,
)
from examples.realtime_stage_selection_probe import build_recipe


SCENARIOS = {
    "diagonal": {1: np.asarray([0, 0]), -1: np.asarray([4, 4])},
    "horizontal": {1: np.asarray([0, 0]), -1: np.asarray([0, 4])},
    "vertical": {1: np.asarray([0, 4]), -1: np.asarray([4, 4])},
    "center_edge": {1: np.asarray([2, 0]), -1: np.asarray([4, 4])},
}
SPEED_PAIRS = ((1, 1), (2, 1), (1, 2), (2, 2))
SCENARIO_FEATURE_MARKERS = (
    "our_speed",
    "opponent_speed",
    "speed_difference",
    "our_start_to_center_node",
    "opponent_start_to_center_node",
    "start_separation",
)


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


def generate_scenario_dataset(
    n_games: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    rng = np.random.RandomState(seed)
    rows: List[np.ndarray] = []
    outcomes: List[float] = []
    contexts: List[np.ndarray] = []
    groups: List[int] = []
    scenario_rows: List[str] = []
    draws = 0
    scenario_counts = {name: 0 for name in SCENARIOS}
    speed_counts = {f"{a}-{b}": 0 for a, b in SPEED_PAIRS}
    scenario_names = list(SCENARIOS)
    for game_id in range(n_games):
        scenario_name = scenario_names[int(rng.randint(len(scenario_names)))]
        speed_pair = SPEED_PAIRS[int(rng.randint(len(SPEED_PAIRS)))]
        scenario_counts[scenario_name] += 1
        speed_counts[f"{speed_pair[0]}-{speed_pair[1]}"] += 1
        simulation_seed = int(rng.randint(0, 2**31 - 1))
        result = play_game(
            game_id,
            np.random.RandomState(simulation_seed),
            start_positions=SCENARIOS[scenario_name],
            speed_override={1: speed_pair[0], -1: speed_pair[1]},
            scenario_features=True,
        )
        game_rows, game_y, game_context, game_groups = result
        if not game_rows:
            draws += 1
            continue
        rows.extend(game_rows)
        outcomes.extend(game_y)
        contexts.extend(game_context)
        groups.extend(game_groups)
        scenario_rows.extend([scenario_name] * len(game_rows))
    return (
        np.row_stack(rows),
        np.asarray(outcomes, dtype=np.float64),
        np.row_stack(contexts),
        np.asarray(groups, dtype=np.int64),
        np.asarray(scenario_rows),
        {
            "games_requested": n_games,
            "games_kept": n_games - draws,
            "draws": draws,
            "scenario_counts": scenario_counts,
            "speed_counts": speed_counts,
        },
    )


def fit_recipe(
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
    validation_probability = model.predict_proba(validation_design)[:, 1]
    test_probability = model.predict_proba(test_design)[:, 1]
    selected_scenario_features = [
        name for name in final.names
        if any(marker in name for marker in SCENARIO_FEATURE_MARKERS)
    ]
    return {
        "recipe": recipe,
        "feature_space": "scenario" if X.shape[1] > len(FEATURE_NAMES) else "core",
        "validation": metrics(y[validation], validation_probability),
        "test": metrics(y[test], test_probability),
        "selected_features": list(final.names),
        "selected_scenario_features": selected_scenario_features,
        "pipeline_recap": pipe.recap(),
    }


def run_probe(n_games: int, seed: int, select: int, recipes: Iterable[str]) -> Dict[str, object]:
    X, y, context, groups, scenario_rows, generation = generate_scenario_dataset(n_games, seed)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    discovery_local, validation_local = next(
        inner.split(X[discovery_val], y[discovery_val], groups[discovery_val])
    )
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]
    baseline_oof = grouped_oof(context[discovery], y[discovery], groups[discovery])
    baseline = logistic_model().fit(context[discovery], y[discovery])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    results = []
    for recipe in recipes:
        print(f"running {recipe} with core features")
        core = fit_recipe(
            X[:, :len(FEATURE_NAMES)], FEATURE_NAMES, y, context,
            discovery, validation, test, baseline_oof, baseline_validation,
            baseline_test, select, recipe,
        )
        print(f"  core test Brier={core['test']['brier']:.4f}")
        print(f"running {recipe} with scenario features")
        scenario = fit_recipe(
            X, SCENARIO_FEATURE_NAMES, y, context,
            discovery, validation, test, baseline_oof, baseline_validation,
            baseline_test, select, recipe,
        )
        print(f"  scenario test Brier={scenario['test']['brier']:.4f}")
        results.extend([core, scenario])
    scenario_test_rows = {}
    for scenario_name in SCENARIOS:
        mask = scenario_rows[test] == scenario_name
        scenario_test_rows[scenario_name] = {
            "rows": int(mask.sum()),
            "win_rate": float(y[test][mask].mean()) if mask.any() else float("nan"),
        }
    return {
        "generation": generation,
        "seed": seed,
        "select": select,
        "recipes": list(recipes),
        "rows": len(X),
        "positive_rate": float(y.mean()),
        "split": {
            "discovery_rows": len(discovery),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "discovery_games": int(np.unique(groups[discovery]).size),
            "validation_games": int(np.unique(groups[validation]).size),
            "test_games": int(np.unique(groups[test]).size),
        },
        "baseline": {
            "validation": metrics(y[validation], baseline_validation),
            "test": metrics(y[test], baseline_test),
        },
        "scenario_test_rows": scenario_test_rows,
        "results": results,
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; rows={result['rows']}")
    print(f"Baseline test Brier: {result['baseline']['test']['brier']:.4f}")
    print("Feature-space ablation:")
    for row in result["results"]:
        print(
            f"  {row['recipe']:12s} {row['feature_space']:8s} "
            f"Brier={row['test']['brier']:.4f} AUC={row['test']['roc_auc']:.4f} "
            f"scenario_features={row['selected_scenario_features']}"
        )


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = []
    values = []
    for row in result["results"]:
        labels.append(f"{row['recipe']}\n{row['feature_space']}")
        values.append(row["test"]["brier"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(np.arange(len(labels)), values)
    ax.axhline(result["baseline"]["test"]["brier"], color="black", linestyle="--", label="baseline")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("test Brier (lower is better)")
    ax.set_title("Initial-state and speed feature ablation")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipes", nargs="*", default=["raw", "carrier_am"])
    parser.add_argument("--stability-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--stability-games", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.games, args.seed, args.select, args.recipes)
    print_summary(result)
    stability = []
    for seed in args.stability_seeds:
        repeat = run_probe(args.stability_games, seed, args.select, args.recipes)
        row = {
            "seed": seed,
            "games": args.stability_games,
            "baseline_test_brier": repeat["baseline"]["test"]["brier"],
            "results": [
                {
                    "recipe": item["recipe"],
                    "feature_space": item["feature_space"],
                    "test_brier": item["test"]["brier"],
                    "selected_scenario_features": item["selected_scenario_features"],
                }
                for item in repeat["results"]
            ],
        }
        stability.append(row)
        best = min(row["results"], key=lambda item: item["test_brier"])
        print(
            f"stability seed {seed}: baseline={row['baseline_test_brier']:.4f}, "
            f"best={best['test_brier']:.4f} "
            f"({best['recipe']}/{best['feature_space']})"
        )
    result["stability_runs"] = stability
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
