#!/usr/bin/env python3
"""Test Potage with continuous initial geometry and held-out speed regimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from examples.realtime_race_probe import FEATURE_NAMES, SCENARIO_FEATURE_NAMES, play_game
from examples.realtime_stage_selection_probe import build_recipe


SAFE_CELLS = [
    np.asarray([row, column], dtype=np.int64)
    for row in range(5)
    for column in range(5)
    if [row, column] not in ([0, 2], [2, 1], [4, 2])
]
IN_SUPPORT_SPEEDS = ((1, 1), (2, 1), (1, 2), (2, 2))
HELDOUT_SPEEDS = ((3, 1), (1, 3), (3, 3))


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


def generate_dataset(
    games: int, seed: int, speed_pairs: Sequence[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.RandomState(seed)
    rows: List[np.ndarray] = []
    outcomes: List[float] = []
    contexts: List[np.ndarray] = []
    groups: List[int] = []
    draws = 0
    for game_id in range(games):
        first, second = rng.choice(len(SAFE_CELLS), size=2, replace=False)
        speed_pair = speed_pairs[int(rng.randint(len(speed_pairs)))]
        simulation_seed = int(rng.randint(0, 2**31 - 1))
        result = play_game(
            game_id,
            np.random.RandomState(simulation_seed),
            start_positions={1: SAFE_CELLS[first], -1: SAFE_CELLS[second]},
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
    return (
        np.row_stack(rows),
        np.asarray(outcomes, dtype=np.float64),
        np.row_stack(contexts),
        np.asarray(groups, dtype=np.int64),
        {"games_requested": games, "games_kept": games - draws, "draws": draws},
    )


def fit_space(
    X_train: np.ndarray,
    names: Sequence[str],
    y_train: np.ndarray,
    context_train: np.ndarray,
    groups_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    context_validation: np.ndarray,
    X_tests: Dict[str, np.ndarray],
    y_tests: Dict[str, np.ndarray],
    context_tests: Dict[str, np.ndarray],
    select: int,
    recipe: str,
    feature_space: str,
) -> Dict[str, object]:
    discovery_local, validation_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=123).split(
            X_train, y_train, groups_train
        )
    )
    discovery = discovery_local
    validation = validation_local
    baseline_oof = grouped_oof(context_train[discovery], y_train[discovery], groups_train[discovery])
    baseline = logistic_model().fit(context_train[discovery], y_train[discovery])
    baseline_validation = baseline.predict_proba(context_train[validation])[:, 1]
    baseline_tests = {
        name: baseline.predict_proba(context_tests[name])[:, 1]
        for name in X_tests
    }
    excluded = {
        "squared", "log", "sqrt", "cube", "tanh", "reciprocal",
        "rank", "am_ratio", "am_logratio", "am_power",
    }
    pipe = SoupPipe(
        X_train[discovery], y_train[discovery], list(names),
        feature_types=["numeric"] * X_train.shape[1],
        X_val=X_train[validation], y_val=y_train[validation],
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
        pipe.transform(X_train[validation]),
    ])
    model = logistic_model().fit(train_design, y_train[discovery])
    # Ensure the validation path was materialized before reporting the test path.
    model.predict_proba(validation_design)
    scores = {
        name: metrics(
            y_tests[name],
            model.predict_proba(
                np.column_stack([
                    logit(np.clip(baseline_tests[name], 1e-6, 1 - 1e-6)),
                    pipe.transform(X_tests[name]),
                ])
            )[:, 1],
        )
        for name in X_tests
    }
    selected_scenario = [
        name for name in final.names
        if any(marker in name for marker in (
            "our_speed", "opponent_speed", "speed_difference",
            "our_start_to_center_node", "opponent_start_to_center_node",
        ))
    ]
    return {
        "feature_space": feature_space,
        "recipe": recipe,
        "test": scores,
        "selected_features": list(final.names),
        "selected_scenario_features": selected_scenario,
    }


def run_probe(games: int, seed: int, select: int, recipe: str) -> Dict[str, object]:
    train_X, train_y, train_context, train_groups, train_generation = generate_dataset(
        games, seed, IN_SUPPORT_SPEEDS
    )
    in_X, in_y, in_context, _, in_generation = generate_dataset(
        max(200, games // 3), seed + 1, IN_SUPPORT_SPEEDS
    )
    out_X, out_y, out_context, _, out_generation = generate_dataset(
        max(200, games // 3), seed + 2, HELDOUT_SPEEDS
    )
    test_X = {"in_support": in_X, "heldout_speed": out_X}
    test_y = {"in_support": in_y, "heldout_speed": out_y}
    test_context = {"in_support": in_context, "heldout_speed": out_context}
    results = []
    for X_space, names, label in (
        (train_X[:, :len(FEATURE_NAMES)], FEATURE_NAMES, "core"),
        (train_X, SCENARIO_FEATURE_NAMES, "scenario"),
    ):
        test_spaces = {
            name: X[:, :len(FEATURE_NAMES)] if label == "core" else X
            for name, X in test_X.items()
        }
        row = fit_space(
            X_space, names, train_y, train_context, train_groups,
            X_space, train_y, train_context,
            test_spaces, test_y, test_context,
            select, recipe, label,
        )
        print(f"{label:8s}: in={row['test']['in_support']['brier']:.4f}, "
              f"heldout-speed={row['test']['heldout_speed']['brier']:.4f}, "
              f"scenario_terms={row['selected_scenario_features']}")
        results.append(row)
    return {
        "seed": seed,
        "games": games,
        "recipe": recipe,
        "train_generation": train_generation,
        "in_support_generation": in_generation,
        "heldout_speed_generation": out_generation,
        "results": results,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = ["core", "scenario"]
    x = np.arange(len(labels))
    in_values = [row["test"]["in_support"]["brier"] for row in result["results"]]
    out_values = [row["test"]["heldout_speed"]["brier"] for row in result["results"]]
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width / 2, in_values, width, label="in-support speed")
    ax.bar(x + width / 2, out_values, width, label="held-out speed 3")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Continuous scenario coverage")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.games, args.seed, args.select, args.recipe)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
