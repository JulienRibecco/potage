#!/usr/bin/env python3
"""Predict which real-time states benefit from speed or position interventions."""

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
from examples.realtime_continuous_scenario_probe import SAFE_CELLS
from examples.realtime_race_probe import (
    FEATURE_NAMES,
    NODE_POSITIONS,
    SCENARIO_FEATURE_NAMES,
    SNAPSHOT_TICKS,
    move_one,
    play_game,
)
from examples.realtime_stage_selection_probe import build_recipe


def logistic_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000))


def metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    auc = (
        float(roc_auc_score(y, probability))
        if np.unique(y).size == 2
        else float("nan")
    )
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": auc,
    }


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def policy_noise(seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {1: rng.gumbel(size=(100, 3)), -1: rng.gumbel(size=(100, 3))}


def game_seed(seed: int, game_id: int, focal: int) -> int:
    return int((seed * 1_000_003 + game_id * 17 + (focal == -1)) % (2**32 - 1))


def randomized_setup(seed: int, game_id: int, focal: int) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    setup_rng = np.random.RandomState(game_seed(seed + 31, game_id, focal))
    first, second = setup_rng.choice(len(SAFE_CELLS), size=2, replace=False)
    speed_pair = ((1, 1), (2, 1), (1, 2), (2, 2))[int(setup_rng.randint(4))]
    return (
        {1: SAFE_CELLS[first].copy(), -1: SAFE_CELLS[second].copy()},
        {1: speed_pair[0], -1: speed_pair[1]},
    )


def winner_for(result, focal: int) -> int:
    _, y, _, _ = result
    if not y:
        return 0
    return 1 if y[0] > 0.5 else -1


def collect_dataset(
    pairs: int,
    seed: int,
    kind: str,
    phases: Sequence[int],
    speed_delta: int = 1,
    position_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rows: List[np.ndarray] = []
    labels: List[float] = []
    contexts: List[np.ndarray] = []
    groups: List[int] = []
    draws = 0
    positive_switches = 0
    for game_id in range(pairs):
        for focal in (1, -1):
            positions, speeds = randomized_setup(seed, game_id, focal)
            treated_positions = {player: value.copy() for player, value in positions.items()}
            treated_speeds = dict(speeds)
            if kind == "speed":
                treated_speeds[focal] = min(4, treated_speeds[focal] + int(speed_delta))
            elif kind == "initial_position":
                moved = move_one(
                    treated_positions[focal], NODE_POSITIONS[1], speed=position_steps
                )
                if np.array_equal(moved, treated_positions[-focal]):
                    moved = move_one(
                        treated_positions[focal], NODE_POSITIONS[0], speed=position_steps
                    )
                treated_positions[focal] = moved
            else:
                raise ValueError(f"unknown intervention kind: {kind}")

            current_seed = game_seed(seed, game_id, focal)
            noise = policy_noise((current_seed + 7919) % (2**32 - 1))
            control = play_game(
                game_id,
                np.random.RandomState(current_seed),
                policy_noise=noise,
                start_positions=positions,
                speed_override=speeds,
                scenario_features=True,
            )
            treated = play_game(
                game_id,
                np.random.RandomState(current_seed),
                policy_noise=noise,
                start_positions=treated_positions,
                speed_override=treated_speeds,
                scenario_features=True,
            )
            control_X, control_y, control_context, _ = control
            treated_X, treated_y, _, _ = treated
            if not control_y or not treated_y:
                draws += 1
                continue
            focal_offset = 0 if focal == 1 else 1
            for phase in phases:
                row_index = 2 * phase + focal_offset
                if row_index >= len(control_X) or row_index >= len(treated_X):
                    continue
                control_winner = winner_for(control, focal)
                treated_winner = winner_for(treated, focal)
                label = float(treated_winner == focal and control_winner != focal)
                rows.append(control_X[row_index])
                contexts.append(control_context[row_index])
                labels.append(label)
                groups.append(game_id)
                positive_switches += int(label)
    if not rows:
        raise RuntimeError("no non-draw intervention pairs were generated")
    return (
        np.row_stack(rows),
        np.asarray(labels, dtype=np.float64),
        np.row_stack(contexts),
        np.asarray(groups, dtype=np.int64),
        {
            "pairs_requested": pairs,
            "pairs_with_draw_or_mismatch": draws,
            "rows": len(rows),
            "positive_switches": positive_switches,
        },
    )


def fit_space(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    context: np.ndarray,
    groups: np.ndarray,
    select: int,
    recipe: str,
    method: str,
) -> Dict[str, object]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=17)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=23)
    discovery_local, validation_local = next(
        inner.split(X[discovery_val], y[discovery_val], groups[discovery_val])
    )
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]
    baseline_oof = grouped_oof(context[discovery], y[discovery], groups[discovery])
    baseline = logistic_model().fit(context[discovery], y[discovery])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    baseline_scores = {
        "validation": metrics(y[validation], baseline_validation),
        "test": metrics(y[test], baseline_test),
    }
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
    final = build_recipe(pipe, recipe, method, select)
    design = lambda base, values: np.column_stack([
        logit(np.clip(base, 1e-6, 1.0 - 1e-6)), values,
    ])
    model = logistic_model().fit(
        design(baseline_oof, final.X), y[discovery]
    )
    validation_probability = model.predict_proba(
        design(baseline_validation, pipe.transform(X[validation]))
    )[:, 1]
    test_probability = model.predict_proba(
        design(baseline_test, pipe.transform(X[test]))
    )[:, 1]
    return {
        "baseline": baseline_scores,
        "potage": {
            "validation": metrics(y[validation], validation_probability),
            "test": metrics(y[test], test_probability),
            "features": [
                {"name": name, "coefficient": float(coefficient)}
                for name, coefficient in sorted(
                    zip(final.names, model[-1].coef_[0][1:]),
                    key=lambda pair: -abs(pair[1]),
                )
            ],
        },
        "split": {
            "discovery_rows": len(discovery),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "discovery_games": int(np.unique(groups[discovery]).size),
            "validation_games": int(np.unique(groups[validation]).size),
            "test_games": int(np.unique(groups[test]).size),
        },
    }


def run_probe(
    pairs: int, seed: int, select: int, recipe: str, method: str, phases: Sequence[int]
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for kind in ("speed", "initial_position"):
        X, y, context, groups, generation = collect_dataset(pairs, seed, kind, phases)
        rows = {}
        for values, names, space in (
            (context, ["time_fraction", "score_difference", "skill_difference", "perspective"], "context"),
            (X[:, :len(FEATURE_NAMES)], FEATURE_NAMES, "core"),
            (X, SCENARIO_FEATURE_NAMES, "scenario"),
        ):
            rows[space] = fit_space(values, names, y, context, groups, select, recipe, method)
            test = rows[space]["potage"]["test"]
            print(
                f"{kind:16s} {space:8s}: Brier={test['brier']:.4f}, "
                f"AUC={test['roc_auc']:.4f}"
            )
        results[kind] = {
            "generation": generation,
            "rows": len(X),
            "positive_rate": float(y.mean()),
            "split": rows["core"]["split"],
            "models": rows,
        }
    return {
        "pairs": pairs,
        "seed": seed,
        "recipe": recipe,
        "method": method,
        "phases": list(phases),
        "results": results,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    spaces = ["context", "core", "scenario"]
    labels = ["context", "core", "scenario"]
    x = np.arange(len(spaces))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, kind in zip((-width / 2, width / 2), ("speed", "initial_position")):
        values = [
            result["results"][kind]["models"][space]["potage"]["test"]["brier"]
            for space in spaces
        ]
        ax.bar(x + offset, values, width, label=kind.replace("_", " "))
    ax.set_xticks(x, labels)
    ax.set_ylabel("test Brier (lower is better)")
    ax.set_title("Predicting heterogeneous scenario effects")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--method", default="correlation")
    parser.add_argument("--phases", type=int, nargs="*", default=list(range(9)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.pairs, args.seed, args.select, args.recipe, args.method, args.phases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
