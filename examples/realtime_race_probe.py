#!/usr/bin/env python3
"""Probe timing and tempo factors in a simultaneous-action microgame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


ARENA_SIZE = 5
NODE_POSITIONS = np.asarray([[0, 2], [2, 1], [4, 2]], dtype=np.int64)
NODE_VALUES = np.asarray([1.0, 1.5, 1.0])
TICKS = 100
SNAPSHOT_TICKS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
PLAYERS = (1, -1)


def manhattan(position: np.ndarray, target: np.ndarray) -> int:
    return int(np.abs(position - target).sum())


def move_one(position: np.ndarray, target: np.ndarray, speed: int = 1) -> np.ndarray:
    result = position.copy()
    for _ in range(max(0, int(speed))):
        delta = target - result
        if abs(int(delta[0])) >= abs(int(delta[1])) and delta[0] != 0:
            result[0] += 1 if delta[0] > 0 else -1
        elif delta[1] != 0:
            result[1] += 1 if delta[1] > 0 else -1
        else:
            break
    return result


def nearest_node_distance(position: np.ndarray) -> float:
    return float(min(manhattan(position, node) for node in NODE_POSITIONS))


def best_node_distance(position: np.ndarray) -> float:
    return float(
        min(
            manhattan(position, node) / NODE_VALUES[index]
            for index, node in enumerate(NODE_POSITIONS)
        )
    )


def choose_target(
    position: np.ndarray,
    node_control: np.ndarray,
    node_progress: np.ndarray,
    skill: float,
    rng: np.random.RandomState,
    noise: Optional[np.ndarray] = None,
) -> int:
    scores = []
    for index, node in enumerate(NODE_POSITIONS):
        distance = manhattan(position, node)
        contest_penalty = 0.7 if node_control[index] == 0 else 0.0
        progress_bonus = 0.5 * abs(node_progress[index])
        scores.append(
            NODE_VALUES[index] * 2.0
            - 0.8 * distance
            + progress_bonus
            - contest_penalty
        )
    noise_scale = 0.2 + 3.5 * (1.0 - skill) ** 2
    random_term = (
        rng.gumbel(scale=noise_scale, size=3)
        if noise is None
        else np.asarray(noise, dtype=np.float64) * noise_scale
    )
    return int(np.argmax(np.asarray(scores) * skill + random_term))


def update_node_control(
    positions: Dict[int, np.ndarray],
    node_control: np.ndarray,
    node_progress: np.ndarray,
    scores: Dict[int, float],
) -> None:
    for index, node in enumerate(NODE_POSITIONS):
        occupants = [player for player in PLAYERS if np.array_equal(positions[player], node)]
        if len(occupants) == 1:
            player = occupants[0]
            node_progress[index] += player
            if abs(node_progress[index]) >= 3:
                node_control[index] = player
                node_progress[index] = 3 * player
            scores[player] += NODE_VALUES[index] if node_control[index] == player else 0.0
        elif len(occupants) == 2:
            node_progress[index] = int(node_progress[index] * 0.8)


def state_features(
    tick: int,
    positions: Dict[int, np.ndarray],
    targets: Dict[int, int],
    pending_due: Dict[int, int],
    node_control: np.ndarray,
    node_progress: np.ndarray,
    scores: Dict[int, float],
    score_history: Dict[int, List[float]],
    player: int,
    extra_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    opponent = -player
    our_position = positions[player]
    opponent_position = positions[opponent]
    our_target = NODE_POSITIONS[targets[player]]
    opponent_target = NODE_POSITIONS[targets[opponent]]
    controlled_ours = float(np.sum(node_control == player))
    controlled_opponent = float(np.sum(node_control == opponent))
    contested = float(np.sum(node_control == 0))
    our_rate = score_history[player][-1] - score_history[player][0] if score_history[player] else 0.0
    opponent_rate = score_history[opponent][-1] - score_history[opponent][0] if score_history[opponent] else 0.0
    base = np.asarray(
        [
            tick / TICKS,
            scores[player] - scores[opponent],
            float(np.sum(node_control == player)),
            float(np.sum(node_control == opponent)),
            contested,
            nearest_node_distance(our_position),
            nearest_node_distance(opponent_position),
            best_node_distance(our_position),
            best_node_distance(opponent_position),
            manhattan(our_position, our_target),
            manhattan(opponent_position, opponent_target),
            float(max(0, pending_due[player] - tick)),
            float(max(0, pending_due[opponent] - tick)),
            float(node_progress[targets[player]] * player),
            float(node_progress[targets[opponent]] * opponent),
            our_rate,
            opponent_rate,
            float(manhattan(our_position, opponent_position)),
            float(np.max(node_progress) - np.min(node_progress)),
            (TICKS - tick) % 2,
        ],
        dtype=np.float64,
    )
    if extra_features is not None:
        return np.concatenate([base, np.asarray(extra_features, dtype=np.float64)])
    return base


FEATURE_NAMES = [
    "time_fraction",
    "score_difference",
    "our_controlled_nodes",
    "opponent_controlled_nodes",
    "contested_nodes",
    "our_nearest_node_distance",
    "opponent_nearest_node_distance",
    "our_best_node_distance",
    "opponent_best_node_distance",
    "our_target_distance",
    "opponent_target_distance",
    "our_pending_action",
    "opponent_pending_action",
    "our_target_progress",
    "opponent_target_progress",
    "our_score_rate",
    "opponent_score_rate",
    "agent_distance",
    "control_progress_spread",
    "remaining_time_parity",
]

SCENARIO_FEATURE_NAMES = FEATURE_NAMES + [
    "our_speed",
    "opponent_speed",
    "speed_difference",
    "our_start_to_center_node",
    "opponent_start_to_center_node",
    "start_separation",
]


def play_game(
    game_id: int,
    rng: np.random.RandomState,
    latency_override: Optional[Dict[int, int]] = None,
    policy_noise: Optional[Dict[int, np.ndarray]] = None,
    latency_override_start: Optional[int] = None,
    start_positions: Optional[Dict[int, np.ndarray]] = None,
    speed_override: Optional[Dict[int, int]] = None,
    scenario_features: bool = False,
    speed_override_start: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[int]]:
    default_positions = {1: np.asarray([0, 0], dtype=np.int64), -1: np.asarray([4, 4], dtype=np.int64)}
    positions = {
        player: np.asarray(
            (start_positions[player] if start_positions and player in start_positions
             else default_positions[player]),
            dtype=np.int64,
        ).copy()
        for player in PLAYERS
    }
    initial_positions = {player: position.copy() for player, position in positions.items()}
    speed = {1: 1, -1: 1}
    if speed_override and speed_override_start is None:
        speed.update({int(player): int(value) for player, value in speed_override.items()})
    base_speed = dict(speed)
    skills = {1: rng.uniform(0.2, 1.0), -1: rng.uniform(0.2, 1.0)}
    latency = {1: int(rng.randint(0, 4)), -1: int(rng.randint(0, 4))}
    if latency_override and latency_override_start is None:
        latency.update({int(player): int(value) for player, value in latency_override.items()})
    base_latency = dict(latency)
    targets = {1: 1, -1: 1}
    pending_due = {1: 0, -1: 0}
    pending_target = {1: 1, -1: 1}
    node_control = np.zeros(3, dtype=np.int64)
    node_progress = np.zeros(3, dtype=np.int64)
    scores = {1: 0.0, -1: 0.0}
    score_history = {1: [0.0], -1: [0.0]}
    player = 1
    snapshots = []

    for tick in range(TICKS):
        if speed_override and speed_override_start is not None and tick >= speed_override_start:
            speed = dict(base_speed)
            speed.update({int(player): int(value) for player, value in speed_override.items()})
        if latency_override and latency_override_start is not None and tick >= latency_override_start:
            latency = dict(base_latency)
            latency.update({int(player): int(value) for player, value in latency_override.items()})
        for current in PLAYERS:
            if pending_due[current] <= tick:
                targets[current] = pending_target[current]
        for current in PLAYERS:
            targets[current] = int(targets[current])
            if tick >= pending_due[current]:
                pending_target[current] = choose_target(
                    positions[current],
                    node_control,
                    node_progress,
                    skills[current],
                    rng,
                    noise=(
                        policy_noise[current][tick]
                        if policy_noise is not None
                        else None
                    ),
                )
                pending_due[current] = tick + latency[current] + 1
        for current in PLAYERS:
            positions[current] = move_one(
                positions[current], NODE_POSITIONS[targets[current]], speed=speed[current]
            )
        update_node_control(positions, node_control, node_progress, scores)
        for current in PLAYERS:
            score_history[current].append(scores[current])

        if tick + 1 in SNAPSHOT_TICKS:
            for current in PLAYERS:
                features = state_features(
                    tick + 1,
                    positions,
                    targets,
                    pending_due,
                    node_control,
                    node_progress,
                    scores,
                    score_history,
                    current,
                    extra_features=(
                        np.asarray(
                            [
                                speed[current],
                                speed[-current],
                                speed[current] - speed[-current],
                                manhattan(initial_positions[current], NODE_POSITIONS[1]),
                                manhattan(initial_positions[-current], NODE_POSITIONS[1]),
                                manhattan(initial_positions[current], initial_positions[-current]),
                            ],
                            dtype=np.float64,
                        )
                        if scenario_features
                        else None
                    ),
                )
                context = np.asarray(
                    [
                        features[0],
                        features[1],
                        skills[current] - skills[-current],
                        float(current == 1),
                    ],
                    dtype=np.float64,
                )
                snapshots.append((features, context, current))

    if scores[1] == scores[-1]:
        return [], [], [], []
    winner = 1 if scores[1] > scores[-1] else -1
    X = [snapshot[0] for snapshot in snapshots]
    y = [float(snapshot[2] == winner) for snapshot in snapshots]
    context = [snapshot[1] for snapshot in snapshots]
    groups = [game_id] * len(snapshots)
    return X, y, context, groups


def generate_dataset(
    n_games: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.RandomState(seed)
    rows: List[np.ndarray] = []
    outcomes: List[float] = []
    contexts: List[np.ndarray] = []
    groups: List[int] = []
    draws = 0
    for game_id in range(n_games):
        game_rows, game_y, game_context, game_groups = play_game(game_id, rng)
        if not game_rows:
            draws += 1
            continue
        rows.extend(game_rows)
        outcomes.extend(game_y)
        contexts.extend(game_context)
        groups.extend(game_groups)
    return (
        np.vstack(rows),
        np.asarray(outcomes, dtype=np.float64),
        np.vstack(contexts),
        np.asarray(groups, dtype=np.int64),
        {"games_requested": n_games, "games_kept": n_games - draws, "draws": draws},
    )


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def logistic_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000))


def grouped_oof_probabilities(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def evaluate_potage(
    pipe: SoupPipe,
    fs,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    train_design = np.column_stack([logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), fs.X])
    validation_design = np.column_stack(
        [logit(np.clip(baseline_validation, 1e-6, 1 - 1e-6)), pipe.transform(X_validation)]
    )
    test_design = np.column_stack(
        [logit(np.clip(baseline_test, 1e-6, 1 - 1e-6)), pipe.transform(X_test)]
    )
    model = logistic_model().fit(train_design, y_train)
    validation_probability = model.predict_proba(validation_design)[:, 1]
    test_probability = model.predict_proba(test_design)[:, 1]
    return {
        "validation": probability_metrics(y_validation, validation_probability),
        "test": probability_metrics(y_test, test_probability),
        "features": [
            {"name": name, "coefficient": float(coefficient)}
            for name, coefficient in sorted(
                zip(fs.names, model[-1].coef_[0][1:]), key=lambda pair: -abs(pair[1])
            )
        ],
    }


def score_model(model, X_train, y_train, X_validation, y_validation, X_test, y_test):
    model.fit(X_train, y_train)
    return {
        "validation": probability_metrics(y_validation, model.predict_proba(X_validation)[:, 1]),
        "test": probability_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    }


def run_probe(n_games: int, seed: int, select: int) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(n_games, seed)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    discovery_local, validation_local = next(inner.split(X[discovery_val], y[discovery_val], groups[discovery_val]))
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]
    baseline_oof = grouped_oof_probabilities(context[discovery], y[discovery], groups[discovery])
    baseline = logistic_model().fit(context[discovery], y[discovery])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    baseline_scores = {
        "validation": probability_metrics(y[validation], baseline_validation),
        "test": probability_metrics(y[test], baseline_test),
    }
    tree_scores = score_model(
        HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed),
        np.column_stack([context[discovery], X[discovery]]), y[discovery],
        np.column_stack([context[validation], X[validation]]), y[validation],
        np.column_stack([context[test], X[test]]), y[test],
    )
    excluded = {"squared", "log", "sqrt", "cube", "tanh", "reciprocal", "rank", "am_ratio", "am_logratio", "am_power"}
    pipe = SoupPipe(
        X[discovery], y[discovery], FEATURE_NAMES,
        feature_types=["numeric"] * X.shape[1],
        X_val=X[validation], y_val=y[validation],
        y_base=baseline_oof, y_val_base=baseline_validation,
        task="regression", exclude_families=excluded, max_candidates=100_000,
    )
    raw = pipe.raw(select=None)
    selected_raw = pipe.select(raw, select=select, method="greedy")
    raw_scores = evaluate_potage(pipe, selected_raw, baseline_oof, baseline_validation, baseline_test, X[validation], X[test], y[discovery], y[validation], y[test])
    am = pipe.am(raw, select=None)
    selected_am = pipe.select(pipe.fuse(raw, am), select=select, method="greedy")
    am_scores = evaluate_potage(pipe, selected_am, baseline_oof, baseline_validation, baseline_test, X[validation], X[test], y[discovery], y[validation], y[test])
    return {
        "generation": generation,
        "seed": seed,
        "snapshot_ticks": list(SNAPSHOT_TICKS),
        "rows": len(X),
        "positive_rate": float(y.mean()),
        "split": {
            "discovery_rows": len(discovery), "validation_rows": len(validation), "test_rows": len(test),
            "discovery_games": int(np.unique(groups[discovery]).size),
            "validation_games": int(np.unique(groups[validation]).size),
            "test_games": int(np.unique(groups[test]).size),
        },
        "baseline": baseline_scores,
        "potage_raw": raw_scores,
        "potage_am": am_scores,
        "full_state_tree": tree_scores,
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; rows={result['rows']}")
    print(f"Split: {result['split']}")
    print("\nModel                     Brier     Log loss   ROC-AUC")
    print("------------------------------------------------------")
    for name in ("baseline", "potage_raw", "potage_am", "full_state_tree"):
        test = result[name]["test"]
        print(f"{name:24s}  {test['brier']:.4f}    {test['log_loss']:.4f}     {test['roc_auc']:.4f}")
    print("\nTop AM features:")
    for row in result["potage_am"]["features"][:12]:
        print(f"  {row['coefficient']:+.3f}  {row['name']}")


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = ["baseline", "Potage raw", "Potage AM", "full-state tree"]
    keys = ["baseline", "potage_raw", "potage_am", "full_state_tree"]
    validation = [result[key]["validation"]["brier"] for key in keys]
    test = [result[key]["test"]["brier"] for key in keys]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, validation, width, label="validation")
    ax.bar(x + width / 2, test, width, label="untouched games")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Real-time tempo-factor discovery")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--select", type=int, default=16)
    parser.add_argument(
        "--stability-seeds",
        type=int,
        nargs="*",
        default=[],
        help="additional simulations to summarize (slower)",
    )
    parser.add_argument(
        "--stability-games",
        type=int,
        default=1_000,
        help="games in each additional stability simulation",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.games, args.seed, args.select)
    print_summary(result)
    stability = []
    for seed in args.stability_seeds:
        repeat = run_probe(args.stability_games, seed, args.select)
        row = {
            "seed": seed,
            "games": args.stability_games,
            "baseline_test_brier": repeat["baseline"]["test"]["brier"],
            "potage_raw_test_brier": repeat["potage_raw"]["test"]["brier"],
            "potage_am_test_brier": repeat["potage_am"]["test"]["brier"],
            "tree_test_brier": repeat["full_state_tree"]["test"]["brier"],
            "top_am_features": [
                feature["name"] for feature in repeat["potage_am"]["features"][:8]
            ],
        }
        stability.append(row)
        print(
            f"stability seed {seed}: raw={row['potage_raw_test_brier']:.4f}, "
            f"AM={row['potage_am_test_brier']:.4f}, "
            f"tree={row['tree_test_brier']:.4f}"
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
