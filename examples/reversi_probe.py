#!/usr/bin/env python3
"""Discover residual Reversi win factors from locally generated self-play.

The probe implements the game rules and a stochastic heuristic player without
external game dependencies. Several snapshots are recorded per game from the
player-to-move's perspective. Complete games, never individual snapshots, are
assigned to discovery, validation, and test partitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


BOARD_SIZE = 8
BLACK = 1
WHITE = -1
FULL_MASK = (1 << 64) - 1
A_FILE = sum(1 << (row * 8) for row in range(8))
H_FILE = sum(1 << (row * 8 + 7) for row in range(8))
NOT_A_FILE = FULL_MASK ^ A_FILE
NOT_H_FILE = FULL_MASK ^ H_FILE
CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
X_SQUARES = ((1, 1), (1, 6), (6, 1), (6, 6))
C_SQUARES = (
    (0, 1),
    (1, 0),
    (0, 6),
    (1, 7),
    (6, 0),
    (7, 1),
    (6, 7),
    (7, 6),
)
POSITION_WEIGHTS = np.array(
    [
        [30, -8, 4, 3, 3, 4, -8, 30],
        [-8, -12, -2, -2, -2, -2, -12, -8],
        [4, -2, 2, 1, 1, 2, -2, 4],
        [3, -2, 1, 0, 0, 1, -2, 3],
        [3, -2, 1, 0, 0, 1, -2, 3],
        [4, -2, 2, 1, 1, 2, -2, 4],
        [-8, -12, -2, -2, -2, -2, -12, -8],
        [30, -8, 4, 3, 3, 4, -8, 30],
    ],
    dtype=np.float64,
)
SNAPSHOT_PLIES = (12, 25, 38, 51, 56)


def shift_east(bits: int) -> int:
    return ((bits & NOT_H_FILE) << 1) & FULL_MASK


def shift_west(bits: int) -> int:
    return (bits & NOT_A_FILE) >> 1


def shift_south(bits: int) -> int:
    return (bits << 8) & FULL_MASK


def shift_north(bits: int) -> int:
    return bits >> 8


def shift_south_east(bits: int) -> int:
    return ((bits & NOT_H_FILE) << 9) & FULL_MASK


def shift_south_west(bits: int) -> int:
    return ((bits & NOT_A_FILE) << 7) & FULL_MASK


def shift_north_east(bits: int) -> int:
    return (bits & NOT_H_FILE) >> 7


def shift_north_west(bits: int) -> int:
    return (bits & NOT_A_FILE) >> 9


SHIFTS = (
    shift_east,
    shift_west,
    shift_south,
    shift_north,
    shift_south_east,
    shift_south_west,
    shift_north_east,
    shift_north_west,
)


def bit_count(bits: int) -> int:
    return bin(bits).count("1")


def bit(row: int, col: int) -> int:
    return 1 << (row * BOARD_SIZE + col)


def square_mask(squares: Sequence[Tuple[int, int]]) -> int:
    return sum(bit(row, col) for row, col in squares)


CORNER_MASK = square_mask(CORNERS)
X_SQUARE_MASK = square_mask(X_SQUARES)
C_SQUARE_MASK = square_mask(C_SQUARES)
EDGE_MASK = (
    sum(bit(0, col) | bit(7, col) for col in range(8))
    | sum(bit(row, 0) | bit(row, 7) for row in range(1, 7))
) & ~CORNER_MASK


def initial_board() -> Tuple[int, int]:
    black = bit(3, 4) | bit(4, 3)
    white = bit(3, 3) | bit(4, 4)
    return black, white


def player_bits(board: Tuple[int, int], player: int) -> Tuple[int, int]:
    black, white = board
    return (black, white) if player == BLACK else (white, black)


def flips_for_move(
    board: Tuple[int, int], player: int, position: int
) -> int:
    current, opponent = player_bits(board, player)
    move = 1 << position
    if move & (current | opponent):
        return 0
    flips = 0
    for shift in SHIFTS:
        captured = 0
        cursor = shift(move) & opponent
        while cursor:
            captured |= cursor
            beyond = shift(cursor)
            if beyond & current:
                flips |= captured
                break
            cursor = beyond & opponent
    return flips


def legal_move_bits(board: Tuple[int, int], player: int) -> int:
    current, opponent = player_bits(board, player)
    empty = FULL_MASK ^ (current | opponent)
    move_bits = 0
    for shift in SHIFTS:
        captured = shift(current) & opponent
        for _ in range(5):
            captured |= shift(captured) & opponent
        move_bits |= shift(captured) & empty
    return move_bits


def legal_move_count(board: Tuple[int, int], player: int) -> int:
    return bit_count(legal_move_bits(board, player))


def legal_moves(
    board: Tuple[int, int], player: int
) -> List[Tuple[int, int]]:
    move_bits = legal_move_bits(board, player)
    moves: List[Tuple[int, int]] = []
    while move_bits:
        move = move_bits & -move_bits
        position = move.bit_length() - 1
        moves.append((position, flips_for_move(board, player, position)))
        move_bits ^= move
    return moves


def apply_move(
    board: Tuple[int, int],
    player: int,
    move: Tuple[int, int],
) -> Tuple[int, int]:
    position, flips = move
    placed = 1 << position
    black, white = board
    if player == BLACK:
        return black | placed | flips, white & ~flips
    return black & ~flips, white | placed | flips


def adjacent_union(bits: int) -> int:
    adjacent = 0
    for shift in SHIFTS:
        adjacent |= shift(bits)
    return adjacent


def frontier_count(board: Tuple[int, int], player: int) -> int:
    current, opponent = player_bits(board, player)
    empty = FULL_MASK ^ (current | opponent)
    return bit_count(current & adjacent_union(empty))


def potential_mobility(board: Tuple[int, int], player: int) -> int:
    current, opponent = player_bits(board, player)
    empty = FULL_MASK ^ (current | opponent)
    return bit_count(empty & adjacent_union(opponent))


def square_count(board: Tuple[int, int], player: int, mask: int) -> int:
    current, _ = player_bits(board, player)
    return bit_count(current & mask)


def move_flip_stats(
    moves: Sequence[Tuple[int, int]]
) -> Tuple[float, float]:
    if not moves:
        return 0.0, 0.0
    counts = np.array([bit_count(move[1]) for move in moves], dtype=np.float64)
    return float(counts.max()), float(counts.mean())


FEATURE_NAMES = [
    "progress",
    "our_discs",
    "opponent_discs",
    "our_mobility",
    "opponent_mobility",
    "our_corners",
    "opponent_corners",
    "our_edges",
    "opponent_edges",
    "our_frontier",
    "opponent_frontier",
    "our_potential_mobility",
    "opponent_potential_mobility",
    "our_x_squares",
    "opponent_x_squares",
    "our_c_squares",
    "opponent_c_squares",
    "our_max_flips",
    "opponent_max_flips",
    "our_mean_flips",
    "opponent_mean_flips",
    "empty_parity",
]


def state_features(board: Tuple[int, int], player: int) -> np.ndarray:
    our_moves = legal_moves(board, player)
    opponent_moves = legal_moves(board, -player)
    our_max, our_mean = move_flip_stats(our_moves)
    opponent_max, opponent_mean = move_flip_stats(opponent_moves)
    current, opponent = player_bits(board, player)
    occupied = bit_count(current | opponent)
    empty = BOARD_SIZE * BOARD_SIZE - occupied
    return np.array(
        [
            occupied / (BOARD_SIZE * BOARD_SIZE),
            bit_count(current),
            bit_count(opponent),
            len(our_moves),
            len(opponent_moves),
            square_count(board, player, CORNER_MASK),
            square_count(board, -player, CORNER_MASK),
            square_count(board, player, EDGE_MASK),
            square_count(board, -player, EDGE_MASK),
            frontier_count(board, player),
            frontier_count(board, -player),
            potential_mobility(board, player),
            potential_mobility(board, -player),
            square_count(board, player, X_SQUARE_MASK),
            square_count(board, -player, X_SQUARE_MASK),
            square_count(board, player, C_SQUARE_MASK),
            square_count(board, -player, C_SQUARE_MASK),
            our_max,
            opponent_max,
            our_mean,
            opponent_mean,
            empty % 2,
        ],
        dtype=np.float64,
    )


def heuristic_move_score(
    board: Tuple[int, int],
    player: int,
    move: Tuple[int, int],
) -> float:
    position, flips = move
    row, col = divmod(position, BOARD_SIZE)
    positional = POSITION_WEIGHTS[row, col]
    # Deliberately cheap: the data generator should spend its time producing
    # independent games, while mobility remains an emergent state descriptor
    # for Potage to evaluate rather than a value hard-coded into every agent.
    return positional + 0.65 * bit_count(flips)


def choose_move(
    board: Tuple[int, int],
    player: int,
    moves: Sequence[Tuple[int, int]],
    skill: float,
    rng: np.random.RandomState,
) -> Tuple[int, int]:
    scores = np.array(
        [heuristic_move_score(board, player, move) for move in moves],
        dtype=np.float64,
    )
    noise_scale = 0.35 + 12.0 * (1.0 - skill) ** 2
    noisy_scores = skill * scores + rng.gumbel(scale=noise_scale, size=len(moves))
    return moves[int(np.argmax(noisy_scores))]


def play_game(
    game_id: int, rng: np.random.RandomState
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[int]]:
    board = initial_board()
    skills = {BLACK: rng.uniform(0.05, 1.0), WHITE: rng.uniform(0.05, 1.0)}
    player = BLACK
    ply = 0
    next_snapshot = 0
    snapshots = []

    while True:
        moves = legal_moves(board, player)
        if not moves and not legal_moves(board, -player):
            break

        while (
            next_snapshot < len(SNAPSHOT_PLIES)
            and ply >= SNAPSHOT_PLIES[next_snapshot]
        ):
            features = state_features(board, player)
            context = np.array(
                [
                    features[0],
                    features[1] - features[2],
                    skills[player] - skills[-player],
                    float(player == BLACK),
                ],
                dtype=np.float64,
            )
            snapshots.append((features, context, player))
            next_snapshot += 1

        if not moves:
            player = -player
            continue
        move = choose_move(board, player, moves, skills[player], rng)
        board = apply_move(board, player, move)
        ply += 1
        player = -player

    black_discs = bit_count(board[0])
    white_discs = bit_count(board[1])
    if black_discs == white_discs:
        return [], [], [], []
    winner = BLACK if black_discs > white_discs else WHITE
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
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000)
    )


def grouped_oof_probabilities(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def score_probability_model(
    model,
    X_discovery: np.ndarray,
    y_discovery: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[object, Dict[str, Dict[str, float]]]:
    model.fit(X_discovery, y_discovery)
    return model, {
        "validation": probability_metrics(
            y_validation, model.predict_proba(X_validation)[:, 1]
        ),
        "test": probability_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    }


def evaluate_potage_features(
    pipe: SoupPipe,
    fs,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_discovery: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    discovery_design = np.column_stack(
        [logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), fs.X]
    )
    validation_design = np.column_stack(
        [
            logit(np.clip(baseline_validation, 1e-6, 1 - 1e-6)),
            pipe.transform(X_validation),
        ]
    )
    test_design = np.column_stack(
        [
            logit(np.clip(baseline_test, 1e-6, 1 - 1e-6)),
            pipe.transform(X_test),
        ]
    )
    model, scores = score_probability_model(
        logistic_model(),
        discovery_design,
        y_discovery,
        validation_design,
        y_validation,
        test_design,
        y_test,
    )
    coefficients = model[-1].coef_[0][1:]
    scores.update(
        {
            "n_features": len(fs.names),
            "features": [
                {
                    "name": name,
                    "standardized_coefficient": float(coefficient),
                }
                for name, coefficient in sorted(
                    zip(fs.names, coefficients), key=lambda pair: -abs(pair[1])
                )
            ],
            "family_counts": dict(sorted(pipe.stages[-1].family_census.items())),
        }
    )
    return scores


def run_probe(n_games: int, seed: int, select: int) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(n_games, seed)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    discovery_local, validation_local = next(
        inner.split(
            X[discovery_val], y[discovery_val], groups[discovery_val]
        )
    )
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]

    baseline_oof = grouped_oof_probabilities(
        context[discovery], y[discovery], groups[discovery]
    )
    baseline, baseline_scores = score_probability_model(
        logistic_model(),
        context[discovery],
        y[discovery],
        context[validation],
        y[validation],
        context[test],
        y[test],
    )
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]

    _, tree_scores = score_probability_model(
        HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ),
        np.column_stack([context[discovery], X[discovery]]),
        y[discovery],
        np.column_stack([context[validation], X[validation]]),
        y[validation],
        np.column_stack([context[test], X[test]]),
        y[test],
    )

    excluded = {
        "squared",
        "log",
        "sqrt",
        "cube",
        "tanh",
        "reciprocal",
        "rank",
        "am_ratio",
        "am_logratio",
        "am_power",
    }
    pipe = SoupPipe(
        X[discovery],
        y[discovery],
        FEATURE_NAMES,
        feature_types=["numeric"] * X.shape[1],
        X_val=X[validation],
        y_val=y[validation],
        y_base=baseline_oof,
        y_val_base=baseline_validation,
        task="regression",
        exclude_families=excluded,
        max_candidates=100_000,
    )
    raw = pipe.raw(select=None)
    selected_raw = pipe.select(raw, select=select, method="greedy")
    raw_scores = evaluate_potage_features(
        pipe,
        selected_raw,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[discovery],
        y[validation],
        y[test],
    )
    am = pipe.am(raw, select=None)
    selected_am = pipe.select(pipe.fuse(raw, am), select=select, method="greedy")
    am_scores = evaluate_potage_features(
        pipe,
        selected_am,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[discovery],
        y[validation],
        y[test],
    )
    am2 = pipe.am(selected_am, select=None)
    selected_am2 = pipe.select(
        pipe.fuse(selected_am, am2), select=select, method="greedy"
    )
    am2_scores = evaluate_potage_features(
        pipe,
        selected_am2,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[discovery],
        y[validation],
        y[test],
    )

    return {
        "generation": generation,
        "seed": seed,
        "snapshot_plies": list(SNAPSHOT_PLIES),
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
        "baseline": baseline_scores,
        "potage_raw": raw_scores,
        "potage_am": am_scores,
        "potage_am2": am2_scores,
        "full_state_tree": tree_scores,
        "pipeline_recap": pipe.recap(),
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; rows={result['rows']}")
    print(f"Split: {result['split']}")
    print("\nModel                     Brier     Log loss   ROC-AUC")
    print("------------------------------------------------------")
    for name in (
        "baseline",
        "potage_raw",
        "potage_am",
        "potage_am2",
        "full_state_tree",
    ):
        test = result[name]["test"]
        print(
            f"{name:24s}  {test['brier']:.4f}    "
            f"{test['log_loss']:.4f}     {test['roc_auc']:.4f}"
        )
    print("\nTop AM features:")
    for row in result["potage_am"]["features"][:12]:
        print(f"  {row['standardized_coefficient']:+.3f}  {row['name']}")


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = [
        "baseline",
        "Potage raw",
        "Potage AM",
        "Potage AM²",
        "full-state tree",
    ]
    keys = [
        "baseline",
        "potage_raw",
        "potage_am",
        "potage_am2",
        "full_state_tree",
    ]
    validation = [result[key]["validation"]["brier"] for key in keys]
    test = [result[key]["test"]["brier"] for key in keys]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, validation, width, label="validation")
    ax.bar(x + width / 2, test, width, label="untouched games")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Reversi residual win-factor discovery")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=107)
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
            "potage_am2_test_brier": repeat["potage_am2"]["test"]["brier"],
            "tree_test_brier": repeat["full_state_tree"]["test"]["brier"],
            "top_am_features": [
                feature["name"]
                for feature in repeat["potage_am"]["features"][:6]
            ],
        }
        stability.append(row)
        print(
            f"stability seed {seed}: raw={row['potage_raw_test_brier']:.4f}, "
            f"AM={row['potage_am_test_brier']:.4f}, "
            f"AM²={row['potage_am2_test_brier']:.4f}"
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
