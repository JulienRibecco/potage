#!/usr/bin/env python3
"""Discover tactical Connect Four win factors with grouped self-play."""

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


ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1
SNAPSHOT_PLIES = (4, 8, 12, 16, 20, 24, 28, 32, 36)
CENTER_WEIGHTS = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])


def legal_columns(board: np.ndarray) -> List[int]:
    return [column for column in range(COLS) if board[0, column] == EMPTY]


def drop_piece(
    board: np.ndarray, player: int, column: int
) -> Tuple[np.ndarray, int]:
    rows = np.flatnonzero(board[:, column] == EMPTY)
    if len(rows) == 0:
        raise ValueError("column is full")
    row = int(rows[-1])
    after = board.copy()
    after[row, column] = player
    return after, row


def line_length(board: np.ndarray, row: int, column: int, player: int) -> int:
    best = 1
    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        length = 1
        for direction in (-1, 1):
            next_row = row + direction * row_step
            next_column = column + direction * column_step
            while (
                0 <= next_row < ROWS
                and 0 <= next_column < COLS
                and board[next_row, next_column] == player
            ):
                length += 1
                next_row += direction * row_step
                next_column += direction * column_step
        best = max(best, length)
    return best


def is_winning_drop(board: np.ndarray, player: int, column: int) -> bool:
    after, row = drop_piece(board, player, column)
    return line_length(after, row, column, player) >= 4


def immediate_wins(board: np.ndarray, player: int) -> List[int]:
    return [
        column
        for column in legal_columns(board)
        if is_winning_drop(board, player, column)
    ]


def threat_count(board: np.ndarray, player: int) -> int:
    """Count moves that create at least two immediate winning replies."""
    count = 0
    for column in legal_columns(board):
        after, _ = drop_piece(board, player, column)
        if len(immediate_wins(after, player)) >= 2:
            count += 1
    return count


def max_run(board: np.ndarray, player: int) -> int:
    best = 0
    for row in range(ROWS):
        for column in range(COLS):
            if board[row, column] == player:
                best = max(best, line_length(board, row, column, player))
    return best


def count_open_threes(board: np.ndarray, player: int) -> int:
    count = 0
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(ROWS):
        for column in range(COLS):
            for row_step, column_step in directions:
                cells = [
                    (row + index * row_step, column + index * column_step)
                    for index in range(4)
                ]
                if not all(
                    0 <= cell_row < ROWS and 0 <= cell_column < COLS
                    for cell_row, cell_column in cells
                ):
                    continue
                values = [board[cell_row, cell_column] for cell_row, cell_column in cells]
                if values.count(player) == 3 and values.count(EMPTY) == 1:
                    empty_index = values.index(EMPTY)
                    empty_row, empty_column = cells[empty_index]
                    if empty_row == ROWS - 1 or board[empty_row + 1, empty_column] != EMPTY:
                        count += 1
    return count


FEATURE_NAMES = [
    "progress",
    "our_discs",
    "opponent_discs",
    "our_legal_columns",
    "opponent_legal_columns",
    "our_immediate_wins",
    "opponent_immediate_wins",
    "our_double_threats",
    "opponent_double_threats",
    "our_center_control",
    "opponent_center_control",
    "our_edge_discs",
    "opponent_edge_discs",
    "our_max_run",
    "opponent_max_run",
    "our_open_threes",
    "opponent_open_threes",
    "our_stack_height",
    "opponent_stack_height",
    "empty_parity",
]


def state_features(board: np.ndarray, player: int) -> np.ndarray:
    occupied = int(np.count_nonzero(board))
    our = board == player
    opponent = board == -player
    our_columns = np.flatnonzero(our).tolist()
    opponent_columns = np.flatnonzero(opponent).tolist()
    our_stack_height = float(np.mean(np.sum(our, axis=0)))
    opponent_stack_height = float(np.mean(np.sum(opponent, axis=0)))
    return np.asarray(
        [
            occupied / (ROWS * COLS),
            float(np.count_nonzero(our)),
            float(np.count_nonzero(opponent)),
            len(legal_columns(board)),
            len(legal_columns(board)),
            len(immediate_wins(board, player)),
            len(immediate_wins(board, -player)),
            threat_count(board, player),
            threat_count(board, -player),
            float(np.sum(our[:, 1:6])),
            float(np.sum(opponent[:, 1:6])),
            float(np.sum(our[:, [0, 6]])),
            float(np.sum(opponent[:, [0, 6]])),
            max_run(board, player),
            max_run(board, -player),
            count_open_threes(board, player),
            count_open_threes(board, -player),
            our_stack_height,
            opponent_stack_height,
            (ROWS * COLS - occupied) % 2,
        ],
        dtype=np.float64,
    )


def heuristic_move_score(board: np.ndarray, player: int, column: int) -> float:
    after, row = drop_piece(board, player, column)
    score = 1.5 * CENTER_WEIGHTS[column] + 0.25 * row
    if line_length(after, row, column, player) >= 4:
        score += 50.0
    opponent_wins_before = len(immediate_wins(board, -player))
    opponent_wins_after = len(immediate_wins(after, -player))
    score += 18.0 * (opponent_wins_before - opponent_wins_after)
    score += 1.5 * len(immediate_wins(after, player))
    return score


def choose_move(
    board: np.ndarray,
    player: int,
    skill: float,
    rng: np.random.RandomState,
) -> int:
    columns = legal_columns(board)
    scores = np.asarray(
        [heuristic_move_score(board, player, column) for column in columns]
    )
    noise_scale = 0.2 + 12.0 * (1.0 - skill) ** 2
    return columns[int(np.argmax(skill * scores + rng.gumbel(scale=noise_scale, size=len(columns))))]


def play_game(
    game_id: int, rng: np.random.RandomState
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[int]]:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    skills = {
        PLAYER_ONE: rng.uniform(0.05, 1.0),
        PLAYER_TWO: rng.uniform(0.05, 1.0),
    }
    player = PLAYER_ONE
    ply = 0
    next_snapshot = 0
    snapshots = []
    winner = None
    while True:
        columns = legal_columns(board)
        if not columns:
            break
        while (
            next_snapshot < len(SNAPSHOT_PLIES)
            and ply >= SNAPSHOT_PLIES[next_snapshot]
        ):
            features = state_features(board, player)
            context = np.asarray(
                [
                    features[0],
                    features[1] - features[2],
                    skills[player] - skills[-player],
                    float(player == PLAYER_ONE),
                ],
                dtype=np.float64,
            )
            snapshots.append((features, context, player))
            next_snapshot += 1
        column = choose_move(board, player, skills[player], rng)
        board, row = drop_piece(board, player, column)
        ply += 1
        if line_length(board, row, column, player) >= 4:
            winner = player
            break
        player = -player

    if winner is None:
        return [], [], [], []
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


def grouped_oof_probabilities(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
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
    train_design = np.column_stack(
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
    model = logistic_model().fit(train_design, y_train)
    validation_probability = model.predict_proba(validation_design)[:, 1]
    test_probability = model.predict_proba(test_design)[:, 1]
    return {
        "validation": probability_metrics(y_validation, validation_probability),
        "test": probability_metrics(y_test, test_probability),
        "features": [
            {
                "name": name,
                "coefficient": float(coefficient),
            }
            for name, coefficient in sorted(
                zip(fs.names, model[-1].coef_[0][1:]),
                key=lambda pair: -abs(pair[1]),
            )
        ],
    }


def run_probe(n_games: int, seed: int, select: int) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(n_games, seed)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    discovery_local, validation_local = next(
        inner.split(X[discovery_val], y[discovery_val], groups[discovery_val])
    )
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]

    baseline_oof = grouped_oof_probabilities(
        context[discovery], y[discovery], groups[discovery]
    )
    baseline = logistic_model().fit(context[discovery], y[discovery])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    baseline_scores = {
        "validation": probability_metrics(y[validation], baseline_validation),
        "test": probability_metrics(y[test], baseline_test),
    }

    _, tree_scores = _score_model(
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
    raw_scores = evaluate_potage(
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
    am_scores = evaluate_potage(
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
        "full_state_tree": tree_scores,
    }


def _score_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    model.fit(X_train, y_train)
    return model, {
        "validation": probability_metrics(
            y_validation, model.predict_proba(X_validation)[:, 1]
        ),
        "test": probability_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; rows={result['rows']}")
    print(f"Split: {result['split']}")
    print("\nModel                     Brier     Log loss   ROC-AUC")
    print("------------------------------------------------------")
    for name in ("baseline", "potage_raw", "potage_am", "full_state_tree"):
        test = result[name]["test"]
        print(
            f"{name:24s}  {test['brier']:.4f}    "
            f"{test['log_loss']:.4f}     {test['roc_auc']:.4f}"
        )
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
    ax.set_title("Connect Four tactical win-factor discovery")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=601)
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
