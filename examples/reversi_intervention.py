#!/usr/bin/env python3
"""Intervene on cross-fitted Reversi factors and measure paired win effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from .reversi_probe import (
        BLACK,
        BOARD_SIZE,
        CORNER_MASK,
        C_SQUARE_MASK,
        POSITION_WEIGHTS,
        WHITE,
        apply_move,
        bit,
        bit_count,
        frontier_count,
        initial_board,
        legal_move_count,
        legal_moves,
        player_bits,
        square_count,
    )
except ImportError:  # Direct execution from examples/
    from reversi_probe import (
        BLACK,
        BOARD_SIZE,
        CORNER_MASK,
        C_SQUARE_MASK,
        POSITION_WEIGHTS,
        WHITE,
        apply_move,
        bit,
        bit_count,
        frontier_count,
        initial_board,
        legal_move_count,
        legal_moves,
        player_bits,
        square_count,
    )


CORNER_C_MASKS = (
    (bit(0, 0), bit(0, 1) | bit(1, 0)),
    (bit(0, 7), bit(0, 6) | bit(1, 7)),
    (bit(7, 0), bit(6, 0) | bit(7, 1)),
    (bit(7, 7), bit(6, 7) | bit(7, 6)),
)
VARIANTS = ("mobility", "safety", "combined")


def conditional_c_score(board: Tuple[int, int], player: int) -> float:
    current, _ = player_bits(board, player)
    score = 0
    for corner, adjacent in CORNER_C_MASKS:
        occupied = bit_count(current & adjacent)
        score += occupied if current & corner else -occupied
    return score / 2.0


def mobility_component(board: Tuple[int, int], player: int) -> float:
    return (
        legal_move_count(board, player) - legal_move_count(board, -player)
    ) / 12.0


def safety_component(board: Tuple[int, int], player: int) -> float:
    frontier_safety = (
        frontier_count(board, -player) - frontier_count(board, player)
    ) / 24.0
    corner_control = (
        square_count(board, player, CORNER_MASK)
        - square_count(board, -player, CORNER_MASK)
    ) / 4.0
    c_safety = (
        conditional_c_score(board, player)
        - conditional_c_score(board, -player)
    ) / 4.0
    return frontier_safety + corner_control + c_safety


def factor_components(
    board: Tuple[int, int], player: int, move: Tuple[int, int]
) -> Dict[str, float]:
    after = apply_move(board, player, move)
    mobility = mobility_component(after, player)
    safety = safety_component(after, player)
    return {
        "mobility": mobility,
        "safety": safety,
        "combined": mobility + safety,
    }


def factor_value(
    board: Tuple[int, int],
    player: int,
    move: Tuple[int, int],
    variant: str,
) -> float:
    after = apply_move(board, player, move)
    if variant == "mobility":
        return mobility_component(after, player)
    if variant == "safety":
        return safety_component(after, player)
    return mobility_component(after, player) + safety_component(after, player)


def choose_policy_move(
    board: Tuple[int, int],
    player: int,
    moves: Sequence[Tuple[int, int]],
    variant: str,
    strength: float,
    rng: np.random.RandomState,
) -> Tuple[Tuple[int, int], bool, float]:
    base_scores = np.array(
        [
            POSITION_WEIGHTS[divmod(position, BOARD_SIZE)]
            + 0.65 * bit_count(flips)
            for position, flips in moves
        ],
        dtype=np.float64,
    )
    noise = rng.gumbel(scale=2.0, size=len(moves))
    baseline_index = int(np.argmax(base_scores + noise))
    factors = np.array(
        [factor_value(board, player, move, variant) for move in moves]
    )
    selected_index = int(np.argmax(base_scores + noise + strength * factors))
    return (
        moves[selected_index],
        selected_index != baseline_index,
        float(factors[selected_index] - factors[baseline_index]),
    )


def random_opening(
    rng: np.random.RandomState, plies: int
) -> Tuple[Tuple[int, int], int]:
    board = initial_board()
    player = BLACK
    completed = 0
    while completed < plies:
        moves = legal_moves(board, player)
        if not moves:
            if not legal_moves(board, -player):
                break
            player = -player
            continue
        board = apply_move(board, player, moves[rng.randint(len(moves))])
        player = -player
        completed += 1
    return board, player


def play_from_opening(
    board: Tuple[int, int],
    player: int,
    challenger: int,
    variant: str,
    strength: float,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    changed = 0
    decisions = 0
    factor_gain = 0.0
    while True:
        moves = legal_moves(board, player)
        if not moves:
            if not legal_moves(board, -player):
                break
            player = -player
            continue
        applied_strength = strength if player == challenger else 0.0
        move, was_changed, gain = choose_policy_move(
            board, player, moves, variant, applied_strength, rng
        )
        if player == challenger:
            decisions += 1
            changed += int(was_changed)
            factor_gain += gain
        board = apply_move(board, player, move)
        player = -player

    black_discs = bit_count(board[0])
    white_discs = bit_count(board[1])
    if black_discs == white_discs:
        score = 0.5
    else:
        winner = BLACK if black_discs > white_discs else WHITE
        score = float(winner == challenger)
    margin = (black_discs - white_discs) * challenger
    return {
        "score": score,
        "margin": float(margin),
        "decisions": float(decisions),
        "changed": float(changed),
        "factor_gain": factor_gain,
    }


def make_scenarios(
    pairs: int, seed: int, opening_plies: int
) -> List[Tuple[Tuple[int, int], int, int]]:
    rng = np.random.RandomState(seed)
    scenarios = []
    for _ in range(pairs):
        board, player = random_opening(rng, opening_plies)
        scenarios.append((board, player, int(rng.randint(1, 2**31 - 1))))
    return scenarios


def evaluate_scenarios(
    scenarios: Sequence[Tuple[Tuple[int, int], int, int]],
    variant: str,
    strength: float,
) -> Dict[str, object]:
    pair_scores = []
    margins = []
    decisions = changed = 0.0
    factor_gain = 0.0
    outcomes = {"wins": 0, "draws": 0, "losses": 0}
    for board, player, seed in scenarios:
        games = [
            play_from_opening(
                board, player, challenger, variant, strength, seed
            )
            for challenger in (BLACK, WHITE)
        ]
        pair_scores.append(float(np.mean([game["score"] for game in games])))
        margins.extend(game["margin"] for game in games)
        decisions += sum(game["decisions"] for game in games)
        changed += sum(game["changed"] for game in games)
        factor_gain += sum(game["factor_gain"] for game in games)
        for game in games:
            key = "wins" if game["score"] == 1 else "losses" if game["score"] == 0 else "draws"
            outcomes[key] += 1
    return {
        "pair_scores": pair_scores,
        "paired_score": float(np.mean(pair_scores)),
        "mean_disc_margin": float(np.mean(margins)),
        "decision_change_rate": float(changed / decisions) if decisions else 0.0,
        "mean_factor_gain_per_decision": float(factor_gain / decisions)
        if decisions
        else 0.0,
        **outcomes,
    }


def paired_effect_interval(
    pair_scores: Sequence[float], seed: int, repetitions: int
) -> Dict[str, float]:
    scores = np.asarray(pair_scores, dtype=np.float64)
    rng = np.random.RandomState(seed)
    effects = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        effects[repeat] = scores[rng.randint(0, len(scores), len(scores))].mean() - 0.5
    low, high = np.percentile(effects, [2.5, 97.5])
    return {
        "estimate": float(scores.mean() - 0.5),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def compact_evaluation(
    evaluation: Dict[str, object], seed: int, repetitions: int
) -> Dict[str, object]:
    result = {
        key: value for key, value in evaluation.items() if key != "pair_scores"
    }
    result["paired_effect"] = paired_effect_interval(
        evaluation["pair_scores"], seed, repetitions
    )
    return result


def run_intervention(
    tune_pairs: int,
    test_pairs: int,
    strengths: Sequence[float],
    opening_plies: int,
    seed: int,
    bootstrap_repetitions: int,
) -> Dict[str, object]:
    tune_scenarios = make_scenarios(tune_pairs, seed, opening_plies)
    test_scenarios = make_scenarios(test_pairs, seed + 1, opening_plies)
    tuning = {}
    selected = {}
    for variant_index, variant in enumerate(VARIANTS):
        print(f"tuning {variant} intervention...", flush=True)
        rows = []
        for strength in strengths:
            evaluation = evaluate_scenarios(tune_scenarios, variant, strength)
            rows.append(
                {
                    "strength": float(strength),
                    **compact_evaluation(
                        evaluation,
                        seed + 100 * variant_index + int(10 * strength),
                        bootstrap_repetitions,
                    ),
                }
            )
        tuning[variant] = rows
        best = max(rows, key=lambda row: row["paired_score"])
        print(
            f"testing {variant} at strength {best['strength']:g}...", flush=True
        )
        evaluation = evaluate_scenarios(
            test_scenarios, variant, best["strength"]
        )
        selected[variant] = {
            "strength": best["strength"],
            **compact_evaluation(
                evaluation,
                seed + 10_000 + variant_index,
                bootstrap_repetitions,
            ),
        }
    return {
        "seed": seed,
        "tune_pairs": tune_pairs,
        "test_pairs": test_pairs,
        "games_per_pair": 2,
        "opening_plies": opening_plies,
        "strengths": list(strengths),
        "bootstrap_repetitions": bootstrap_repetitions,
        "tuning": tuning,
        "untouched_test": selected,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc

    fig, (tune_ax, test_ax) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for variant in VARIANTS:
        rows = result["tuning"][variant]
        tune_ax.plot(
            [row["strength"] for row in rows],
            [row["paired_score"] for row in rows],
            marker="o",
            label=variant,
        )
    tune_ax.axhline(0.5, color="black", linewidth=1)
    tune_ax.set_xlabel("Intervention strength")
    tune_ax.set_ylabel("Tuning paired score")
    tune_ax.set_title("Intervention tuning")
    tune_ax.legend()
    tune_ax.grid(alpha=0.2)

    test_rows = [result["untouched_test"][variant] for variant in VARIANTS]
    effects = np.array([row["paired_effect"]["estimate"] for row in test_rows])
    low = np.array([row["paired_effect"]["ci_low"] for row in test_rows])
    high = np.array([row["paired_effect"]["ci_high"] for row in test_rows])
    x = np.arange(len(VARIANTS))
    test_ax.errorbar(
        x,
        effects,
        yerr=np.vstack([effects - low, high - effects]),
        fmt="o",
        markersize=8,
        capsize=5,
    )
    test_ax.axhline(0.0, color="black", linewidth=1)
    test_ax.set_xticks(x, VARIANTS)
    test_ax.set_ylabel("Untouched paired-score gain over 0.5")
    test_ax.set_title("Causal test on fresh openings")
    test_ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(result: Dict[str, object]) -> None:
    print("variant       strength  paired score  effect [95% CI]  changed")
    for variant in VARIANTS:
        row = result["untouched_test"][variant]
        effect = row["paired_effect"]
        print(
            f"{variant:12s} {row['strength']:8.1f}  {row['paired_score']:.4f}       "
            f"{effect['estimate']:+.4f} [{effect['ci_low']:+.4f}, "
            f"{effect['ci_high']:+.4f}]  {row['decision_change_rate']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-pairs", type=int, default=100)
    parser.add_argument("--test-pairs", type=int, default=500)
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    )
    parser.add_argument("--opening-plies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_intervention(
        args.tune_pairs,
        args.test_pairs,
        args.strengths,
        args.opening_plies,
        args.seed,
        args.bootstrap_repetitions,
    )
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
