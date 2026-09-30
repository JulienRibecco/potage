#!/usr/bin/env python3
"""Matched causal interventions on speed and initial geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.realtime_race_probe import play_game


FAR_POSITIONS = {
    1: np.asarray([0, 0], dtype=np.int64),
    -1: np.asarray([4, 4], dtype=np.int64),
}


def near_positions(focal: int) -> Dict[int, np.ndarray]:
    positions = {player: value.copy() for player, value in FAR_POSITIONS.items()}
    positions[focal] = np.asarray([2, 0] if focal == 1 else [2, 4], dtype=np.int64)
    return positions


def policy_noise(seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {1: rng.gumbel(size=(100, 3)), -1: rng.gumbel(size=(100, 3))}


def winner_from_game(result) -> int:
    _, y, _, _ = result
    if not y:
        return 0
    return 1 if y[0] > 0.5 else -1


def game_seed(seed: int, game_id: int, focal: int) -> int:
    return int((seed * 1_000_003 + game_id * 17 + (focal == -1)) % (2**32 - 1))


def paired_intervention(
    pairs: int,
    seed: int,
    focal: int,
    kind: str,
) -> Dict[str, object]:
    treated_wins: List[float] = []
    control_wins: List[float] = []
    deltas: List[float] = []
    draws = 0
    for game_id in range(pairs):
        current_seed = game_seed(seed, game_id, focal)
        noise = policy_noise((current_seed + 7919) % (2**32 - 1))
        if kind == "speed":
            control_kwargs = {
                "speed_override": {1: 1, -1: 1},
                "start_positions": FAR_POSITIONS,
            }
            treated_kwargs = {
                "speed_override": {1: 1, -1: 1},
                "start_positions": FAR_POSITIONS,
            }
            treated_kwargs["speed_override"][focal] = 2
        elif kind == "initial_position":
            control_kwargs = {
                "speed_override": {1: 1, -1: 1},
                "start_positions": FAR_POSITIONS,
            }
            treated_kwargs = {
                "speed_override": {1: 1, -1: 1},
                "start_positions": near_positions(focal),
            }
        else:
            raise ValueError(f"Unknown intervention kind: {kind}")
        control = play_game(
            game_id,
            np.random.RandomState(current_seed),
            policy_noise=noise,
            **control_kwargs,
        )
        treated = play_game(
            game_id,
            np.random.RandomState(current_seed),
            policy_noise=noise,
            **treated_kwargs,
        )
        control_winner = winner_from_game(control)
        treated_winner = winner_from_game(treated)
        if control_winner == 0 or treated_winner == 0:
            draws += 1
            continue
        control_win = float(control_winner == focal)
        treated_win = float(treated_winner == focal)
        control_wins.append(control_win)
        treated_wins.append(treated_win)
        deltas.append(treated_win - control_win)
    return {
        "kind": kind,
        "seed": seed,
        "focal_player": focal,
        "pairs_requested": pairs,
        "pairs_kept": len(deltas),
        "draw_pairs": draws,
        "control_win_rate": float(np.mean(control_wins)),
        "treated_win_rate": float(np.mean(treated_wins)),
        "paired_effect": float(np.mean(deltas)),
        "treated_wins_control_losses": int(np.sum(np.asarray(deltas) > 0)),
        "treated_losses_control_wins": int(np.sum(np.asarray(deltas) < 0)),
        "_deltas": deltas,
    }


def bootstrap_ci(values: Sequence[float], seed: int, draws: int = 4000) -> List[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_probe(pairs: int, seeds: Sequence[int]) -> Dict[str, object]:
    runs = []
    for kind in ("speed", "initial_position"):
        for seed in seeds:
            for focal in (1, -1):
                run = paired_intervention(pairs, seed, focal, kind)
                run["paired_effect_ci"] = bootstrap_ci(
                    run.pop("_deltas"), seed + 1000 + (focal == -1)
                )
                runs.append(run)
                print(
                    f"{kind:16s} seed {seed}, focal {focal:+d}: "
                    f"control={run['control_win_rate']:.3f}, "
                    f"treated={run['treated_win_rate']:.3f}, "
                    f"effect={run['paired_effect']:+.3f} "
                    f"CI=[{run['paired_effect_ci'][0]:+.3f}, {run['paired_effect_ci'][1]:+.3f}]"
                )
    return {
        "pairs_per_run": pairs,
        "seeds": list(seeds),
        "runs": runs,
        "mean_effect_by_kind": {
            kind: float(np.mean([r["paired_effect"] for r in runs if r["kind"] == kind]))
            for kind in ("speed", "initial_position")
        },
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = [f"{run['kind']}\nseed {run['seed']} / p{run['focal_player']}" for run in result["runs"]]
    effects = [run["paired_effect"] for run in result["runs"]]
    lower = [run["paired_effect"] - run["paired_effect_ci"][0] for run in result["runs"]]
    upper = [run["paired_effect_ci"][1] - run["paired_effect"] for run in result["runs"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.errorbar(x, effects, yerr=[lower, upper], fmt="o", capsize=4)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("paired win-rate effect")
    ax.set_title("Speed and initial-position interventions")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[19, 41, 97])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.pairs, args.seeds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
