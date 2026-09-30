#!/usr/bin/env python3
"""Matched latency intervention for the simultaneous-action arena."""

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

from examples.realtime_race_probe import TICKS, play_game


def winner_from_game(result) -> int:
    """Recover the winner from the first player-1 snapshot label."""
    _, y, _, _ = result
    if not y:
        return 0
    return 1 if y[0] > 0.5 else -1


def policy_noise(seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        1: rng.gumbel(size=(TICKS, 3)),
        -1: rng.gumbel(size=(TICKS, 3)),
    }


def paired_intervention(
    pairs: int,
    seed: int,
    focal: int,
    low_latency: int = 0,
    high_latency: int = 3,
) -> Dict[str, object]:
    deltas: List[float] = []
    low_wins: List[float] = []
    high_wins: List[float] = []
    draws = 0
    for game_id in range(pairs):
        game_seed = (seed * 1_000_003 + game_id * 17 + (focal == -1)) % (2**32 - 1)
        noise = policy_noise((game_seed + 7919) % (2**32 - 1))
        low = play_game(
            game_id,
            np.random.RandomState(game_seed),
            latency_override={focal: low_latency},
            policy_noise=noise,
        )
        high = play_game(
            game_id,
            np.random.RandomState(game_seed),
            latency_override={focal: high_latency},
            policy_noise=noise,
        )
        low_winner = winner_from_game(low)
        high_winner = winner_from_game(high)
        if low_winner == 0 or high_winner == 0:
            draws += 1
            continue
        low_win = float(low_winner == focal)
        high_win = float(high_winner == focal)
        low_wins.append(low_win)
        high_wins.append(high_win)
        deltas.append(low_win - high_win)
    return {
        "seed": seed,
        "focal_player": focal,
        "pairs_requested": pairs,
        "pairs_kept": len(deltas),
        "draw_or_mismatch_pairs": draws,
        "low_latency": low_latency,
        "high_latency": high_latency,
        "low_win_rate": float(np.mean(low_wins)) if low_wins else float("nan"),
        "high_win_rate": float(np.mean(high_wins)) if high_wins else float("nan"),
        "paired_effect": float(np.mean(deltas)) if deltas else float("nan"),
        "discordant_pairs": int(np.sum(np.asarray(deltas) != 0)),
        "low_wins_high_losses": int(np.sum(np.asarray(deltas) > 0)),
        "low_losses_high_wins": int(np.sum(np.asarray(deltas) < 0)),
        "_deltas": deltas,
    }


def bootstrap_ci(values: Sequence[float], seed: int, draws: int = 4000) -> List[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    sample_means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample_means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(sample_means, 0.025)), float(np.quantile(sample_means, 0.975))]


def run_probe(
    pairs: int,
    seeds: Sequence[int],
    low_latency: int = 0,
    high_latency: int = 3,
) -> Dict[str, object]:
    runs = []
    for seed in seeds:
        for focal in (1, -1):
            run = paired_intervention(
                pairs, seed, focal, low_latency=low_latency, high_latency=high_latency
            )
            run["paired_effect_ci"] = bootstrap_ci(
                run.pop("_deltas"), seed=seed + 1000 + (focal == -1)
            )
            runs.append(run)
            print(
                f"seed {seed}, focal {focal:+d}: "
                f"low={run['low_win_rate']:.3f}, high={run['high_win_rate']:.3f}, "
                f"effect={run['paired_effect']:+.3f} "
                f"CI=[{run['paired_effect_ci'][0]:+.3f}, {run['paired_effect_ci'][1]:+.3f}]"
            )
    effects = [run["paired_effect"] for run in runs]
    return {
        "pairs_per_run": pairs,
        "seeds": list(seeds),
        "low_latency": low_latency,
        "high_latency": high_latency,
        "runs": runs,
        "mean_run_effect": float(np.mean(effects)),
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = [f"seed {run['seed']} / p{run['focal_player']}" for run in result["runs"]]
    effects = [run["paired_effect"] for run in result["runs"]]
    lower = [run["paired_effect"] - run["paired_effect_ci"][0] for run in result["runs"]]
    upper = [run["paired_effect_ci"][1] - run["paired_effect"] for run in result["runs"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.errorbar(x, effects, yerr=[lower, upper], fmt="o", capsize=4)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("paired win-rate effect (latency 0 − latency 3)")
    ax.set_title("Causal latency intervention")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[19, 41, 97])
    parser.add_argument("--low-latency", type=int, default=0, choices=range(4))
    parser.add_argument("--high-latency", type=int, default=3, choices=range(4))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.pairs, args.seeds, args.low_latency, args.high_latency)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
