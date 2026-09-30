#!/usr/bin/env python3
"""Test the selected Reversi intervention across game phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    from .reversi_intervention import (
        compact_evaluation,
        evaluate_scenarios,
        make_scenarios,
    )
except ImportError:  # Direct execution from examples/
    from reversi_intervention import (
        compact_evaluation,
        evaluate_scenarios,
        make_scenarios,
    )


def run_phase_probe(
    pairs: int,
    opening_plies: Sequence[int],
    strength: float,
    seed: int,
    bootstrap_repetitions: int,
) -> Dict[str, object]:
    rows = []
    for index, plies in enumerate(opening_plies):
        print(f"testing combined intervention after {plies} opening plies...", flush=True)
        scenarios = make_scenarios(pairs, seed + index, plies)
        evaluation = evaluate_scenarios(scenarios, "combined", strength)
        rows.append(
            {
                "opening_plies": int(plies),
                **compact_evaluation(
                    evaluation, seed + 1_000 + index, bootstrap_repetitions
                ),
            }
        )
    return {
        "variant": "combined",
        "strength": strength,
        "pairs_per_phase": pairs,
        "games_per_pair": 2,
        "seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "phases": rows,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc

    rows = result["phases"]
    plies = np.array([row["opening_plies"] for row in rows])
    effects = np.array([row["paired_effect"]["estimate"] for row in rows])
    low = np.array([row["paired_effect"]["ci_low"] for row in rows])
    high = np.array([row["paired_effect"]["ci_high"] for row in rows])
    changed = np.array([row["decision_change_rate"] for row in rows])

    fig, (effect_ax, mechanism_ax) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    effect_ax.errorbar(
        plies,
        effects,
        yerr=np.vstack([effects - low, high - effects]),
        marker="o",
        capsize=5,
    )
    effect_ax.axhline(0.0, color="black", linewidth=1)
    effect_ax.set_xlabel("Random opening plies before intervention")
    effect_ax.set_ylabel("Paired-score gain over 0.5")
    effect_ax.set_title("Win effect across game phase")
    effect_ax.grid(alpha=0.2)

    mechanism_ax.plot(plies, changed, marker="o")
    mechanism_ax.set_xlabel("Random opening plies before intervention")
    mechanism_ax.set_ylabel("Challenger decisions changed")
    mechanism_ax.set_ylim(0.0, min(1.0, max(changed) * 1.25))
    mechanism_ax.set_title("Intervention exposure")
    mechanism_ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(result: Dict[str, object]) -> None:
    print("opening plies  paired score  effect [95% CI]  changed")
    for row in result["phases"]:
        effect = row["paired_effect"]
        print(
            f"{row['opening_plies']:13d}  {row['paired_score']:.4f}       "
            f"{effect['estimate']:+.4f} [{effect['ci_low']:+.4f}, "
            f"{effect['ci_high']:+.4f}]  {row['decision_change_rate']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--opening-plies", type=int, nargs="+", default=[0, 6, 10, 20, 30])
    parser.add_argument("--strength", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase_probe(
        args.pairs,
        args.opening_plies,
        args.strength,
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
