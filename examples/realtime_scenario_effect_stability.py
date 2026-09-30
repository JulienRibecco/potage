#!/usr/bin/env python3
"""Measure scenario-effect feature and metric stability across random seeds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.realtime_scenario_effect_probe import run_probe


def consensus_counts(feature_lists: Iterable[Sequence[str]]) -> List[Dict[str, object]]:
    lists = [list(features) for features in feature_lists]
    counts = Counter(feature for features in lists for feature in set(features))
    return [
        {"name": name, "runs": int(count), "fraction": float(count / len(lists))}
        for name, count in counts.most_common()
    ]


def run_stability(
    pairs: int,
    seeds: Sequence[int],
    select: int,
    recipe: str,
    method: str,
    phases: Sequence[int],
) -> Dict[str, object]:
    runs = []
    for seed in seeds:
        print(f"Running paired-effect stability seed {seed}")
        result = run_probe(pairs, seed, select, recipe, method, phases)
        runs.append(result)
    summary: Dict[str, object] = {"seeds": list(seeds), "pairs": pairs, "runs": runs}
    by_kind: Dict[str, object] = {}
    for kind in ("speed", "initial_position"):
        by_space: Dict[str, object] = {}
        for space in ("context", "core", "scenario"):
            scores = [
                run["results"][kind]["models"][space]["potage"]["test"]
                for run in runs
            ]
            feature_lists = [
                [feature["name"] for feature in run["results"][kind]["models"][space]["potage"]["features"]]
                for run in runs
            ]
            scenario_counts = [
                sum(
                    any(marker in feature["name"] for marker in (
                        "our_speed", "opponent_speed", "speed_difference",
                        "our_start_to_center_node", "opponent_start_to_center_node",
                        "start_separation",
                    ))
                    for feature in run["results"][kind]["models"][space]["potage"]["features"]
                )
                for run in runs
            ]
            by_space[space] = {
                "brier_mean": float(np.mean([score["brier"] for score in scores])),
                "brier_std": float(np.std([score["brier"] for score in scores], ddof=1)),
                "auc_mean": float(np.mean([score["roc_auc"] for score in scores])),
                "auc_std": float(np.std([score["roc_auc"] for score in scores], ddof=1)),
                "scenario_terms_mean": float(np.mean(scenario_counts)),
                "top_feature_consensus": consensus_counts(feature_lists),
            }
        by_kind[kind] = by_space
    summary["summary"] = by_kind
    return summary


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = ["context", "core", "scenario"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
    for axis, kind in zip(axes, ("speed", "initial_position")):
        means = [result["summary"][kind][space]["brier_mean"] for space in labels]
        spreads = [result["summary"][kind][space]["brier_std"] for space in labels]
        axis.bar(x, means, yerr=spreads, capsize=4)
        axis.set_xticks(x, labels)
        axis.set_title(kind.replace("_", " "))
        axis.set_ylabel("test Brier ± seed SD")
    fig.suptitle("Scenario-effect stability across seeds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="*", default=[19, 41, 97])
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--method", default="correlation")
    parser.add_argument("--phases", type=int, nargs="*", default=list(range(9)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stability(
        args.pairs, args.seeds, args.select, args.recipe, args.method, args.phases
    )
    for kind in ("speed", "initial_position"):
        for space in ("context", "core", "scenario"):
            row = result["summary"][kind][space]
            print(
                f"{kind:16s} {space:8s}: Brier={row['brier_mean']:.4f}±{row['brier_std']:.4f}, "
                f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}, "
                f"scenario_terms={row['scenario_terms_mean']:.1f}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
