#!/usr/bin/env python3
"""Measure Reversi composition depth on nested training sets and one fixed test.

Unlike repeated standalone probes, every point in this experiment shares the
same validation and untouched test games. Confidence intervals resample whole
test games rather than individual snapshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import logit

try:
    from .reversi_probe import (
        FEATURE_NAMES,
        generate_dataset,
        grouped_oof_probabilities,
        logistic_model,
        probability_metrics,
    )
except ImportError:  # Direct execution: python examples/reversi_learning_curve.py
    from reversi_probe import (
        FEATURE_NAMES,
        generate_dataset,
        grouped_oof_probabilities,
        logistic_model,
        probability_metrics,
    )
from potage import SoupPipe


def fixed_game_split(
    groups: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nested-train order, validation games, and fixed test games."""
    game_ids = np.unique(groups).copy()
    np.random.RandomState(seed).shuffle(game_ids)
    n_test = max(1, int(round(0.20 * len(game_ids))))
    n_validation = max(1, int(round(0.20 * len(game_ids))))
    test_games = game_ids[:n_test]
    validation_games = game_ids[n_test : n_test + n_validation]
    train_order = game_ids[n_test + n_validation :]
    return train_order, validation_games, test_games


def rows_for_games(groups: np.ndarray, game_ids: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(groups, np.asarray(game_ids)))


def fit_residual_model(
    pipe: SoupPipe,
    feature_set,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_design = np.column_stack(
        [logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), feature_set.X]
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
    return (
        model.predict_proba(validation_design)[:, 1],
        model.predict_proba(test_design)[:, 1],
    )


def grouped_bootstrap_brier(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    rng: np.random.RandomState,
    repetitions: int,
) -> Tuple[float, float]:
    game_ids = np.unique(groups)
    squared_error = (y - probability) ** 2
    error_sums = np.array(
        [squared_error[groups == game].sum() for game in game_ids]
    )
    row_counts = np.array([(groups == game).sum() for game in game_ids])
    estimates = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        sample = rng.randint(0, len(game_ids), size=len(game_ids))
        estimates[repeat] = error_sums[sample].sum() / row_counts[sample].sum()
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def grouped_bootstrap_delta(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    groups: np.ndarray,
    rng: np.random.RandomState,
    repetitions: int,
) -> Dict[str, float]:
    game_ids = np.unique(groups)
    difference = (y - probability_a) ** 2 - (y - probability_b) ** 2
    difference_sums = np.array(
        [difference[groups == game].sum() for game in game_ids]
    )
    row_counts = np.array([(groups == game).sum() for game in game_ids])
    estimates = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        sample = rng.randint(0, len(game_ids), size=len(game_ids))
        estimates[repeat] = (
            difference_sums[sample].sum() / row_counts[sample].sum()
        )
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def evaluate_size(
    X: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    groups: np.ndarray,
    train_games: np.ndarray,
    validation_games: np.ndarray,
    test_games: np.ndarray,
    select: int,
    seed: int,
    bootstrap_repetitions: int,
) -> Dict[str, object]:
    train = rows_for_games(groups, train_games)
    validation = rows_for_games(groups, validation_games)
    test = rows_for_games(groups, test_games)

    baseline_oof = grouped_oof_probabilities(
        context[train], y[train], groups[train]
    )
    baseline = logistic_model().fit(context[train], y[train])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]

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
        X[train],
        y[train],
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

    probabilities = {"baseline": baseline_test}
    features: Dict[str, List[str]] = {}

    raw = pipe.raw(select=None)
    selected_raw = pipe.select(raw, select=select, method="greedy")
    _, probabilities["raw"] = fit_residual_model(
        pipe,
        selected_raw,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[train],
    )
    features["raw"] = list(selected_raw.names)

    am = pipe.am(raw, select=None)
    selected_am = pipe.select(pipe.fuse(raw, am), select=select, method="greedy")
    _, probabilities["am"] = fit_residual_model(
        pipe,
        selected_am,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[train],
    )
    features["am"] = list(selected_am.names)

    am2 = pipe.am(selected_am, select=None)
    selected_am2 = pipe.select(
        pipe.fuse(selected_am, am2), select=select, method="greedy"
    )
    _, probabilities["am2"] = fit_residual_model(
        pipe,
        selected_am2,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[train],
    )
    features["am2"] = list(selected_am2.names)

    rng = np.random.RandomState(seed)
    models = {}
    for name, probability in probabilities.items():
        metrics = probability_metrics(y[test], probability)
        low, high = grouped_bootstrap_brier(
            y[test],
            probability,
            groups[test],
            rng,
            bootstrap_repetitions,
        )
        metrics["brier_ci_low"] = low
        metrics["brier_ci_high"] = high
        models[name] = metrics

    return {
        "train_games": int(len(train_games)),
        "train_snapshots": int(len(train)),
        "models": models,
        "deltas": {
            "am_minus_raw": grouped_bootstrap_delta(
                y[test],
                probabilities["am"],
                probabilities["raw"],
                groups[test],
                rng,
                bootstrap_repetitions,
            ),
            "am2_minus_am": grouped_bootstrap_delta(
                y[test],
                probabilities["am2"],
                probabilities["am"],
                groups[test],
                rng,
                bootstrap_repetitions,
            ),
        },
        "selected_features": features,
    }


def run_learning_curve(
    games: int,
    sizes: Sequence[int],
    seed: int,
    select: int,
    bootstrap_repetitions: int,
) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(games, seed)
    train_order, validation_games, test_games = fixed_game_split(groups, seed + 17)
    usable_sizes = sorted(
        {min(size, len(train_order)) for size in sizes if size > 0}
    )
    if not usable_sizes:
        raise ValueError(f"no requested size fits {len(train_order)} training games")

    curve = []
    for size in usable_sizes:
        print(f"evaluating {size} training games...", flush=True)
        curve.append(
            evaluate_size(
                X,
                y,
                context,
                groups,
                train_order[:size],
                validation_games,
                test_games,
                select,
                seed + size,
                bootstrap_repetitions,
            )
        )
    return {
        "generation": generation,
        "seed": seed,
        "fixed_validation_games": int(len(validation_games)),
        "fixed_test_games": int(len(test_games)),
        "available_training_games": int(len(train_order)),
        "requested_sizes": list(sizes),
        "bootstrap_repetitions": bootstrap_repetitions,
        "curve": curve,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc

    curve = result["curve"]
    sizes = np.array([row["train_games"] for row in curve])
    labels = {"raw": "raw", "am": "AM", "am2": "AM²"}
    fig, (score_ax, delta_ax) = plt.subplots(
        2, 1, figsize=(8.7, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for key, label in labels.items():
        values = np.array([row["models"][key]["brier"] for row in curve])
        low = np.array([row["models"][key]["brier_ci_low"] for row in curve])
        high = np.array([row["models"][key]["brier_ci_high"] for row in curve])
        score_ax.plot(sizes, values, marker="o", label=label)
        score_ax.fill_between(sizes, low, high, alpha=0.12)
    score_ax.set_ylabel("Fixed-test Brier score")
    score_ax.set_title("Reversi composition-depth learning curve")
    score_ax.legend()
    score_ax.grid(alpha=0.2)

    for key, label in (("am_minus_raw", "AM − raw"), ("am2_minus_am", "AM² − AM")):
        values = np.array([row["deltas"][key]["estimate"] for row in curve])
        low = np.array([row["deltas"][key]["ci_low"] for row in curve])
        high = np.array([row["deltas"][key]["ci_high"] for row in curve])
        delta_ax.errorbar(
            sizes,
            values,
            yerr=np.vstack([values - low, high - values]),
            marker="o",
            capsize=3,
            label=label,
        )
    delta_ax.axhline(0.0, color="black", linewidth=1)
    delta_ax.set_ylabel("Brier difference")
    delta_ax.set_xlabel("Nested training games (log scale)")
    delta_ax.set_xscale("log", base=2)
    delta_ax.set_xticks(sizes, [str(size) for size in sizes])
    delta_ax.legend()
    delta_ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(result: Dict[str, object]) -> None:
    print(
        f"Fixed validation={result['fixed_validation_games']} games; "
        f"fixed test={result['fixed_test_games']} games"
    )
    print("games       raw        AM       AM²     AM-raw    AM²-AM")
    for row in result["curve"]:
        print(
            f"{row['train_games']:5d}  "
            f"{row['models']['raw']['brier']:.5f}  "
            f"{row['models']['am']['brier']:.5f}  "
            f"{row['models']['am2']['brier']:.5f}  "
            f"{row['deltas']['am_minus_raw']['estimate']:+.5f}  "
            f"{row['deltas']['am2_minus_am']['estimate']:+.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6_000)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[250, 500, 1_000, 2_000, 3_500]
    )
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--select", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_learning_curve(
        args.games,
        args.sizes,
        args.seed,
        args.select,
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
