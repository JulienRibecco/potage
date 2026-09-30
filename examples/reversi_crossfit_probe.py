#!/usr/bin/env python3
"""Cross-fit Reversi feature discovery and test consensus predictors."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold

try:
    from .reversi_learning_curve import (
        fit_residual_model,
        grouped_bootstrap_brier,
        grouped_bootstrap_delta,
        rows_for_games,
    )
    from .reversi_probe import (
        FEATURE_NAMES,
        generate_dataset,
        grouped_oof_probabilities,
        logistic_model,
        probability_metrics,
    )
except ImportError:  # Direct execution from examples/
    from reversi_learning_curve import (
        fit_residual_model,
        grouped_bootstrap_brier,
        grouped_bootstrap_delta,
        rows_for_games,
    )
    from reversi_probe import (
        FEATURE_NAMES,
        generate_dataset,
        grouped_oof_probabilities,
        logistic_model,
        probability_metrics,
    )

from potage import SoupPipe


EXCLUDED_FAMILIES = {
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


def ingredients_in_expression(expression: str) -> List[str]:
    return [
        name
        for name in FEATURE_NAMES
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", expression)
    ]


def split_development_test(
    groups: np.ndarray, seed: int, test_fraction: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    game_ids = np.unique(groups).copy()
    np.random.RandomState(seed).shuffle(game_ids)
    n_test = max(1, int(round(test_fraction * len(game_ids))))
    return game_ids[n_test:], game_ids[:n_test]


def new_pipe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_oof: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray = None,
    baseline_validation: np.ndarray = None,
) -> SoupPipe:
    return SoupPipe(
        X_train,
        y_train,
        FEATURE_NAMES,
        feature_types=["numeric"] * X_train.shape[1],
        X_val=X_validation,
        y_val=y_validation,
        y_base=baseline_oof,
        y_val_base=baseline_validation,
        task="regression",
        exclude_families=EXCLUDED_FAMILIES,
        max_candidates=100_000,
    )


def fit_discovery_fold(
    X: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    select: int,
) -> Dict[str, object]:
    baseline_oof = grouped_oof_probabilities(
        context[train], y[train], groups[train]
    )
    baseline = logistic_model().fit(context[train], y[train])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    pipe = new_pipe(
        X[train],
        y[train],
        baseline_oof,
        X[validation],
        y[validation],
        baseline_validation,
    )

    raw = pipe.raw(select=None)
    selected_raw = pipe.select(raw, select=select, method="greedy")
    _, raw_test = fit_residual_model(
        pipe,
        selected_raw,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[train],
    )

    am = pipe.am(raw, select=None)
    selected_am = pipe.select(pipe.fuse(raw, am), select=select, method="greedy")
    _, am_test = fit_residual_model(
        pipe,
        selected_am,
        baseline_oof,
        baseline_validation,
        baseline_test,
        X[validation],
        X[test],
        y[train],
    )
    return {
        "baseline_test": baseline_test,
        "raw_test": raw_test,
        "am_test": am_test,
        "raw_features": list(selected_raw.names),
        "am_features": list(selected_am.names),
    }


def fit_consensus_models(
    X: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    groups: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
    ingredient_names: Sequence[str],
    expression_names: Sequence[str],
) -> Dict[str, object]:
    baseline_oof = grouped_oof_probabilities(
        context[development], y[development], groups[development]
    )
    baseline = logistic_model().fit(context[development], y[development])
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    base_train = logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6))
    base_test = logit(np.clip(baseline_test, 1e-6, 1 - 1e-6))

    ingredient_indices = [FEATURE_NAMES.index(name) for name in ingredient_names]
    ingredient_model = logistic_model().fit(
        np.column_stack([base_train, X[development][:, ingredient_indices]]),
        y[development],
    )
    ingredient_test = ingredient_model.predict_proba(
        np.column_stack([base_test, X[test][:, ingredient_indices]])
    )[:, 1]

    pipe = new_pipe(
        X[development], y[development], baseline_oof, X[test]
    )
    raw = pipe.raw(select=None)
    am = pipe.am(raw, select=None)
    pool = pipe.fuse(raw, am)
    name_to_index = {name: index for index, name in enumerate(pool.names)}
    retained = [name for name in expression_names if name in name_to_index]
    missing = [name for name in expression_names if name not in name_to_index]
    if retained:
        indices = [name_to_index[name] for name in retained]
        expression_model = logistic_model().fit(
            np.column_stack([base_train, pool.X[:, indices]]), y[development]
        )
        expression_test = expression_model.predict_proba(
            np.column_stack([base_test, pool._X_val[:, indices]])
        )[:, 1]
    else:
        expression_test = baseline_test.copy()
    return {
        "baseline_test": baseline_test,
        "ingredient_test": ingredient_test,
        "expression_test": expression_test,
        "retained_expressions": retained,
        "missing_expressions": missing,
    }


def metric_with_interval(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    rng: np.random.RandomState,
    repetitions: int,
) -> Dict[str, float]:
    result = probability_metrics(y, probability)
    low, high = grouped_bootstrap_brier(
        y, probability, groups, rng, repetitions
    )
    result.update({"brier_ci_low": low, "brier_ci_high": high})
    return result


def run_probe(
    games: int,
    seed: int,
    folds: int,
    select: int,
    consensus_folds: int,
    bootstrap_repetitions: int,
) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(games, seed)
    development_games, test_games = split_development_test(groups, seed + 1)
    development = rows_for_games(groups, development_games)
    test = rows_for_games(groups, test_games)

    fold_results = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_local, validation_local) in enumerate(
        splitter.split(X[development], y[development], groups[development]), 1
    ):
        print(f"discovering fold {fold}/{folds}...", flush=True)
        fold_results.append(
            fit_discovery_fold(
                X,
                y,
                context,
                groups,
                development[fit_local],
                development[validation_local],
                test,
                select,
            )
        )

    ensemble_probabilities = {
        "fold_baseline_ensemble": np.mean(
            [row["baseline_test"] for row in fold_results], axis=0
        ),
        "raw_selection_ensemble": np.mean(
            [row["raw_test"] for row in fold_results], axis=0
        ),
        "am_selection_ensemble": np.mean(
            [row["am_test"] for row in fold_results], axis=0
        ),
    }

    expression_counts = Counter(
        feature
        for row in fold_results
        for feature in set(row["am_features"])
    )
    ingredient_counts = Counter()
    for row in fold_results:
        fold_ingredients = {
            ingredient
            for expression in row["am_features"]
            for ingredient in ingredients_in_expression(expression)
        }
        ingredient_counts.update(fold_ingredients)

    consensus_ingredients = sorted(
        name for name, count in ingredient_counts.items() if count >= consensus_folds
    )
    consensus_expressions = sorted(
        name for name, count in expression_counts.items() if count >= consensus_folds
    )
    consensus = fit_consensus_models(
        X,
        y,
        context,
        groups,
        development,
        test,
        consensus_ingredients,
        consensus_expressions,
    )
    ensemble_probabilities.update(
        {
            "full_development_baseline": consensus["baseline_test"],
            "ingredient_consensus": consensus["ingredient_test"],
            "expression_consensus": consensus["expression_test"],
        }
    )

    rng = np.random.RandomState(seed + 2)
    metrics = {
        name: metric_with_interval(
            y[test], probability, groups[test], rng, bootstrap_repetitions
        )
        for name, probability in ensemble_probabilities.items()
    }
    deltas = {
        "am_ensemble_minus_raw_ensemble": grouped_bootstrap_delta(
            y[test],
            ensemble_probabilities["am_selection_ensemble"],
            ensemble_probabilities["raw_selection_ensemble"],
            groups[test],
            rng,
            bootstrap_repetitions,
        ),
        "ingredient_consensus_minus_full_baseline": grouped_bootstrap_delta(
            y[test],
            ensemble_probabilities["ingredient_consensus"],
            ensemble_probabilities["full_development_baseline"],
            groups[test],
            rng,
            bootstrap_repetitions,
        ),
        "expression_consensus_minus_full_baseline": grouped_bootstrap_delta(
            y[test],
            ensemble_probabilities["expression_consensus"],
            ensemble_probabilities["full_development_baseline"],
            groups[test],
            rng,
            bootstrap_repetitions,
        ),
    }

    fold_sets = [set(row["am_features"]) for row in fold_results]
    pairwise_jaccard = [
        len(fold_sets[i] & fold_sets[j]) / len(fold_sets[i] | fold_sets[j])
        for i in range(folds)
        for j in range(i + 1, folds)
    ]
    return {
        "generation": generation,
        "seed": seed,
        "folds": folds,
        "selection_size": select,
        "consensus_threshold_folds": consensus_folds,
        "development_games": int(len(development_games)),
        "test_games": int(len(test_games)),
        "test_snapshots": int(len(test)),
        "metrics": metrics,
        "deltas": deltas,
        "stability": {
            "ingredient_fold_counts": dict(
                sorted(ingredient_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "expression_fold_counts": dict(
                sorted(expression_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "consensus_ingredients": consensus_ingredients,
            "consensus_expressions": consensus_expressions,
            "retained_consensus_expressions": consensus["retained_expressions"],
            "missing_consensus_expressions": consensus["missing_expressions"],
            "mean_pairwise_expression_jaccard": float(np.mean(pairwise_jaccard)),
            "pairwise_expression_jaccard": pairwise_jaccard,
        },
        "fold_selections": [
            {
                "fold": index + 1,
                "raw_features": row["raw_features"],
                "am_features": row["am_features"],
            }
            for index, row in enumerate(fold_results)
        ],
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc

    metric_names = [
        "full_development_baseline",
        "raw_selection_ensemble",
        "am_selection_ensemble",
        "ingredient_consensus",
        "expression_consensus",
    ]
    labels = ["baseline", "raw ensemble", "AM ensemble", "ingredients", "expressions"]
    values = np.array([result["metrics"][name]["brier"] for name in metric_names])
    low = np.array([result["metrics"][name]["brier_ci_low"] for name in metric_names])
    high = np.array([result["metrics"][name]["brier_ci_high"] for name in metric_names])

    ingredient_counts = result["stability"]["ingredient_fold_counts"]
    ordered = sorted(ingredient_counts.items(), key=lambda item: (item[1], item[0]))
    names = [item[0] for item in ordered]
    counts = [item[1] for item in ordered]

    fig, (metric_ax, stability_ax) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(labels))
    metric_ax.bar(x, values, yerr=np.vstack([values - low, high - values]), capsize=3)
    metric_ax.set_xticks(x, labels, rotation=20, ha="right")
    metric_ax.set_ylabel("Untouched-test Brier score (lower is better)")
    metric_ax.set_title("Cross-fitted consensus performance")
    metric_ax.grid(axis="y", alpha=0.2)

    stability_ax.barh(names, counts)
    stability_ax.axvline(result["consensus_threshold_folds"] - 0.5, color="black", linestyle="--")
    stability_ax.set_xlim(0, result["folds"] + 0.3)
    stability_ax.set_xlabel("Discovery folds containing ingredient")
    stability_ax.set_title("Ingredient stability")
    stability_ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(result: Dict[str, object]) -> None:
    print(
        f"development={result['development_games']} games; "
        f"untouched test={result['test_games']} games"
    )
    for name, metrics in result["metrics"].items():
        print(f"{name:34s} {metrics['brier']:.5f}  AUC={metrics['roc_auc']:.4f}")
    print("consensus ingredients:", result["stability"]["consensus_ingredients"])
    print("consensus expressions:", result["stability"]["consensus_expressions"])
    print(
        "mean expression Jaccard:",
        f"{result['stability']['mean_pairwise_expression_jaccard']:.3f}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--select", type=int, default=16)
    parser.add_argument(
        "--consensus-folds",
        type=int,
        default=3,
        help="folds required for majority consensus (default: 3 of 5)",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(
        args.games,
        args.seed,
        args.folds,
        args.select,
        args.consensus_folds,
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
