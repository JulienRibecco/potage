#!/usr/bin/env python3
"""Ablate Potage stage arrangements and selectors on the real-time arena."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from potage import SoupPipe
from examples.realtime_race_probe import FEATURE_NAMES, generate_dataset


DEFAULT_RECIPES = ("raw", "raw_am", "raw_carriers", "deep_am", "carrier_am")
DEFAULT_METHODS = ("omp", "greedy", "correlation")


def logistic_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000))


def metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def build_recipe(pipe: SoupPipe, recipe: str, method: str, select: int):
    """Build one recipe using only public SoupPipe composition methods."""
    if recipe == "early_select":
        raw = pipe.raw(select=select, method=method)
        am = pipe.am(raw, select=select, method=method)
        return pipe.select(pipe.fuse(raw, am), select=select, method=method)

    raw = pipe.raw(select=None, method=method)
    if recipe == "raw":
        return pipe.select(raw, select=select, method=method)

    if recipe == "raw_am":
        am = pipe.am(raw, select=select * 2, method=method)
        return pipe.select(pipe.fuse(raw, am), select=select, method=method)

    if recipe == "raw_carriers":
        carriers = pipe.carriers(raw, select=select * 2, method=method)
        return pipe.select(pipe.fuse(raw, carriers), select=select, method=method)

    if recipe == "deep_am":
        am1 = pipe.am(raw, select=select * 2, method=method)
        am2 = pipe.am(am1, select=select * 2, method=method)
        return pipe.select(pipe.fuse(raw, am1, am2), select=select, method=method)

    if recipe == "carrier_am":
        carriers = pipe.carriers(raw, select=select * 2, method=method)
        am = pipe.am(carriers, select=select * 2, method=method)
        return pipe.select(pipe.fuse(raw, carriers, am), select=select, method=method)

    raise ValueError(f"Unknown recipe: {recipe}")


def run_configuration(
    recipe: str,
    method: str,
    select: int,
    X: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    discovery: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
) -> Dict[str, object]:
    excluded = {
        "squared", "log", "sqrt", "cube", "tanh", "reciprocal",
        "rank", "am_ratio", "am_logratio", "am_power",
    }
    pipe = SoupPipe(
        X[discovery], y[discovery], FEATURE_NAMES,
        feature_types=["numeric"] * X.shape[1],
        X_val=X[validation], y_val=y[validation],
        y_base=baseline_oof, y_val_base=baseline_validation,
        task="regression", exclude_families=excluded,
        max_candidates=100_000,
    )
    final = build_recipe(pipe, recipe, method, select)
    train_design = np.column_stack([
        logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), final.X,
    ])
    validation_design = np.column_stack([
        logit(np.clip(baseline_validation, 1e-6, 1 - 1e-6)),
        pipe.transform(X[validation]),
    ])
    test_design = np.column_stack([
        logit(np.clip(baseline_test, 1e-6, 1 - 1e-6)),
        pipe.transform(X[test]),
    ])
    model = logistic_model().fit(train_design, y[discovery])
    return {
        "recipe": recipe,
        "method": method,
        "selected_features": list(final.names),
        "validation": metrics(y[validation], model.predict_proba(validation_design)[:, 1]),
        "test": metrics(y[test], model.predict_proba(test_design)[:, 1]),
        "pipeline_recap": pipe.recap(),
    }


def run_probe(
    n_games: int,
    seed: int,
    select: int,
    recipes: Iterable[str] = DEFAULT_RECIPES,
    methods: Iterable[str] = DEFAULT_METHODS,
) -> Dict[str, object]:
    X, y, context, groups, generation = generate_dataset(n_games, seed)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    discovery_local, validation_local = next(
        inner.split(X[discovery_val], y[discovery_val], groups[discovery_val])
    )
    discovery = discovery_val[discovery_local]
    validation = discovery_val[validation_local]
    baseline_oof = grouped_oof(context[discovery], y[discovery], groups[discovery])
    baseline = logistic_model().fit(context[discovery], y[discovery])
    baseline_validation = baseline.predict_proba(context[validation])[:, 1]
    baseline_test = baseline.predict_proba(context[test])[:, 1]
    results: List[Dict[str, object]] = []
    for recipe in recipes:
        for method in methods:
            print(f"running {recipe:14s} / {method:11s}")
            result = run_configuration(
                recipe, method, select, X, y, context, discovery, validation, test,
                baseline_oof, baseline_validation, baseline_test,
            )
            results.append(result)
            print(f"  test Brier={result['test']['brier']:.4f}")
    return {
        "generation": generation,
        "seed": seed,
        "select": select,
        "recipes": list(recipes),
        "methods": list(methods),
        "rows": len(X),
        "split": {
            "discovery_rows": len(discovery),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "discovery_games": int(np.unique(groups[discovery]).size),
            "validation_games": int(np.unique(groups[validation]).size),
            "test_games": int(np.unique(groups[test]).size),
        },
        "baseline": {
            "validation": metrics(y[validation], baseline_validation),
            "test": metrics(y[test], baseline_test),
        },
        "results": results,
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; rows={result['rows']}")
    print(f"Split: {result['split']}")
    print("\nRecipe          Selector       Validation   Test Brier")
    print("------------------------------------------------------")
    for row in result["results"]:
        print(
            f"{row['recipe']:16s} {row['method']:12s} "
            f"{row['validation']['brier']:.4f}       {row['test']['brier']:.4f}"
        )


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    recipes = result["recipes"]
    methods = result["methods"]
    matrix = np.full((len(recipes), len(methods)), np.nan)
    for row in result["results"]:
        matrix[recipes.index(row["recipe"]), methods.index(row["method"])] = row["test"]["brier"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(methods)), methods)
    ax.set_yticks(np.arange(len(recipes)), recipes)
    ax.set_xlabel("selection strategy")
    ax.set_ylabel("stage arrangement")
    ax.set_title("Potage stage × selector ablation (test Brier)")
    for i in range(len(recipes)):
        for j in range(len(methods)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Brier score (lower is better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipes", nargs="*", choices=DEFAULT_RECIPES + ("early_select",), default=list(DEFAULT_RECIPES))
    parser.add_argument("--methods", nargs="*", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--stability-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--stability-games", type=int, default=400)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.games, args.seed, args.select, args.recipes, args.methods)
    print_summary(result)
    stability = []
    for seed in args.stability_seeds:
        repeat = run_probe(
            args.stability_games, seed, args.select, args.recipes, args.methods
        )
        best = sorted(
            (
                row["test"]["brier"],
                row["recipe"],
                row["method"],
            )
            for row in repeat["results"]
        )[:5]
        row = {
            "seed": seed,
            "games": args.stability_games,
            "baseline_test_brier": repeat["baseline"]["test"]["brier"],
            "best": [
                {"test_brier": score, "recipe": recipe, "method": method}
                for score, recipe, method in best
            ],
        }
        stability.append(row)
        print(
            f"stability seed {seed}: baseline={row['baseline_test_brier']:.4f}, "
            f"best={best[0][0]:.4f} ({best[0][1]}/{best[0][2]})"
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
