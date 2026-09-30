#!/usr/bin/env python3
"""Benchmark Potage on the UCI concrete compressive-strength dataset.

The primary protocol groups rows by their seven ingredient quantities before
splitting.  Measurements of the same recipe at different curing ages therefore
cannot appear in both discovery and test data.

Examples
--------
Run the full, leakage-resistant benchmark and save its outputs::

    python examples/concrete_strength.py \
        --protocol group \
        --output examples/concrete_strength_results.json \
        --plot examples/concrete_strength_results.png

Run a quicker smoke test through the AM stage::

    python examples/concrete_strength.py --max-stage 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Make ``python examples/concrete_strength.py`` work from a source checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from potage import SoupPipe


OPENML_DATA_ID = 44959
UCI_DOI = "https://doi.org/10.24432/C5PK67"
ALPHAS = np.logspace(-4, 4, 33)


def load_data(data_home: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the 1,030-row UCI dataset through its OpenML mirror."""
    try:
        bundle = fetch_openml(
            data_id=OPENML_DATA_ID,
            as_frame=False,
            data_home=data_home,
        )
    except Exception as exc:  # pragma: no cover - depends on network/cache
        raise RuntimeError(
            "Could not load the concrete dataset. Connect to the network once "
            f"or populate the scikit-learn cache at {data_home!r}."
        ) from exc

    X = np.asarray(bundle.data, dtype=np.float64)
    y = np.asarray(bundle.target, dtype=np.float64)
    return X, y, list(bundle.feature_names)


def recipe_groups(X: np.ndarray) -> np.ndarray:
    """Assign one group to each unique seven-ingredient formulation."""
    _, groups = np.unique(X[:, :7], axis=0, return_inverse=True)
    return groups


def split_indices(
    X: np.ndarray,
    y: np.ndarray,
    protocol: str,
    test_seed: int,
    validation_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 60/20/20 discovery/validation/test indices."""
    indices = np.arange(len(X))
    if protocol == "random":
        discovery_val, test = train_test_split(
            indices, test_size=0.20, random_state=test_seed
        )
        discovery, validation = train_test_split(
            discovery_val, test_size=0.25, random_state=validation_seed
        )
        return discovery, validation, test

    groups = recipe_groups(X)
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=test_seed
    )
    discovery_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=validation_seed
    )
    inner_discovery, inner_validation = next(
        inner.split(
            X[discovery_val], y[discovery_val], groups[discovery_val]
        )
    )
    return (
        discovery_val[inner_discovery],
        discovery_val[inner_validation],
        test,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_mpa": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mpa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_model(
    model,
    X_discovery: np.ndarray,
    y_discovery: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    started = perf_counter()
    model.fit(X_discovery, y_discovery)
    result: Dict[str, object] = {
        "validation": metrics(y_validation, model.predict(X_validation)),
        "test": metrics(y_test, model.predict(X_test)),
        "fit_seconds": perf_counter() - started,
    }
    ridge = getattr(model, "named_steps", {}).get("ridgecv")
    if ridge is not None:
        result["ridge_alpha"] = float(ridge.alpha_)
    return result


def baseline_models() -> Dict[str, object]:
    return {
        "mean": DummyRegressor(),
        "raw_ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS)
        ),
        "quadratic_ridge": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=ALPHAS),
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, l2_regularization=1.0, random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }


def evaluate_potage_stage(
    pipe: SoupPipe,
    fs,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_discovery: np.ndarray,
    started: float,
) -> Dict[str, object]:
    """Fit a regularized linear readout and score the current final stage."""
    X_validation_soup = pipe.transform(X_validation)
    X_test_soup = pipe.transform(X_test)
    readout = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    readout.fit(fs.X, y_discovery)
    return {
        "n_features": len(fs.names),
        "selected_features": list(fs.names),
        "family_counts": dict(sorted(pipe.stages[-1].family_census.items())),
        "ridge_alpha": float(readout[-1].alpha_),
        "validation": metrics(
            y_validation, readout.predict(X_validation_soup)
        ),
        "test": metrics(y_test, readout.predict(X_test_soup)),
        "cumulative_seconds": perf_counter() - started,
    }


def run_potage(
    X_discovery: np.ndarray,
    y_discovery: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    names: Sequence[str],
    max_stage: int,
    select: int,
) -> Tuple[Dict[str, Dict[str, object]], str]:
    """Build one progressive pipeline and score each selection checkpoint."""
    pipe = SoupPipe(
        X_discovery,
        y_discovery,
        names,
        feature_types=["numeric"] * len(names),
        X_val=X_validation,
        y_val=y_validation,
        max_candidates=100_000,
    )
    started = perf_counter()
    results: Dict[str, Dict[str, object]] = {}

    raw = pipe.raw(select=None)
    pool = raw
    selected = pipe.select(pool, select=select, method="greedy")
    results["stage_0_unary"] = evaluate_potage_stage(
        pipe,
        selected,
        X_validation,
        y_validation,
        X_test,
        y_test,
        y_discovery,
        started,
    )
    print_result("Potage stage 0", results["stage_0_unary"])

    if max_stage >= 1:
        am = pipe.am(raw, select=None)
        pool = pipe.fuse(pool, am)
        selected = pipe.select(pool, select=select, method="greedy")
        results["stage_1_am"] = evaluate_potage_stage(
            pipe,
            selected,
            X_validation,
            y_validation,
            X_test,
            y_test,
            y_discovery,
            started,
        )
        print_result("Potage stage 1", results["stage_1_am"])

    if max_stage >= 2:
        carriers = pipe.carriers(raw, select=None)
        pool = pipe.fuse(pool, carriers)
        selected = pipe.select(pool, select=select, method="greedy")
        results["stage_2_carriers"] = evaluate_potage_stage(
            pipe,
            selected,
            X_validation,
            y_validation,
            X_test,
            y_test,
            y_discovery,
            started,
        )
        print_result("Potage stage 2", results["stage_2_carriers"])

    if max_stage >= 3:
        fm = pipe.fm(raw, selected, select=None)
        pool = pipe.fuse(pool, fm)
        selected = pipe.select(pool, select=select, method="greedy")
        results["stage_3_fm_pm"] = evaluate_potage_stage(
            pipe,
            selected,
            X_validation,
            y_validation,
            X_test,
            y_test,
            y_discovery,
            started,
        )
        print_result("Potage stage 3", results["stage_3_fm_pm"])

    if max_stage >= 4:
        nonsmooth = pipe.fm_nonsmooth(raw, selected, select=None)
        pool = pipe.fuse(pool, nonsmooth)
        selected = pipe.select(pool, select=select, method="greedy")
        results["stage_4_fm_nonsmooth"] = evaluate_potage_stage(
            pipe,
            selected,
            X_validation,
            y_validation,
            X_test,
            y_test,
            y_discovery,
            started,
        )
        print_result("Potage stage 4", results["stage_4_fm_nonsmooth"])

    if max_stage >= 5:
        wm = pipe.wm(raw, selected, select=None)
        pool = pipe.fuse(pool, wm)
        selected = pipe.select(pool, select=select, method="greedy")
        results["stage_5_wm"] = evaluate_potage_stage(
            pipe,
            selected,
            X_validation,
            y_validation,
            X_test,
            y_test,
            y_discovery,
            started,
        )
        print_result("Potage stage 5", results["stage_5_wm"])

    return results, pipe.recap()


def print_result(name: str, result: Dict[str, object]) -> None:
    test = result["test"]
    print(
        f"{name:25s} test R2={test['r2']:7.4f}  "
        f"MAE={test['mae_mpa']:6.3f} MPa"
    )


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("--plot requires matplotlib") from exc

    labels: List[str] = []
    validation: List[float] = []
    test: List[float] = []
    for name, row in result["baselines"].items():
        labels.append(name.replace("_", " "))
        validation.append(row["validation"]["r2"])
        test.append(row["test"]["r2"])
    for name, row in result["potage"].items():
        labels.append(name.replace("_", " "))
        validation.append(row["validation"]["r2"])
        test.append(row["test"]["r2"])

    y_pos = np.arange(len(labels))
    width = 0.38
    fig_height = max(5.0, 0.48 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(y_pos - width / 2, validation, width, label="validation")
    ax.barh(y_pos + width / 2, test, width, label="held-out test")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("R2")
    ax.set_title(
        "Concrete strength: validation fit vs unseen-recipe generalization"
    )
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=("group", "random"), default="group"
    )
    parser.add_argument("--max-stage", type=int, choices=range(6), default=5)
    parser.add_argument("--select", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--data-home", default=".cache")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, names = load_data(args.data_home)
    discovery, validation, test = split_indices(
        X, y, args.protocol, args.test_seed, args.validation_seed
    )
    groups = recipe_groups(X)
    split_summary = {
        "discovery_rows": len(discovery),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "total_unique_recipes": int(np.unique(groups).size),
        "discovery_unique_recipes": int(np.unique(groups[discovery]).size),
        "validation_unique_recipes": int(np.unique(groups[validation]).size),
        "test_unique_recipes": int(np.unique(groups[test]).size),
    }
    print(f"Protocol: {args.protocol}; split: {split_summary}")

    baselines: Dict[str, Dict[str, object]] = {}
    for name, model in baseline_models().items():
        baselines[name] = evaluate_model(
            model,
            X[discovery],
            y[discovery],
            X[validation],
            y[validation],
            X[test],
            y[test],
        )
        print_result(name, baselines[name])

    potage, recap = run_potage(
        X[discovery],
        y[discovery],
        X[validation],
        y[validation],
        X[test],
        y[test],
        names,
        args.max_stage,
        args.select,
    )
    result: Dict[str, object] = {
        "dataset": {
            "name": "Concrete Compressive Strength",
            "uci_doi": UCI_DOI,
            "openml_data_id": OPENML_DATA_ID,
            "rows": len(X),
            "features": names,
            "target": "strength_mpa",
        },
        "protocol": args.protocol,
        "seeds": {
            "test": args.test_seed,
            "validation": args.validation_seed,
        },
        "split": split_summary,
        "baselines": baselines,
        "potage": potage,
        "potage_recap": recap,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
