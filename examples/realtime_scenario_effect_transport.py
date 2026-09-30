#!/usr/bin/env python3
"""Test whether Potage scenario-effect factors transport across interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

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
from examples.realtime_race_probe import FEATURE_NAMES, SCENARIO_FEATURE_NAMES
from examples.realtime_scenario_effect_probe import collect_dataset
from examples.realtime_stage_selection_probe import build_recipe


CONTEXT_NAMES = ["time_fraction", "score_difference", "skill_difference", "perspective"]


def logistic_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000))


def metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": (
            float(roc_auc_score(y, probability))
            if np.unique(y).size == 2
            else float("nan")
        ),
    }


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    for fit, held_out in GroupKFold(n_splits=5).split(X, y, groups):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def fit_transport_space(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_context: np.ndarray,
    train_groups: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    test_context: np.ndarray,
    names: Sequence[str],
    recipe: str,
    method: str,
    select: int,
    feature_space: str,
    seed: int,
) -> Dict[str, object]:
    discovery, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train_X, train_y, train_groups
        )
    )
    baseline_oof = grouped_oof(
        train_context[discovery], train_y[discovery], train_groups[discovery]
    )
    baseline = logistic_model().fit(train_context[discovery], train_y[discovery])
    baseline_validation = baseline.predict_proba(train_context[validation])[:, 1]
    baseline_test = baseline.predict_proba(test_context)[:, 1]
    excluded = {
        "squared", "log", "sqrt", "cube", "tanh", "reciprocal",
        "rank", "am_ratio", "am_logratio", "am_power",
    }
    pipe = SoupPipe(
        train_X[discovery], train_y[discovery], list(names),
        feature_types=["numeric"] * train_X.shape[1],
        X_val=train_X[validation], y_val=train_y[validation],
        y_base=baseline_oof, y_val_base=baseline_validation,
        task="regression", exclude_families=excluded,
        max_candidates=100_000,
    )
    final = build_recipe(pipe, recipe, method, select)
    design = lambda base, values: np.column_stack([
        logit(np.clip(base, 1e-6, 1.0 - 1e-6)), values,
    ])
    model = logistic_model().fit(design(baseline_oof, final.X), train_y[discovery])
    test_probability = model.predict_proba(
        design(baseline_test, pipe.transform(test_X))
    )[:, 1]
    return {
        "feature_space": feature_space,
        "test": metrics(test_y, test_probability),
        "selected_features": list(final.names),
        "selected_scenario_features": [
            name for name in final.names
            if any(marker in name for marker in (
                "our_speed", "opponent_speed", "speed_difference",
                "our_start_to_center_node", "opponent_start_to_center_node",
                "start_separation",
            ))
        ],
    }


def run_probe(
    pairs: int,
    seed: int,
    select: int,
    recipe: str,
    method: str,
    phases: Sequence[int],
) -> Dict[str, object]:
    all_results = {}
    regimes = {
        "speed": {"train": {"speed_delta": 1}, "test": {"speed_delta": 2}},
        "initial_position": {
            "train": {"position_steps": 1},
            "test": {"position_steps": 2},
        },
    }
    for kind, config in regimes.items():
        train_X, train_y, train_context, train_groups, train_generation = collect_dataset(
            pairs, seed, kind, phases, **config["train"]
        )
        test_X, test_y, test_context, test_groups, test_generation = collect_dataset(
            pairs, seed + 1, kind, phases, **config["test"]
        )
        spaces = {
            "context": (train_context, test_context, CONTEXT_NAMES),
            "core": (
                train_X[:, :len(FEATURE_NAMES)],
                test_X[:, :len(FEATURE_NAMES)],
                FEATURE_NAMES,
            ),
            "scenario": (train_X, test_X, SCENARIO_FEATURE_NAMES),
        }
        models = {}
        for space, (space_train, space_test, names) in spaces.items():
            models[space] = fit_transport_space(
                space_train, train_y, train_context, train_groups,
                space_test, test_y, test_context, names,
                recipe, method, select, space, seed + 101,
            )
            print(
                f"{kind:16s} {space:8s}: held-out Brier="
                f"{models[space]['test']['brier']:.4f}, "
                f"AUC={models[space]['test']['roc_auc']:.4f}"
            )
        all_results[kind] = {
            "train_generation": train_generation,
            "test_generation": test_generation,
            "train_positive_rate": float(train_y.mean()),
            "test_positive_rate": float(test_y.mean()),
            "train_groups": int(np.unique(train_groups).size),
            "test_groups": int(np.unique(test_groups).size),
            "models": models,
        }
    return {
        "pairs": pairs,
        "seed": seed,
        "train_regimes": {"speed_delta": 1, "position_steps": 1},
        "heldout_regimes": {"speed_delta": 2, "position_steps": 2},
        "recipe": recipe,
        "method": method,
        "phases": list(phases),
        "results": all_results,
    }


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    spaces = ["context", "core", "scenario"]
    x = np.arange(len(spaces))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, kind in zip((-width / 2, width / 2), ("speed", "initial_position")):
        values = [
            result["results"][kind]["models"][space]["test"]["brier"]
            for space in spaces
        ]
        ax.bar(x + offset, values, width, label=kind.replace("_", " "))
    ax.set_xticks(x, spaces)
    ax.set_ylabel("held-out Brier (lower is better)")
    ax.set_title("Transport across intervention magnitude")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--method", default="correlation")
    parser.add_argument("--phases", type=int, nargs="*", default=list(range(9)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.pairs, args.seed, args.select, args.recipe, args.method, args.phases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
    if args.plot:
        save_plot(result, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
