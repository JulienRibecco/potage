#!/usr/bin/env python3
"""Probe which real-time states benefit most from lower command latency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
from examples.realtime_race_probe import (
    FEATURE_NAMES,
    SNAPSHOT_TICKS,
    play_game,
)
from examples.realtime_stage_selection_probe import build_recipe


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


def policy_noise(seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {1: rng.gumbel(size=(100, 3)), -1: rng.gumbel(size=(100, 3))}


def game_seed(seed: int, game_id: int, focal: int) -> int:
    return int((seed * 1_000_003 + game_id * 17 + (focal == -1)) % (2**32 - 1))


def collect_dataset(
    pairs: int,
    seed: int,
    phases: Sequence[int],
    low_latency: int = 0,
    high_latency: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rows: List[np.ndarray] = []
    contexts: List[np.ndarray] = []
    labels: List[float] = []
    groups: List[int] = []
    draws = 0
    positive_switches = 0
    for game_id in range(pairs):
        for focal in (1, -1):
            current_seed = game_seed(seed, game_id, focal)
            noise = policy_noise((current_seed + 7919) % (2**32 - 1))
            low = play_game(
                game_id,
                np.random.RandomState(current_seed),
                latency_override={focal: low_latency},
                policy_noise=noise,
            )
            high = play_game(
                game_id,
                np.random.RandomState(current_seed),
                latency_override={focal: high_latency},
                policy_noise=noise,
            )
            low_X, low_y, _, _ = low
            high_X, high_y, high_context, _ = high
            if not low_y or not high_y:
                draws += 1
                continue
            # Both focal perspectives from one game remain in one group.
            group = game_id
            focal_offset = 0 if focal == 1 else 1
            for phase in phases:
                row_index = 2 * phase + focal_offset
                if row_index >= len(high_X) or row_index >= len(low_y):
                    continue
                # Reference features are the high-latency state; label is the
                # matched potential-outcome switch under latency reduction.
                rows.append(high_X[row_index])
                contexts.append(high_context[row_index])
                low_win = float(low_y[row_index])
                high_win = float(high_y[row_index])
                label = float(low_win > high_win)
                labels.append(label)
                groups.append(group)
                positive_switches += int(label)
    if not rows:
        raise RuntimeError("no non-draw intervention pairs were generated")
    return (
        np.vstack(rows),
        np.asarray(labels, dtype=np.float64),
        np.vstack(contexts),
        np.asarray(groups, dtype=np.int64),
        {
            "pairs_requested": pairs,
            "pairs_with_draw_or_mismatch": draws,
            "rows": len(rows),
            "positive_switches": positive_switches,
        },
    )


def run_probe(
    pairs: int,
    seed: int,
    select: int,
    recipe: str,
    method: str,
    phases: Sequence[int],
) -> Dict[str, object]:
    X, y, context, groups, generation = collect_dataset(pairs, seed, phases)
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
    baseline_scores = {
        "validation": metrics(y[validation], baseline_validation),
        "test": metrics(y[test], baseline_test),
    }

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
    validation_probability = model.predict_proba(validation_design)[:, 1]
    test_probability = model.predict_proba(test_design)[:, 1]
    potage_scores = {
        "validation": metrics(y[validation], validation_probability),
        "test": metrics(y[test], test_probability),
        "features": [
            {"name": name, "coefficient": float(coefficient)}
            for name, coefficient in sorted(
                zip(final.names, model[-1].coef_[0][1:]),
                key=lambda pair: -abs(pair[1]),
            )
        ],
    }
    phase_rate = {}
    for phase in phases:
        # phases are consecutive; use the explicit time feature instead.
        phase_value = SNAPSHOT_TICKS[phase] / 100.0
        mask = np.isclose(X[:, 0], phase_value)
        test_mask = np.isclose(X[test, 0], phase_value)
        phase_rate[str(SNAPSHOT_TICKS[phase])] = {
            "rows": int(mask.sum()),
            "positive_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            "test_predicted_benefit": (
                float(test_probability[test_mask].mean())
                if test_mask.any()
                else float("nan")
            ),
        }
    return {
        "pairs": pairs,
        "seed": seed,
        "recipe": recipe,
        "method": method,
        "phases": list(phases),
        "generation": generation,
        "positive_rate": float(y.mean()),
        "split": {
            "discovery_rows": len(discovery),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "discovery_games": int(np.unique(groups[discovery]).size),
            "validation_games": int(np.unique(groups[validation]).size),
            "test_games": int(np.unique(groups[test]).size),
        },
        "baseline": baseline_scores,
        "potage": potage_scores,
        "phase_positive_rate": phase_rate,
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"Generated: {result['generation']}; positive rate={result['positive_rate']:.4f}")
    print(f"Split: {result['split']}")
    for name in ("baseline", "potage"):
        test = result[name]["test"]
        print(
            f"{name:8s}: Brier={test['brier']:.4f}, "
            f"logloss={test['log_loss']:.4f}, AUC={test['roc_auc']:.4f}"
        )
    print("Top Potage effect features:")
    for feature in result["potage"]["features"][:10]:
        print(f"  {feature['coefficient']:+.3f}  {feature['name']}")


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc
    labels = ["baseline", "Potage"]
    values = [result["baseline"]["test"]["brier"], result["potage"]["test"]["brier"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].bar(labels, values)
    axes[0].set_ylabel("test Brier (lower is better)")
    axes[0].set_title("Effect prediction")
    phase_labels = list(result["phase_positive_rate"])
    phase_values = [
        result["phase_positive_rate"][key]["test_predicted_benefit"]
        for key in phase_labels
    ]
    axes[1].plot(phase_labels, phase_values, marker="o")
    axes[1].set_xlabel("reference snapshot tick")
    axes[1].set_ylabel("predicted latency-benefit probability")
    axes[1].set_title("Predicted benefit by reference phase")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--recipe", default="carrier_am")
    parser.add_argument("--method", default="correlation")
    parser.add_argument("--phases", type=int, nargs="*", default=list(range(9)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(
        args.pairs, args.seed, args.select, args.recipe, args.method, args.phases
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
