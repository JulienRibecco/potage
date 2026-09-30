#!/usr/bin/env python3
"""Probe whether Potage can recover planted win factors from game states.

Each row is one randomly timed snapshot from an independent match. The outcome
is sampled from a hidden win-probability function containing four mechanisms:

* relative gold (a log ratio),
* objective pressure combined with opponent health,
* counter units matched against the opponent's heavy units,
* map control whose value changes with game time.

Potage sees only the raw state. It selects features against the residual of an
obvious-state baseline, and a final logistic model tests whether those features
improve probabilities on untouched matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from potage import SoupPipe


PLANTED_FACTORS = {
    "relative_gold": (("our_gold", "opponent_gold"),),
    "objective_finish": (
        ("our_objective_pressure", "opponent_health"),
        ("opponent_objective_pressure", "our_health"),
    ),
    "counter_matchup": (
        ("our_counter_units", "opponent_heavy_units"),
        ("opponent_counter_units", "our_heavy_units"),
    ),
    "time_x_map": (
        ("time_minutes", "our_map_control"),
        ("time_minutes", "opponent_map_control"),
    ),
}


def zscore(x: np.ndarray) -> np.ndarray:
    scale = x.std()
    return (x - x.mean()) / (scale if scale > 1e-12 else 1.0)


def simulate_game_states(
    n_matches: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Generate independent snapshots and outcomes from a known hidden law."""
    rng = np.random.RandomState(seed)
    time = rng.uniform(5.0, 45.0, n_matches)
    skill = rng.normal(size=n_matches)

    score_pace = np.maximum(time / 9.0, 0.4)
    our_score = rng.poisson(score_pace * np.exp(0.16 * skill))
    opponent_score = rng.poisson(score_pace * np.exp(-0.16 * skill))

    gold_base = 2_000.0 + 430.0 * time
    gold_log_advantage = 0.06 * skill + rng.normal(scale=0.24, size=n_matches)
    our_gold = gold_base * np.exp(0.5 * gold_log_advantage)
    opponent_gold = gold_base * np.exp(-0.5 * gold_log_advantage)

    our_health = rng.uniform(0.05, 1.0, n_matches)
    opponent_health = rng.uniform(0.05, 1.0, n_matches)
    our_map_control = rng.uniform(0.05, 0.95, n_matches)
    opponent_map_control = rng.uniform(0.05, 0.95, n_matches)
    our_objective_pressure = rng.uniform(0.0, 1.0, n_matches)
    opponent_objective_pressure = rng.uniform(0.0, 1.0, n_matches)
    our_counter_units = rng.binomial(5, 0.5, n_matches)
    opponent_counter_units = rng.binomial(5, 0.5, n_matches)
    our_heavy_units = rng.binomial(5, 0.5, n_matches)
    opponent_heavy_units = rng.binomial(5, 0.5, n_matches)
    our_alive = rng.binomial(5, np.clip(0.58 + 0.08 * skill, 0.15, 0.95))
    opponent_alive = rng.binomial(
        5, np.clip(0.58 - 0.08 * skill, 0.15, 0.95)
    )

    noise = rng.normal(size=(n_matches, 6))
    names = [
        "time_minutes",
        "our_score",
        "opponent_score",
        "our_gold",
        "opponent_gold",
        "our_health",
        "opponent_health",
        "our_map_control",
        "opponent_map_control",
        "our_objective_pressure",
        "opponent_objective_pressure",
        "our_counter_units",
        "opponent_counter_units",
        "our_heavy_units",
        "opponent_heavy_units",
        "our_alive",
        "opponent_alive",
    ] + [f"noise_{i}" for i in range(noise.shape[1])]
    X = np.column_stack(
        [
            time,
            our_score,
            opponent_score,
            our_gold,
            opponent_gold,
            our_health,
            opponent_health,
            our_map_control,
            opponent_map_control,
            our_objective_pressure,
            opponent_objective_pressure,
            our_counter_units,
            opponent_counter_units,
            our_heavy_units,
            opponent_heavy_units,
            our_alive,
            opponent_alive,
            noise,
        ]
    )

    relative_gold = zscore(np.log(our_gold / opponent_gold))
    objective_finish = zscore(
        (our_objective_pressure - 0.5) * (0.525 - opponent_health)
        - (opponent_objective_pressure - 0.5) * (0.525 - our_health)
    )
    counter_matchup = zscore(
        our_counter_units * opponent_heavy_units
        - opponent_counter_units * our_heavy_units
    )
    time_x_map = zscore(
        ((time - 25.0) / 20.0)
        * (our_map_control - opponent_map_control)
    )
    obvious_score = zscore(our_score - opponent_score)
    obvious_alive = zscore(our_alive - opponent_alive)

    planted = {
        "relative_gold": relative_gold,
        "objective_finish": objective_finish,
        "counter_matchup": counter_matchup,
        "time_x_map": time_x_map,
    }
    oracle_X = np.column_stack(
        [obvious_score, obvious_alive] + [planted[k] for k in PLANTED_FACTORS]
    )
    hidden_logit = (
        0.42 * obvious_score
        + 0.34 * obvious_alive
        + 0.68 * relative_gold
        + 0.95 * objective_finish
        + 0.68 * counter_matchup
        + 0.95 * time_x_map
    )
    probability = expit(hidden_logit - hidden_logit.mean())
    y = rng.binomial(1, probability).astype(np.float64)
    return X, y, names, oracle_X, planted


def baseline_matrix(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    columns = {name: X[:, i] for i, name in enumerate(names)}
    return np.column_stack(
        [
            columns["time_minutes"],
            columns["our_score"] - columns["opponent_score"],
            columns["our_alive"] - columns["opponent_alive"],
        ]
    )


def logistic_model() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, max_iter=2_000)
    )


def cross_fitted_probabilities(
    X: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit, held_out in folds.split(X, y):
        model = logistic_model().fit(X[fit], y[fit])
        prediction[held_out] = model.predict_proba(X[held_out])[:, 1]
    return prediction


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[object, Dict[str, Dict[str, float]]]:
    model = logistic_model().fit(X_train, y_train)
    result = {
        "validation": probability_metrics(
            y_validation, model.predict_proba(X_validation)[:, 1]
        ),
        "test": probability_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    }
    return model, result


def factor_matches(feature_name: str) -> List[str]:
    return [
        factor
        for factor, alternatives in PLANTED_FACTORS.items()
        if any(
            all(token in feature_name for token in tokens)
            for tokens in alternatives
        )
    ]


def evaluate_selected_features(
    pipe: SoupPipe,
    fs,
    baseline_oof: np.ndarray,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
    y_discovery: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
) -> Dict[str, object]:
    discovery_design = np.column_stack(
        [logit(np.clip(baseline_oof, 1e-6, 1 - 1e-6)), fs.X]
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
    model, scores = fit_and_score(
        discovery_design,
        y_discovery,
        validation_design,
        y_validation,
        test_design,
        y_test,
    )
    coefficients = model[-1].coef_[0][1:]
    feature_rows = [
        {
            "name": name,
            "standardized_coefficient": float(coefficient),
            "matches_planted": factor_matches(name),
        }
        for name, coefficient in sorted(
            zip(fs.names, coefficients), key=lambda pair: -abs(pair[1])
        )
    ]
    recovered = sorted(
        {
            factor
            for row in feature_rows
            for factor in row["matches_planted"]
        }
    )
    scores.update(
        {
            "n_features": len(fs.names),
            "features": feature_rows,
            "recovered_planted_factors": recovered,
            "family_counts": dict(
                sorted(pipe.stages[-1].family_census.items())
            ),
        }
    )
    return scores


def run_probe(n_matches: int, seed: int, select: int) -> Dict[str, object]:
    X, y, names, oracle_X, _ = simulate_game_states(n_matches, seed)
    discovery_val, test = train_test_split(
        np.arange(n_matches), test_size=0.20, stratify=y, random_state=seed + 1
    )
    discovery, validation = train_test_split(
        discovery_val,
        test_size=0.25,
        stratify=y[discovery_val],
        random_state=seed + 2,
    )

    baseline_X = baseline_matrix(X, names)
    baseline_oof = cross_fitted_probabilities(
        baseline_X[discovery], y[discovery], seed + 3
    )
    baseline, baseline_scores = fit_and_score(
        baseline_X[discovery],
        y[discovery],
        baseline_X[validation],
        y[validation],
        baseline_X[test],
        y[test],
    )
    baseline_validation = baseline.predict_proba(baseline_X[validation])[:, 1]
    baseline_test = baseline.predict_proba(baseline_X[test])[:, 1]

    _, oracle_scores = fit_and_score(
        oracle_X[discovery],
        y[discovery],
        oracle_X[validation],
        y[validation],
        oracle_X[test],
        y[test],
    )

    excluded = {
        "squared",
        "log",
        "sqrt",
        "cube",
        "tanh",
        "reciprocal",
        "rank",
        "am_ratio",
        "am_power",
    }
    pipe = SoupPipe(
        X[discovery],
        y[discovery],
        names,
        feature_types=["numeric"] * len(names),
        X_val=X[validation],
        y_val=y[validation],
        y_base=baseline_oof,
        y_val_base=baseline_validation,
        task="regression",
        exclude_families=excluded,
        max_candidates=100_000,
    )
    raw = pipe.raw(select=None)
    raw_selected = pipe.select(raw, select=select, method="greedy")
    raw_scores = evaluate_selected_features(
        pipe,
        raw_selected,
        baseline_oof,
        baseline_validation,
        baseline_test,
        y[discovery],
        y[validation],
        y[test],
        X[validation],
        X[test],
    )

    am = pipe.am(raw, select=None)
    am_selected = pipe.select(pipe.fuse(raw, am), select=select, method="greedy")
    am_scores = evaluate_selected_features(
        pipe,
        am_selected,
        baseline_oof,
        baseline_validation,
        baseline_test,
        y[discovery],
        y[validation],
        y[test],
        X[validation],
        X[test],
    )

    return {
        "simulation": {
            "n_matches": n_matches,
            "seed": seed,
            "positive_rate": float(y.mean()),
            "features": names,
            "planted_factors": list(PLANTED_FACTORS),
        },
        "split": {
            "discovery": len(discovery),
            "validation": len(validation),
            "test": len(test),
        },
        "baseline": baseline_scores,
        "potage_raw": raw_scores,
        "potage_am": am_scores,
        "oracle": oracle_scores,
        "pipeline_recap": pipe.recap(),
    }


def print_summary(result: Dict[str, object]) -> None:
    print("\nModel                     Brier     Log loss   ROC-AUC")
    print("------------------------------------------------------")
    for name in ("baseline", "potage_raw", "potage_am", "oracle"):
        test = result[name]["test"]
        print(
            f"{name:24s}  {test['brier']:.4f}    "
            f"{test['log_loss']:.4f}     {test['roc_auc']:.4f}"
        )
    print("\nAM recovered:", result["potage_am"]["recovered_planted_factors"])
    print("Top AM features:")
    for row in result["potage_am"]["features"][:10]:
        marker = f" -> {row['matches_planted']}" if row["matches_planted"] else ""
        print(
            f"  {row['standardized_coefficient']:+.3f}  "
            f"{row['name']}{marker}"
        )


def save_plot(result: Dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--plot requires matplotlib") from exc

    labels = ["baseline", "Potage raw", "Potage AM", "oracle"]
    keys = ["baseline", "potage_raw", "potage_am", "oracle"]
    validation = [result[key]["validation"]["brier"] for key in keys]
    test = [result[key]["test"]["brier"] for key in keys]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, validation, width, label="validation")
    ax.bar(x + width / 2, test, width, label="untouched test")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Planted game-state win-factor recovery")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--select", type=int, default=16)
    parser.add_argument(
        "--stability-seeds",
        type=int,
        nargs="*",
        default=[],
        help="additional simulations to summarize (slower)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.matches, args.seed, args.select)
    print_summary(result)
    stability = []
    for seed in args.stability_seeds:
        repeat = run_probe(args.matches, seed, args.select)
        row = {
            "seed": seed,
            "baseline_test_brier": repeat["baseline"]["test"]["brier"],
            "potage_am_test_brier": repeat["potage_am"]["test"]["brier"],
            "oracle_test_brier": repeat["oracle"]["test"]["brier"],
            "recovered_planted_factors": repeat["potage_am"][
                "recovered_planted_factors"
            ],
        }
        stability.append(row)
        print(
            f"stability seed {seed}: AM Brier "
            f"{row['potage_am_test_brier']:.4f}; recovered "
            f"{row['recovered_planted_factors']}"
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
