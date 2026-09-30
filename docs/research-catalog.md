> Historical research catalog. Saved scores and figures predate the transform
> replay and selection-scoring fixes unless a result explicitly records a rerun.
> Use the repository README and corrected concrete benchmark as the current
> entry points. Commands below run from the repository root. In particular,
> older conclusions about feature instability may mix statistical overfitting
> with implementation defects; they need new experiments before reuse.

The [AM-only replay audit](../research/am-replay-audit/README.md) revisits
concrete strength and metallic-glass descriptors across three grouped splits.
It isolates the AM defect from the other library fixes; it does not revalidate
the historical full neural cascades or game/intervention results.

# potage — Domain-Agnostic Feature Engineering

Composable feature engineering library for tabular/scientific data.
Build diverse feature libraries from raw columns using signal-inspired
transformations (products, sinusoidal carriers, FM/PM modulation,
Gaussian windows), then select the most predictive subset.

Pure numpy/scipy/sklearn. Zero external dependencies.

## Install

```bash
pip install numpy scipy scikit-learn
# Then add potage/ to your Python path or copy it into your project
```

## Quick start

```python
from potage import SoupPipe, SoupConfig

pipe = SoupPipe(X, y, feature_names)
fs_raw = pipe.raw(select=15)              # stage 0: identity, sq, log
fs_car = pipe.carriers(fs_raw, select=15) # stage 2: sin/cos carriers
fs_all = pipe.fuse(fs_raw, fs_car)
fs_final = pipe.select(fs_all, select=20)

X_new = pipe.transform(X_test)            # replay on new data
```

## Worked benchmark

`examples/concrete_strength.py` evaluates Potage on the UCI Concrete
Compressive Strength dataset. Its default protocol keeps identical ingredient
recipes in a single split, so measurements of one recipe at different curing
ages cannot leak into both discovery and test data.

```bash
python examples/concrete_strength.py \
    --output examples/concrete_strength_results.json \
    --plot examples/concrete_strength_results.png
```

The script compares raw and quadratic ridge models, three tree ensembles, and
each Potage stage from unary transforms through window modulation. Use
`--max-stage 1` for a quick smoke test.

`examples/game_state_probe.py` is a falsifiable mechanism-recovery experiment.
It generates game states with four planted win factors, fits an obvious-state
baseline, and asks Potage to recover the remaining interactions without seeing
the hidden formula.

```bash
python examples/game_state_probe.py \
    --output examples/game_state_probe_results.json \
    --plot examples/game_state_probe_results.png \
    --stability-seeds 19 41 97
```

`examples/reversi_probe.py` advances the same test to an emergent game. It
implements Reversi locally, generates stochastic self-play between players of
varying strength, groups multiple snapshots by complete game, and searches for
positional factors beyond phase, disc lead, color, and player strength.
It then composes the selected factors a second time, making the proposed
depth-versus-overfitting comparison explicit on untouched games.

```bash
python examples/reversi_probe.py \
    --output examples/reversi_probe_results.json \
    --plot examples/reversi_probe_results.png \
    --stability-seeds 19 41 97
```

Use `--games 6000` for the larger-data depth check.

`examples/reversi_learning_curve.py` performs the stricter test: all training
sizes are nested and share one fixed validation/test split, with uncertainty
bootstrapped by complete game.

```bash
python examples/reversi_learning_curve.py \
    --output examples/reversi_learning_curve_results.json \
    --plot examples/reversi_learning_curve_results.png
```

`examples/reversi_crossfit_probe.py` then repeats feature discovery across game
folds, measures exact-expression and ingredient consensus, and ensembles the
independently selected predictors on one untouched test set.

```bash
python examples/reversi_crossfit_probe.py \
    --output examples/reversi_crossfit_results.json \
    --plot examples/reversi_crossfit_results.png
```

Finally, `examples/reversi_intervention.py` tests whether acting on the stable
mobility/safety vocabulary changes paired win rate, and
`examples/reversi_intervention_phases.py` repeats the selected intervention
across game phases.

```bash
python examples/reversi_intervention.py \
    --output examples/reversi_intervention_results.json \
    --plot examples/reversi_intervention_results.png

python examples/reversi_intervention_phases.py \
    --output examples/reversi_intervention_phases_results.json \
    --plot examples/reversi_intervention_phases_results.png
```

Connect Four is the next game probe. It changes the vocabulary from positional
pressure to tactical threats, immediate wins, and open threes:

```bash
python examples/connect_four_probe.py \
    --output examples/connect_four_probe_results.json \
    --plot examples/connect_four_probe_results.png \
    --stability-seeds 19 41 97
```

For simultaneous-action timing, `examples/realtime_race_probe.py` uses a small
arena with delayed target commands, node capture, and score snapshots. The
baseline sees only coarse time/score context; Potage searches for compositional
tempo and control factors without using the hidden latency directly:

```bash
python examples/realtime_race_probe.py \
    --output examples/realtime_race_probe_results.json \
    --plot examples/realtime_race_probe_results.png \
    --stability-seeds 19 41 97
```

`examples/realtime_stage_selection_probe.py` keeps that arena split fixed and
ablates stage arrangements (`raw`, AM, carriers, and deeper compositions) ×
selection strategy (`omp`, `greedy`, `correlation`):

```bash
python examples/realtime_stage_selection_probe.py \
    --games 600 \
    --output examples/realtime_stage_selection_results.json \
    --plot examples/realtime_stage_selection_results.png
```

Finally, `examples/realtime_latency_intervention.py` performs a matched causal
test: it replays the same game with one agent's command latency set to 0 versus
3 ticks while holding policy noise and all other game variables fixed.

```bash
python examples/realtime_latency_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output examples/realtime_latency_intervention_results.json \
    --plot examples/realtime_latency_intervention_results.png
```

`examples/realtime_heterogeneous_effect_probe.py` then predicts which matched
states benefit from the intervention, using the low-versus-high latency outcome
switch as its target:

```bash
python examples/realtime_heterogeneous_effect_probe.py \
    --pairs 1000 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_heterogeneous_effect_results.json \
    --plot examples/realtime_heterogeneous_effect_results.png
```

`examples/realtime_phase_branch_probe.py` performs the stricter phase-causal
test: it shares a game prefix, branches at a chosen tick, and changes latency
only after that branch.

```bash
python examples/realtime_phase_branch_probe.py \
    --pairs 500 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_phase_branch_results.json \
    --plot examples/realtime_phase_branch_results.png
```

`examples/realtime_scenario_probe.py` varies initial geometry and agent speed,
then compares Potage with and without those explicit scenario descriptors. This
tests whether characteristics such as speed enter as robust factors or only as
conditional interactions.

```bash
python examples/realtime_scenario_probe.py \
    --games 1500 \
    --recipes raw carrier_am \
    --output examples/realtime_scenario_results.json \
    --plot examples/realtime_scenario_results.png
```

`examples/realtime_scenario_intervention.py` follows with matched causal tests
of speed and initial position, holding policy noise and opponent parameters
fixed:

```bash
python examples/realtime_scenario_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output examples/realtime_scenario_intervention_results.json \
    --plot examples/realtime_scenario_intervention_results.png
```

`examples/realtime_scenario_crossfit_probe.py` holds out each initial geometry
in turn to test whether selected speed/start factors transport to unseen
scenarios.

```bash
python examples/realtime_scenario_crossfit_probe.py \
    --games 800 \
    --recipe carrier_am \
    --output examples/realtime_scenario_crossfit_results.json \
    --plot examples/realtime_scenario_crossfit_results.png
```

`examples/realtime_continuous_scenario_probe.py` samples start positions across
the playable arena and trains on a continuous mixture of speed pairs. It then
tests both in-support speeds and a held-out speed-3 regime, checking whether
broader coverage closes the scenario transport gap.

```bash
python examples/realtime_continuous_scenario_probe.py \
    --games 1200 \
    --recipe carrier_am \
    --select 12 \
    --output examples/realtime_continuous_scenario_results.json \
    --plot examples/realtime_continuous_scenario_results.png
```

`examples/realtime_scenario_effect_probe.py` changes the target from final
outcome prediction to heterogeneous treatment effect: it asks which matched
states benefit when one agent is sped up or starts one step closer to the
center. The interventions share policy noise and game seeds, while start
positions and base speeds are randomized across the arena.

```bash
python examples/realtime_scenario_effect_probe.py \
    --pairs 1000 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_results.json \
    --plot examples/realtime_scenario_effect_results.png
```

`examples/realtime_scenario_effect_transport.py` trains on a +1-speed or
one-step intervention and evaluates on a held-out +2-speed or two-step
intervention, with a fresh randomized geometry sample.

```bash
python examples/realtime_scenario_effect_transport.py \
    --pairs 800 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_transport_results.json \
    --plot examples/realtime_scenario_effect_transport_results.png
```

`examples/realtime_scenario_effect_stability.py` repeats the paired-effect
probe across seeds and reports held-out metric variance plus exact selected
feature consensus.

```bash
python examples/realtime_scenario_effect_stability.py \
    --pairs 600 \
    --seeds 19 41 97 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_stability_results.json \
    --plot examples/realtime_scenario_effect_stability_results.png
```

## Stages

| Method | What it does |
|--------|-------------|
| `raw()` | identity, sq, log, one-hot |
| `am(fs)` | pairwise products, ratios |
| `carriers(fs)` | sin/cos/tri carriers |
| `fm(fs_carrier, fs_mod)` | frequency/phase modulation |
| `wm(fs)` | Gaussian window modulation |

## Feature types

Columns are typed as `numeric`, `bool`, or `categorical`:
- Numeric: all unary/binary ops
- Bool: gating (products) only
- Categorical: one-hot expand in stage 0, then gates

## Dependencies

- numpy
- scipy
- scikit-learn

Optional: `signalfault.classify._nn` for `neuron_cluster_features()` only.

## License

CC BY 4.0
