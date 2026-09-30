# Potage research catalog

This catalog maps research questions to scripts and saved reports. For library
installation and usage, start with the [README](../README.md); API behavior is
documented in the [reference](../REFERENCE.md).

## Validation status

| Record | Status | Entry point |
|---|---|---|
| Bike sharing | Current chronological comparison; eight Potage recipes and manual/tree baselines | [Report](../examples/bike_sharing_results.md) |
| Concrete strength | Rerun after replay/scoring fixes and module separation; one previously inspected split | [Report](../examples/concrete_strength_results.md) |
| AM replay audit | Paired diagnostic on concrete and metallic glass; three grouped splits, batch checks, negative controls | [Audit](../research/am-replay-audit/README.md) |
| Game, intervention, and transport experiments below | Historical results; not rerun after the library fixes | Scripts and adjacent result reports under `examples/` |

Historical scores and figures are exploratory records, not current release
benchmarks. Older interpretations of unstable features may mix statistical
overfitting with implementation defects. Their simulator tests check code
behavior; passing those tests does not revalidate the saved research findings.
The AM audit does not revalidate the full metallic-glass neural cascade or the
game/intervention experiments. Its [data notes](../research/am-replay-audit/DATA.md)
explain which inputs are available for independent reproduction.

## Running the experiments

Run commands from the repository root after completing the README setup:

```bash
python -m pip install -e '.[examples]'
```

Commands below write to ignored `artifacts/` paths so fresh runs do not overwrite
the historical records. Most synthetic game experiments need no dataset
download, but the larger settings and seed sweeps can take substantial time.

## Bike-sharing demand

The [bike-sharing report](../examples/bike_sharing_results.md) compares current
Potage recipes on fixed chronological training/validation/test windows. It
includes the newly corrected field replay, checks batch agreement, and keeps
manual calendar features and boosting as explicit baselines.

```bash
python examples/bike_sharing.py --recipes carriers --output artifacts/bike_carriers.json
```

This scores validation only. The report explains when to use `--evaluate-test`;
the saved held-out quarter should not become a target for further tuning.

## Concrete-strength benchmark

`examples/concrete_strength.py` evaluates Potage on the UCI Concrete
Compressive Strength dataset. Its default protocol keeps identical ingredient
recipes in a single split, so measurements of one recipe at different curing
ages cannot leak into both discovery and test data.

```bash
python examples/concrete_strength.py \
    --output artifacts/concrete_strength_results.json \
    --plot artifacts/concrete_strength_results.png
```

The script compares raw and quadratic ridge models, three tree ensembles, and
each Potage stage from unary transforms through window modulation. Use
`--max-stage 1` for a quick smoke test.

## Historical game and mechanism probes

`examples/game_state_probe.py` is a falsifiable mechanism-recovery experiment.
It generates game states with four planted win factors, fits an obvious-state
baseline, and asks Potage to recover the remaining interactions without seeing
the hidden formula.

```bash
python examples/game_state_probe.py \
    --output artifacts/game_state_probe_results.json \
    --plot artifacts/game_state_probe_results.png \
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
    --output artifacts/reversi_probe_results.json \
    --plot artifacts/reversi_probe_results.png \
    --stability-seeds 19 41 97
```

Use `--games 6000` for the larger-data depth check.

`examples/reversi_learning_curve.py` performs the stricter test: all training
sizes are nested and share one fixed validation/test split, with uncertainty
bootstrapped by complete game.

```bash
python examples/reversi_learning_curve.py \
    --output artifacts/reversi_learning_curve_results.json \
    --plot artifacts/reversi_learning_curve_results.png
```

`examples/reversi_crossfit_probe.py` then repeats feature discovery across game
folds, measures exact-expression and ingredient consensus, and ensembles the
independently selected predictors on one untouched test set.

```bash
python examples/reversi_crossfit_probe.py \
    --output artifacts/reversi_crossfit_results.json \
    --plot artifacts/reversi_crossfit_results.png
```

Finally, `examples/reversi_intervention.py` tests whether acting on the stable
mobility/safety vocabulary changes paired win rate, and
`examples/reversi_intervention_phases.py` repeats the selected intervention
across game phases.

```bash
python examples/reversi_intervention.py \
    --output artifacts/reversi_intervention_results.json \
    --plot artifacts/reversi_intervention_results.png

python examples/reversi_intervention_phases.py \
    --output artifacts/reversi_intervention_phases_results.json \
    --plot artifacts/reversi_intervention_phases_results.png
```

Connect Four is the next game probe. It changes the vocabulary from positional
pressure to tactical threats, immediate wins, and open threes:

```bash
python examples/connect_four_probe.py \
    --output artifacts/connect_four_probe_results.json \
    --plot artifacts/connect_four_probe_results.png \
    --stability-seeds 19 41 97
```

## Historical timing and intervention probes

For simultaneous-action timing, `examples/realtime_race_probe.py` uses a small
arena with delayed target commands, node capture, and score snapshots. The
baseline sees only coarse time/score context; Potage searches for compositional
tempo and control factors without using the hidden latency directly:

```bash
python examples/realtime_race_probe.py \
    --output artifacts/realtime_race_probe_results.json \
    --plot artifacts/realtime_race_probe_results.png \
    --stability-seeds 19 41 97
```

`examples/realtime_stage_selection_probe.py` keeps that arena split fixed and
ablates stage arrangements (`raw`, AM, carriers, and deeper compositions) ×
selection strategy (`omp`, `greedy`, `correlation`):

```bash
python examples/realtime_stage_selection_probe.py \
    --games 600 \
    --output artifacts/realtime_stage_selection_results.json \
    --plot artifacts/realtime_stage_selection_results.png
```

Finally, `examples/realtime_latency_intervention.py` performs a matched causal
test: it replays the same game with one agent's command latency set to 0 versus
3 ticks while holding policy noise and all other game variables fixed.

```bash
python examples/realtime_latency_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output artifacts/realtime_latency_intervention_results.json \
    --plot artifacts/realtime_latency_intervention_results.png
```

`examples/realtime_heterogeneous_effect_probe.py` then predicts which matched
states benefit from the intervention, using the low-versus-high latency outcome
switch as its target:

```bash
python examples/realtime_heterogeneous_effect_probe.py \
    --pairs 1000 \
    --recipe carrier_am \
    --method correlation \
    --output artifacts/realtime_heterogeneous_effect_results.json \
    --plot artifacts/realtime_heterogeneous_effect_results.png
```

`examples/realtime_phase_branch_probe.py` performs the stricter phase-causal
test: it shares a game prefix, branches at a chosen tick, and changes latency
only after that branch.

```bash
python examples/realtime_phase_branch_probe.py \
    --pairs 500 \
    --recipe carrier_am \
    --method correlation \
    --output artifacts/realtime_phase_branch_results.json \
    --plot artifacts/realtime_phase_branch_results.png
```

## Historical scenario and transport probes

`examples/realtime_scenario_probe.py` varies initial geometry and agent speed,
then compares Potage with and without those explicit scenario descriptors. This
tests whether characteristics such as speed enter as robust factors or only as
conditional interactions.

```bash
python examples/realtime_scenario_probe.py \
    --games 1500 \
    --recipes raw carrier_am \
    --output artifacts/realtime_scenario_results.json \
    --plot artifacts/realtime_scenario_results.png
```

`examples/realtime_scenario_intervention.py` follows with matched causal tests
of speed and initial position, holding policy noise and opponent parameters
fixed:

```bash
python examples/realtime_scenario_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output artifacts/realtime_scenario_intervention_results.json \
    --plot artifacts/realtime_scenario_intervention_results.png
```

`examples/realtime_scenario_crossfit_probe.py` holds out each initial geometry
in turn to test whether selected speed/start factors transport to unseen
scenarios.

```bash
python examples/realtime_scenario_crossfit_probe.py \
    --games 800 \
    --recipe carrier_am \
    --output artifacts/realtime_scenario_crossfit_results.json \
    --plot artifacts/realtime_scenario_crossfit_results.png
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
    --output artifacts/realtime_continuous_scenario_results.json \
    --plot artifacts/realtime_continuous_scenario_results.png
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
    --output artifacts/realtime_scenario_effect_results.json \
    --plot artifacts/realtime_scenario_effect_results.png
```

`examples/realtime_scenario_effect_transport.py` trains on a +1-speed or
one-step intervention and evaluates on a held-out +2-speed or two-step
intervention, with a fresh randomized geometry sample.

```bash
python examples/realtime_scenario_effect_transport.py \
    --pairs 800 \
    --recipe carrier_am \
    --method correlation \
    --output artifacts/realtime_scenario_effect_transport_results.json \
    --plot artifacts/realtime_scenario_effect_transport_results.png
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
    --output artifacts/realtime_scenario_effect_stability_results.json \
    --plot artifacts/realtime_scenario_effect_stability_results.png
```
