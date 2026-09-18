# Cheeta co-adaptation guide

This repo reimplements the Half-Cheetah morphology/behaviour co-adaptation
task of Luck, Ben Amor & Calandra, *"Data-efficient Co-Adaptation of
Morphology and Behaviour with Deep RL"* (CoRL 2019), in Isaac Lab. A design is
a vector of six scale factors `xi in [0.8, 2.0]` on the leg segment lengths
(`bthigh, bshin, bfoot, fthigh, fshin, ffoot`, base lengths
0.29/0.30/0.188 m per leg). The goal is to find the design *and* the policy
that run the fastest.

The workflow mirrors the paper's two networks:

1. a **general policy** that observes the design vector and is trained across
   many random morphologies at once (the paper's population network),
2. **specialised policies** for promising designs, fine-tuned from the general
   one inside an evolutionary loop that uses the general policy to screen
   candidates cheaply.

## Setup

All commands run from the repo root in a terminal with the `env_isaaclab`
conda env active (Isaac Lab 2.3.2 + Isaac Sim installed, this package
installed with `pip install -e source/cheeta`).

Every Isaac Sim script accepts the standard `AppLauncher` flags: add
`--headless` to run without the viewer (much faster), drop it to watch. The
drivers (`evolve.py`, `train_general.py`, `benchmark_metrics.py`) run without
Isaac Sim and launch one worker process per job.

## Environment, reward and observation

The env (`Template-Cheeta-Direct-v0`) rewards forward speed, per the paper:

```
r = max(v_x * reward_speed_scale, 0) - ctrl_cost_weight * |a|^2
```

with defaults `reward_speed_scale = 0.1`, `ctrl_cost_weight = 0` (i.e. the
paper's design-independent reward `r = max(v_x/10, 0)`). Knobs in
`CheetaEnvCfg` ([cheeta_env_cfg.py](../source/cheeta/cheeta/tasks/direct/cheeta/cheeta_env_cfg.py)):

- `clip_speed_reward = False` allows negative reward for moving backwards
- `ctrl_cost_weight = 0.1` + `clip_speed_reward = False` gives a
  Gym-HalfCheetah-style reward
- `design_vector` sets one morphology for all envs (CLI override below)
- `design_set` puts several morphologies in one scene: env `i` gets design
  `i % len(design_set)` (physics replication is switched off automatically)
- `observe_design = True` appends the design vector, mapped to `[-1, 1]`, to
  the 17-D state, so the observation is 23-D and one policy can serve every
  design. Checkpoints trained before this change (17-D) no longer load.

**Score** of a design = mean episodic return of the trained policy over one
deterministic evaluation episode in every parallel env (~1000 steps, 16.7 s).
The forward distance in meters is reported alongside; with several designs in
a scene both are also reported per design.

## 1. Sanity checks

```
python scripts/run_random_cheeta.py --headless --num_envs 4 --steps 600
python scripts/check_multi_design.py --headless --num_envs 8 --designs 4
```

The second one spawns four random designs over eight envs and prints the
per-design distances and the observation shape.

## 2. Build cheetahs with different limb lengths

```
python scripts/make_cheeta.py --headless                                   # default design, print URDF/USD paths
python scripts/make_cheeta.py --design 1.6 1.6 1.2 1.0 1.0 1.0 --preview   # long back legs, watch it stand
python scripts/make_cheeta.py --random 5 --preview                         # 5 random designs side by side
```

Assets are cached per design in `%TEMP%\cheeta_assets\half_cheetah\<design>\`
(the URDF -> USD conversion is skipped when nothing changed).

## 3. Train the general policy

```
python scripts/coadapt/train_general.py --phases 4 --designs-per-phase 64 --num_envs 512 --iterations 300 --wandb
```

Each phase is one worker process with a fresh batch of random designs spread
over the envs (8 envs per design here), warm-started from the previous phase.
Building 64 assets takes a minute or two the first time. Outputs under
`logs\coadapt\general\<timestamp>\`:

- `general.pt` — the latest checkpoint (what the other scripts take as
  `--general-checkpoint`)
- `phase_XX\result.json` — mean score plus per-design scores of the phase
- `phase_XX\metrics.json` — the learning curves

Interrupted runs resume with `--run-dir logs\coadapt\general\<timestamp>`.

The general policy plays any design directly:

```
python scripts/coadapt/play_design.py --design 1.3 0.9 1.2 0.8 0.8 0.8 --checkpoint logs\coadapt\general\<run>\general.pt
```

## 4. Evolve designs (co-adaptation loop)

```
python scripts/coadapt/evolve.py --general-checkpoint logs\coadapt\general\<run>\general.pt ^
    --generations 10 --population 12 --elites 2 --immigrants 2 --train-top 3 ^
    --num_envs 256 --iterations 100 --wandb
```

Per generation:

1. **propose**: the top `--elites` trained designs survive, `--immigrants`
   fresh random designs are added, the rest are Gaussian mutations
   (`--sigma`, decaying by `--sigma-decay`) of elite designs.
2. **screen**: all candidates are scored zero-shot by the general policy in a
   single Isaac Sim session (all designs in one scene, no training, ~1 min).
   The critic's value at the initial state `V(s0)` is recorded too; it is the
   paper's fitness proxy and needs no rollout at all.
3. **specialise**: elites keep training their own policy; of the new
   candidates the `--train-top` best by `--select-by` (`zero_shot`, `value_s0`,
   `random`, `none` = all) are fine-tuned for `--iterations` PPO iterations,
   each in its own process, warm-started from the parent's policy
   (`--inherit parent`, default) or from the general policy
   (`--inherit general`).
4. the fine-tuned score is the fitness. Screened-only candidates keep their
   zero-shot score in the history and cannot become elites.

Every generation prints the Spearman rank correlation between the screening
metrics and the fine-tuned scores of the trained candidates: the cheapest
running check of whether the screening metric is worth anything.

Without `--general-checkpoint` every candidate is trained from its parent's
checkpoint (the original loop). `scripts/coadapt/evolve_ng.py` is the same
loop driven by a nevergrad optimizer (CMA-ES, PSO, ...) instead of the GA; it
also accepts `--general-checkpoint`.

Outputs, under `logs\coadapt\<timestamp>\`:

- `history.csv` — every candidate ever proposed: generation, role, parent,
  trained flag, score, zero-shot score, `V(s0)`, distance, learning-curve
  slope, the six design scales
- `best.json` — best design so far, with its checkpoint path
- `gen_XX\screen\result.json` — the generation's zero-shot screening
- `gen_XX\cand_YY\{result.json, metrics.json, agent.pt, skrl\}` — per
  fine-tuned candidate: result, learning curves, checkpoint, TensorBoard logs

**Resuming:** re-run with `--run-dir logs\coadapt\<timestamp>`; finished jobs
are detected by their `result.json` and skipped.

**Budget guidance:** `--iterations 100` at `--num_envs 256` is ~800k env
steps per fine-tuned candidate (2-3 min on the RTX 4090 including the 30-45 s
Isaac Sim start). Screening costs one session per generation regardless of
the population size, so large populations with a small `--train-top` are cheap.
For a final result, take the best design from `best.json` and give it a long
single-design run warm-started from its checkpoint:

```
python scripts/skrl/train.py --task Template-Cheeta-Direct-v0 --headless --num_envs 512 --wandb --checkpoint logs\coadapt\<run>\gen_XX\cand_YY\agent.pt "env.design_vector=[...]"
```

## 5. Which RL metrics predict a good design?

`benchmark_metrics.py` measures how well the cheap signals predict the score
after a long fine-tuning run:

```
# proof of concept, ~15 min
python scripts/coadapt/benchmark_metrics.py --general-checkpoint logs\coadapt\general\<run>\general.pt --designs 6 --iterations 60 --repeat 2 --wandb
# overnight
python scripts/coadapt/benchmark_metrics.py --general-checkpoint logs\coadapt\general\<run>\general.pt --designs 40 --iterations 400 --repeat 6 --wandb
```

For every random design the worker records, before and during fine-tuning:

| metric | what it is | cost |
|---|---|---|
| `zero_shot_score` | return of the general policy on the design | one episode |
| `value_s0` | the general critic's value at the initial state | one forward pass |
| `reward_step_at_10/25/50pct` | mean training reward per step at 10/25/50 % of the budget | free |
| `reward_step_slope`, `reward_step_gain` | learning-curve slope / total improvement | free |
| `value_loss_*`, `policy_loss_*`, `policy_std_*`, `learning_rate_*` | PPO internals (the KL-adaptive learning rate reflects the policy's KL per update) | free |

The final score is the ground truth. `correlations.csv` lists the Spearman and
Pearson correlation of every metric with it, `table.csv` has the raw rows,
and the `--repeat` designs trained with a second seed give the noise floor
(how well the final score predicts itself across seeds; no metric can beat
that). With `--wandb` the tables and a bar chart of the correlations are
logged to the run. `--analyze-only --run-dir <run>` recomputes the tables.

A metric that correlates well can be plugged into the evolution as
`--select-by` (zero-shot and `V(s0)` are wired up; add new ones in
`choose_training_set` in [evolve.py](../scripts/coadapt/evolve.py)).

## 6. Weights & Biases

Add `--wandb` to any driver or to `scripts/skrl/train.py`. Project defaults to
`cheeta` (`--wandb-project`), all runs of one driver share a group named after
the run directory, and job types tell them apart: `general-train`, `screen`,
`candidate`, `benchmark`, plus a `*-driver` run that carries the
per-generation / per-phase summaries and tables. Worker runs sync skrl's
TensorBoard scalars (`Info / episode_return`, `Info / distance_x_m`, losses,
learning rate) and finish with the `result/*`, `probe/*` and
`train_metrics/*` summaries. `--wandb-mode offline` works without network.

## 7. Watch a design

`evolve.py` prints the exact command when it finishes; in general:

```
python scripts/coadapt/play_design.py --design 1.2 0.9 1.0 1.4 1.0 0.8 --checkpoint logs\coadapt\<run>\gen_04\cand_02\agent.pt --real-time
```

Without `--checkpoint` the design runs passively (zero torques).

## Script reference

| Script | Purpose |
|---|---|
| `scripts/run_random_cheeta.py` | random-action rollout, distance report |
| `scripts/check_multi_design.py` | several designs in one scene, per-design distances |
| `scripts/make_cheeta.py` | build/preview designs with custom limb lengths |
| `scripts/skrl/train.py` / `play.py` | PPO training/playback for one design (`--wandb`) |
| `scripts/coadapt/train_design.py` | worker: train + score one design or a design set; `--iterations 0` = evaluate only |
| `scripts/coadapt/train_general.py` | train the design-conditioned general policy |
| `scripts/coadapt/evolve.py` | evolutionary co-adaptation driver (screening + fine-tuning) |
| `scripts/coadapt/evolve_ng.py` | nevergrad-driven variant |
| `scripts/coadapt/benchmark_metrics.py` | correlate cheap RL metrics with final design scores |
| `scripts/coadapt/play_design.py` | watch a design + checkpoint |
| `scripts/list_envs.py`, `zero_agent.py`, `random_agent.py` | template utilities |
