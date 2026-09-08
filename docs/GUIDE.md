# Cheeta co-adaptation guide

This repo reimplements the Half-Cheetah morphology/behaviour co-adaptation
task of Luck, Ben Amor & Calandra, *"Data-efficient Co-Adaptation of
Morphology and Behaviour with Deep RL"* (CoRL 2019), in Isaac Lab. A design is
a vector of six scale factors `xi in [0.8, 2.0]` on the leg segment lengths
(`bthigh, bshin, bfoot, fthigh, fshin, ffoot`, base lengths
0.29/0.30/0.188 m per leg). The goal is to find the design *and* the policy
that run the fastest.

## Setup

All commands run from the repo root in a terminal with the `env_isaaclab`
conda env active (Isaac Lab 2.3.2 + Isaac Sim installed, this package
installed with `pip install -e source/cheeta`).

Every script accepts the standard Isaac Lab `AppLauncher` flags: add
`--headless` to run without the viewer (much faster), drop it to watch.

## Reward and scoring

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
- `design_vector` sets the morphology; overridable from the CLI (see below)

**Score** of a design = mean episodic return of the trained policy over one
deterministic evaluation episode in every parallel env (~1000 steps, 16.7 s).
The forward distance in meters is reported alongside. During training both
appear in TensorBoard as `Info / episode_return` and `Info / distance_x_m`.

## 1. Sanity check: random actions

```
python scripts/run_random_cheeta.py --headless --num_envs 4 --steps 600
```

## 2. Build cheetahs with different limb lengths

```
# default design, print URDF/USD paths
python scripts/make_cheeta.py --headless

# long back legs, watch it stand
python scripts/make_cheeta.py --design 1.6 1.6 1.2 1.0 1.0 1.0 --preview

# segment lengths in meters instead of scale factors
python scripts/make_cheeta.py --lengths 0.35 0.40 0.20 0.29 0.30 0.188 --headless

# compare 5 random designs side by side in the viewer
python scripts/make_cheeta.py --random 5 --preview
```

`--design` / `--lengths` can be repeated to build several designs at once.
Assets land in `%TEMP%\cheeta_assets\half_cheetah\<design>\` unless `--out`
is given.

## 3. Train a single design with skrl (PPO)

```
python scripts/skrl/train.py --task Template-Cheeta-Direct-v0 --headless --num_envs 512
```

Logs and checkpoints go to `logs\skrl\cheeta_direct\<timestamp>_PPO_torch\`;
inspect with `tensorboard --logdir logs\skrl\cheeta_direct`. Train a specific
morphology via the hydra override (note the quoting):

```
python scripts/skrl/train.py --task Template-Cheeta-Direct-v0 --headless --num_envs 512 "env.design_vector=[1.5,1.0,1.0,1.2,1.0,1.0]"
```

Watch a trained checkpoint:

```
python scripts/skrl/play.py --task Template-Cheeta-Direct-v0 --num_envs 4 "env.design_vector=[...same as training...]"
```

The budget knob is `--max_iterations` (PPO rollout cycles of 32 steps x
num_envs samples each; the yaml default is 500 cycles).

## 4. Evolve designs (co-adaptation loop)

```
python scripts/coadapt/evolve.py --generations 10 --population 8 --num_envs 256 --iterations 100
```

Per generation each candidate design is trained with PPO for `--iterations`
rollout cycles and scored (one fresh Isaac Sim session per candidate;
~30-45 s startup each, so a generation of 8 with a small training budget
takes roughly 10 minutes on the RTX 4090). Then:

- the top `--elites` (default 2) designs survive unchanged and keep training
  their own policy next generation,
- children are Gaussian mutations (`--sigma`, default 0.15, decaying by
  `--sigma-decay` per generation) of elite designs and **warm-start their
  policy from the parent's checkpoint** — this policy inheritance is what
  makes iterative design + RL improvement affordable,
- `--immigrants` (default 1) fresh random designs per generation keep
  exploring.

Outputs, under `logs\coadapt\<timestamp>\`:

- `history.csv` — every candidate ever evaluated: generation, score,
  distance, the six design scales
- `best.json` — best design so far, with its checkpoint path
- `gen_XX\cand_YY\result.json` + `agent.pt` — per-candidate details
- `gen_XX\cand_YY\skrl\` — per-candidate TensorBoard logs

**Resuming:** if a run is interrupted, re-run with
`--run-dir logs\coadapt\<the old timestamp>`; finished candidates are
detected by their `result.json` and skipped.

**Budget guidance:** `--iterations 100` at `--num_envs 256` is ~800k env
steps per candidate — enough to rank designs, not enough for a polished gait.
For a final result, take the best design from `best.json` and give it a long
single-design training run (step 3) warm-started from its checkpoint:

```
python scripts/skrl/train.py --task Template-Cheeta-Direct-v0 --headless --num_envs 512 --checkpoint logs\coadapt\<run>\gen_XX\cand_YY\agent.pt "env.design_vector=[...]"
```

## 5. Watch the winner

`evolve.py` prints the exact command when it finishes; in general:

```
python scripts/coadapt/play_design.py --design 1.2 0.9 1.0 1.4 1.0 0.8 --checkpoint logs\coadapt\<run>\gen_04\cand_02\agent.pt --real-time
```

Without `--checkpoint` the design runs passively (zero torques).

## Script reference

| Script | Purpose |
|---|---|
| `scripts/run_random_cheeta.py` | random-action rollout, distance report |
| `scripts/make_cheeta.py` | build/preview designs with custom limb lengths |
| `scripts/skrl/train.py` / `play.py` | PPO training/playback for one design |
| `scripts/coadapt/train_design.py` | worker: train + score one design (used by evolve) |
| `scripts/coadapt/evolve.py` | evolutionary co-adaptation driver |
| `scripts/coadapt/play_design.py` | watch a design + checkpoint |
| `scripts/list_envs.py`, `zero_agent.py`, `random_agent.py` | template utilities |
