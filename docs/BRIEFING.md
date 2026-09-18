# Cheeta co-adaptation: status briefing (2026-09-18)

Written for: the next project discussion (people who know the Luck et al. co-adaptation setup, not necessarily this code).

## Goal

Find the Half-Cheetah leg design (six segment-length scale factors in [0.8, 2.0]) and the policy that run fastest, following Luck, Ben Amor & Calandra (CoRL 2019), reimplemented in Isaac Lab with PPO (skrl) instead of SAC.

## What exists now

**1. A general, design-conditioned policy (the paper's "population network").**
The policy observes the design vector next to the 17-D state. Many designs are trained at once in a single simulation: each parallel env holds its own morphology (64 designs over 512 envs is the default). Training runs in phases with a fresh random design batch per phase. Script: `scripts/coadapt/train_general.py`.

**2. Evolution with cheap screening.**
`scripts/coadapt/evolve.py` proposes a population (elites, mutations, random immigrants), scores every candidate zero-shot with the general policy in one simulation session (no training), and fine-tunes only the top few, each from its parent's specialised policy or from the general one. The fine-tuned score is the fitness. The critic's value at the start state, the paper's fitness proxy, is recorded for every candidate as well.

**3. A benchmark for selection metrics.**
`scripts/coadapt/benchmark_metrics.py` trains random designs long, records cheap signals collected before and during fine-tuning (zero-shot return, critic value, learning-curve slope and early rewards, PPO losses, learning rate as a KL proxy) and reports each metric's Spearman correlation with the final score, plus a seed-repeat noise floor. The evolution loop prints the same correlation every generation.

**4. Weights & Biases logging** on every script (`--wandb`, project `cheeta`), with per-generation candidate tables.

## Findings and caveats

- Multi-design scenes work in Isaac Lab (heterogeneous envs, physics replication off). Building 64 assets costs 1-2 minutes on first use, then they are cached.
- Two pre-existing bugs were found and fixed: skrl's wrapper reseeded every run to seed 0 (so seeds never mattered), and the old evaluation started mid-episode instead of from a real reset. Scores from before 2026-09-18 are not comparable to new ones.
- Everything has only been tested at toy budgets (a few PPO iterations). No real general policy has been trained yet, so there are no results on which metrics predict good designs.
- One simulation session per job remains necessary (Isaac Sim cannot rebuild the scene in-process); screening amortises the 30-45 s start-up over the whole population.

## Next steps

1. Train the general policy for real: 4 phases x 64 designs x 300 iterations (roughly 1-2 h). Check on wandb that the mean score over designs keeps rising across phases; if it plateaus, more phases or a larger network are the first knobs.
2. Proof-of-concept benchmark: 6 designs, 60 iterations, 2 repeats (about 15 min). Then the overnight run: 40 designs, 400 iterations, 6 repeats.
3. Run evolution with `--train-top 3` and compare `--select-by zero_shot`, `value_s0` and `random` on the same seed and budget.

## Open questions for the discussion

- Should specialised training data flow back into the general policy (periodic retraining on the current elites)? Not implemented yet.
- Ground-truth budget for the benchmark: how long must fine-tuning be before scores stop reordering designs? The seed-repeat noise floor from the overnight run will tell.
- Is the reward `max(v_x/10, 0)` still the right objective, or should a control cost be added as in Gym's HalfCheetah?
