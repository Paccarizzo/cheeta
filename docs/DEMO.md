# Meeting demo: showing that the pipeline works, and what the benchmark tells us

Written for: the person presenting the project in a meeting, using the demo run under `logs\coadapt\demo\` and the wandb group `demo-2026-09-18`.

## 1. Proof that everything works (5 minutes)

The demo run is a real, small instance of the whole pipeline: 2 phases of general training on 16 designs each, 2 generations of evolution with screening, and a 6-design benchmark with 2 seed repeats. Budgets are about 5 % of a production run, so the numbers are indicative, not results.

Show these in order:

1. **The general policy learns across designs.** Open wandb, group `demo-2026-09-18`, run `general-phase00` and `general-phase01`: `Info / episode_return` rises within each phase, and phase 1 starts higher than phase 0 (warm start on new designs). The driver run `general-driver` plots `score_mean` per phase. File: `logs\coadapt\demo\general\phase_01\result.json`, field `per_design`: one score per design from a single policy.
2. **Screening scores a whole population in one session.** `logs\coadapt\demo\evolve\gen_00\screen\result.json`: 8 designs scored zero-shot, with `value_s0` per design, in about one minute. In wandb the `screen` runs have no training curves, only the `per_design` table.
3. **Only the selected candidates are trained.** `logs\coadapt\demo\evolve\history.csv`: column `trained` is 1 for the elite plus the top 2 by zero-shot score, 0 for the rest, which keep their zero-shot score. Column `zero_shot_score` next to `score` shows what the screening predicted versus what fine-tuning delivered.
4. **The loop converges on a design.** `logs\coadapt\demo\evolve\best.json` names the best design and its checkpoint. Live demo, takes 40 s to start:
   ```
   python scripts/coadapt/play_design.py --design <best design> --checkpoint <best checkpoint> --real-time
   ```
   Then the same command with `--checkpoint logs\coadapt\demo\general\general.pt` shows the general policy driving that design without specialisation.
5. **The benchmark produces the correlation table.** `logs\coadapt\demo\bench\correlations.csv`, or the `correlations` table and `spearman_bar` chart in the wandb run `bench-driver`.

If asked "is it fast enough": one Isaac Sim session costs 30 to 45 s to start, a 100-iteration fine-tune at 256 envs is 2 to 3 minutes, screening a population is one session regardless of its size.

## 2. What the benchmark will tell us

The question behind it: **which cheap signal, available before or early in fine-tuning, predicts a design's score after full training?** If one does, evolution can spend training budget only on designs that will pay off.

The benchmark trains N random designs to a long budget (the ground truth) and records, for each design:

| signal | available after | meaning |
|---|---|---|
| `zero_shot_score` | one episode, no training | how good the general policy already is on the design |
| `value_s0` | one forward pass | the critic's own estimate of that; the paper's fitness proxy |
| `reward_step_at_10/25/50pct` | 10 / 25 / 50 % of the budget | score of a short fine-tune |
| `reward_step_slope`, `reward_step_gain` | the whole run | how fast the design learns |
| `value_loss_*`, `policy_std_*`, `learning_rate_*` | free during training | PPO internals; the KL-adaptive learning rate reflects how far the policy moves per update |

The output is one Spearman rank correlation per signal against the final score, plus a **noise floor**: the correlation of the final score with itself across two seeds on the same designs. Reading it:

- A signal near the noise floor is as good as a full training run. Use it for selection.
- A signal near zero carries no information; ignore it, however plausible it sounds.
- If the noise floor itself is low, the final scores are too noisy at that budget: increase iterations or evaluation episodes before trusting any ranking.
- `reward_step_at_25pct` close to the floor means a quarter of the budget already ranks designs, which justifies successive halving: train everything briefly, keep the top half, continue.

Expected shape of the answer, to be confirmed by the overnight run: zero-shot and the critic value rank designs coarsely (they see only what the general policy can already do), early fine-tune reward ranks them well, and PPO internals are weak on their own. Where the final score falls between those is what decides the selection rule.

## 3. Caveats to state up front

- The demo budgets are tiny. Six designs and three seeds give correlations with huge error bars; the overnight run (40 designs, 400 iterations, 6 repeats) is the real measurement.
- Scores from before 18 September used a different evaluation and are not comparable.
- The general policy has not yet been trained to convergence, so its zero-shot ranking is currently a lower bound on what screening can do.

## 4. Decisions to ask for

1. Budget for the overnight benchmark and the production general policy.
2. Whether to implement successive halving and learning-curve extrapolation next, or critic-gradient design proposals first.
3. Whether the reward stays `max(v_x/10, 0)` or gets a control cost.
