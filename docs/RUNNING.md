# Running the co-adaptation pipeline

Written for: whoever runs the experiments on this machine (Windows, RTX 4090). Full details are in [GUIDE.md](GUIDE.md).

## Setup (once per terminal)

```
conda activate env_isaaclab
cd C:\Users\knorbaeck\code\cheeta
```

`pip install -e source/cheeta` must have been run once. `wandb login` must have been done once for `--wandb`; drop the flag to run without logging.

## 1. Smoke test (1 min)

```
python scripts/check_multi_design.py --headless --num_envs 8 --designs 4
```

Prints the designs, the observation shape (8 x 23) and the per-design distances. If this works, everything else will.

## 2. Train the general policy (1-2 h)

```
python scripts/coadapt/train_general.py --phases 4 --designs-per-phase 64 --num_envs 512 --iterations 300 --wandb
```

Result: `logs\coadapt\general\<timestamp>\general.pt`. Resume an interrupted run with `--run-dir logs\coadapt\general\<timestamp>`.

## 3. Evolve designs (hours, scales with population and iterations)

```
python scripts/coadapt/evolve.py --general-checkpoint logs\coadapt\general\<timestamp>\general.pt --generations 10 --population 12 --elites 2 --immigrants 2 --train-top 3 --num_envs 256 --iterations 100 --wandb
```

Result: `logs\coadapt\<timestamp>\history.csv` and `best.json`. Each generation costs one screening session plus one fine-tuning session per trained candidate (2-3 min each). Resume with `--run-dir`.

## 4. Metric benchmark

```
:: proof of concept, ~15 min
python scripts/coadapt/benchmark_metrics.py --general-checkpoint logs\coadapt\general\<timestamp>\general.pt --designs 6 --iterations 60 --repeat 2 --wandb
:: overnight
python scripts/coadapt/benchmark_metrics.py --general-checkpoint logs\coadapt\general\<timestamp>\general.pt --designs 40 --iterations 400 --repeat 6 --wandb
```

Result: `logs\coadapt\benchmark\<timestamp>\correlations.csv` (one line per metric, sorted by |Spearman|), `table.csv`, `summary.json`. Re-analyse without training: add `--analyze-only --run-dir <that folder>`.

## 5. Watch a result

```
python scripts/coadapt/play_design.py --design 1.2 0.9 1.0 1.4 1.0 0.8 --checkpoint logs\coadapt\<run>\gen_04\cand_02\agent.pt --real-time
```

`evolve.py` prints the exact command for its best design when it finishes. The general checkpoint plays any design.

## Tips

- Each job is one Isaac Sim process; the 30-45 s start-up is unavoidable.
- All drivers skip finished jobs when re-run with `--run-dir`, so a crash costs at most one job.
- wandb: one group per run, job types `general-train`, `screen`, `candidate`, `benchmark`, and a `*-driver` run with the summary tables.
- Old checkpoints (before the 23-D observation) do not load.
