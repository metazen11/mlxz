# AGENTS.md

This repository uses `CONTRACT.md` as the binding operating rule for all
engine, benchmark, and docs work.

## Must Read

Before editing or reviewing anything substantive, read:

1. `CONTRACT.md`
2. `README.md`
3. `docs/whitepaper.md`
4. `docs/experiments.md`

## Required Behavior

- Treat the contract as the default rule set for performance work.
- Do not propose or merge engine changes without benchmark evidence.
- Keep changes aligned with the continual test-improve-measure loop until the
  project reaches its goal of consistently outperforming plain `mlx-lm` on the
  agreed workloads.
- If a change does not move the engine, document it in the experiments log
  instead of claiming it as progress.

