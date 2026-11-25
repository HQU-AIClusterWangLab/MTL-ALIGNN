# Execution order

These are historical task labels from the original VS Code project. For the compact package, use the root README entry points and regenerate omitted graph caches explicitly.

## Initial checks

Run the following VS Code tasks in order:

1. `00 - Environment, data and model smoke test`
2. `01 - Run unit tests`

## Staged execution

1. `10 - Prepare all data and supervised graphs`
2. `20 - Run all main experiments`
3. `30 - Run all ablation experiments`
4. `40 - Screen Unknown pool and export Top-100`

## Sequential execution

`90 - Run all default experiments sequentially`

This task does not run OOD experiments. Rerun it after an interruption: completed experiments are skipped, and incomplete neural-network experiments resume from `last_checkpoint.pt`.

## Optional OOD experiments

After the default main experiments, run:

`95 - Run optional brute-force OOD experiments`

## Expected computational workload

The default workflow includes multiple seeds and 18 ablation training members, so a complete run may take substantial time. Complete Smoke checks and one M04 and M06 seed before starting the full sequential workflow.
