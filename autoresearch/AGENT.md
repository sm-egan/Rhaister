# Research Agent Instructions

Read `PROGRAM.md` for project context and `IDEAS.md` for research directions, then follow these instructions.

## Execution loop

1. Read current `train.py`, `results.jsonl`, and the recent entries at the top of `EXPERIMENTS.md` (no need to read the full file — search it if a direction seems familiar)
2. **Branch** before making changes: `git checkout -b exp/<experiment_name>`
3. Make one change, run `python train.py <experiment_name>` (always `tahoe_5_holdout`)
4. **Review results**: read the experiment's entry from `results.jsonl` and compare all metrics (see below) against the current baseline
5. **Decide**: accept if the majority of metrics improve without others regressing significantly; reject if most metrics regress
6. **Record decision**: call `update_decision(experiment_name, "accepted")` or `update_decision(experiment_name, "rejected")` (imported from `prepare_combined`)
7. **Act on decision**: if accepted, merge to master; if rejected, revert the change, stay on master
8. **Log** the experiment in `EXPERIMENTS.md` regardless of outcome
9. **Decide** the next experiment based on what you learned. Don't follow the idea list blindly -- if an experiment reveals something surprising, pursue that lead. If a direction shows diminishing returns, switch to a different idea.
10. **Keep going.** A failed experiment is data, not a stop signal. Log what you learned, update IDEAS.md, and start the next experiment. Only stop when you run out of context.

## Managing IDEAS.md

`IDEAS.md` is a short, actionable list of things to try — not an archive. When an idea has been explored (whether it worked or not), **remove it from IDEAS.md**. The record lives in `EXPERIMENTS.md` and `results.jsonl`. Add new ideas as experiments suggest them. If a direction is exhausted, delete it. Keep IDEAS.md lean so it's quick to scan.

## Metrics to report

The model is evaluated on six State paper metrics plus legacy metrics. Always report all of them:

| Key | Short name | Target |
|-----|-----------|--------|
| `celleval_static/pearson_delta_mean` | ce_pearson | D (expression deltas, primary) |
| `pdex_static/pearson_delta_mean` | fc_pearson | Y (fold change) |
| `state/pearson_delta_mean` | delta_pearson | D (expression deltas) |
| `state/spearman_lfc_sig_mean` | spearman_lfc | Y, F (FC + FDR) |
| `state/pr_auc_mean` | pr_auc | F (FDR) |
| `state/de_overlap_mean` | de_overlap | Y, F (FC + FDR) |
| `state/de_spearman_sig` | spearman_sig | F (FDR) |

The current baseline is whatever the most recent `decision: accepted` entry in `results.jsonl` records — compare against that, not against a number hard-coded here.

## Experiment notes (`EXPERIMENTS.md`)

Maintain a running log of experiments. For each entry record:
- **Name**: experiment name as passed to `train.py`
- **Hypothesis**: what you expected and why
- **Change**: what you modified (briefly)
- **Results**: all metrics in a table
- **Interpretation**: what this tells you -- why did it work or not?
- **Next**: what this suggests trying next -- add promising new ideas to `IDEAS.md`

New entries go at the top (newest first). This log is your memory across experiments — search it before starting a new experiment to avoid repeating failed approaches and to build on what worked.

## Constraints

- Modify `train.py` only (do NOT modify `prepare_combined.py`, `prepare.py`, `state_metrics.py`, or `eval_splits.py`)
- Every experiment must call `log_result()` and then `update_decision()`
- **NO test set leakage**: `evaluate_test` called once, at the very end
- Always report all metrics listed above
- **Always use `tahoe_5_holdout`** -- do not pass `--split` with other values
- Each experiment MUST complete in under 15 minutes (900 s). Wall time for the full `train_and_evaluate()` call is recorded as the top-level `runtime_seconds` field on each `results.jsonl` record — use it to check your budget and to compare runtime against the baseline.
- Do NOT make runtime significantly worse without clear accuracy gains
