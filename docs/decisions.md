# Design Decisions and User Preferences

## Decisions made during refactor (2026-05-06)

### Keep autoresearch loop intact
- The agent triplet (train.py + prepare_combined.py + prepare.py) stays as-is
- Don't extract evaluation/logging from prepare_combined.py (would break agent contract)
- Agent only modifies train.py, everything else immutable
- Re-exports in prepare_combined.py maintain backward compatibility

### DESeq2 path abandoned
- prepare_pcl.py deleted (DESeq2 fold changes, `/teamspace/` paths)
- All PCL titration scripts deleted
- The *concept* of per-cell-line gene masking is worth keeping — but should use pdex data, not DESeq2

### Separate parquets for pdex and cell_eval
- pdex (long format): FC, p-value, FDR, ref_mean — covers 62K genes
- cell_eval (wide format): expression deltas — may cover different gene subsets
- Kept separate because gene sets may differ (e.g. HVG 2K vs full 62K)

### Fold changes computed within plate
- pdex and cell_eval deltas are computed per-plate using that plate's DMSO controls
- When aggregated across plates: mean FC, min p-value/FDR
- Most treatments appear on only 1 plate (~86%), so this rarely matters for Tahoe
- For Parse, the equivalent is per-donor (each donor has its own PBS baseline)

### target_sum is dataset-specific
- Tahoe: `target_sum=1872` (for normalize_total in full-genome cell_eval)
- Parse: `target_sum=3320` (dataset-wide median transcript count)
- Configured in YAML (`scripts/data_prep/configs/`)

### Sensitivity prediction: two strategies
- **Strategy 1**: Train Rhaister on all 57K Tahoe pairs, use DE predictions as features for downstream ranker. Sparse labels (~282 seqrun, ~9K PRISM).
- **Strategy 2**: Train Rhaister directly on smaller sensitivity-labelled subsets. Dense labels, weaker DE model.
- Both should be tested empirically (issue #11)

### Rhaister0 = zero-shot variant (integrated)
- Rhaister fewshot: ALS + regression + MLP + calnet (default mode)
- RhaisterO zeroshot: EB shrinkage + subspace projection using DMSO centroids
- Integrated into `train.py` via `--mode zeroshot` dispatch to `_zeroshot_train_and_evaluate()`
- Zeroshot uses separate cache (`_zs` suffix) and split config (`zeroshot_*.toml`)
- FC and p-val heads ported from RhaisterO master branch (D→FC scale + z-score calibration)

## User preferences

### Code style
- Use editable install (`pip install -e .`) instead of sys.path hacks
- Follow tahoe-x1 repo structure for release quality
- Config-driven scripts (YAML) instead of per-dataset script variants
- Pre-commit with ruff lint + format

### Evaluation
- A/B half-split correlation is a replication floor, not a noise ceiling
- Use Spearman-Brown (2*r_ab/(1+r_ab)) for the real upper bound
- d_pred (expression deltas) preferred over y_pred (log2 FC) for sensitivity features

### Data
- Storage at /nvme-shared/shreshth/ for scratch; don't delete others' files
- Never push to master without confirmation
- All changes committed locally first, confirm before GitHub pushes
