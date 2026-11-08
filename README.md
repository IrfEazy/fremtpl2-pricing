# fremtpl2-pricing

Motor third-party liability pricing on the public freMTPL2 data (French policies): claim
frequency, claim severity and pure premium, comparing GLMs with gradient boosting and
evaluating the results with actuarial criteria.

Status: milestone 1 (scaffold, download, data audit, cleaning rules, pre-registered split) is
complete, with the post-audit, pre-model amendments of 2026-11-07 applied (matched-row targets,
per-claim cap 100000, no ClaimNb clipping; see section 8 of `reports/data_audit.md`). Milestone 2
(grouped cross-validation of the frequency, severity and Tweedie GLMs on the training set only, with
baselines and final refits) is implemented as `fremtpl2 train`; its outputs are `results/cv_glm.json`,
`results/glm_relativities_*.csv` and `reports/glm_cv.md`. The declared cleaning sensitivities are run in CV only
(`fremtpl2 sensitivities`). The LightGBM candidates (tuning, refit, monotonicity diagnostic) are implemented as `fremtpl2 gbm`. The test set is untouched. Milestone 3 (gradient boosting and the single final test evaluation) is
pre-registered in [`reports/evaluation_protocol.md`](reports/evaluation_protocol.md), written before any test access.

## Reproduce

```sh
make setup      # uv sync (installs the pinned Python and dependencies)
make download   # fetch freMTPL2freq / freMTPL2sev from OpenML into data/raw/
make audit      # results/audit.json + reports/data_audit.md + results/orphan_ledger.csv
make split      # verifies the frozen data/processed/split.parquet (never overwritten); results/split_summary.json,
                # results/targets_summary.json (about 2.5 minutes)
make train      # CV of the GLMs on train only: results/cv_glm.json, glm_relativities_*.csv, reports/glm_cv.md
make gbm        # LightGBM candidates tuned in grouped CV on train only: results/cv_gbm.json, results/gbm_models.json, reports/gbm_cv.md
make test lint
```

The audit report documents every cleaning rule with measured counts; the judgement calls were decided
on 2026-11-07 and are listed in its section 8. Settings (paths, seed, dataset ids, cleaning rules, split) live in `configs/default.yaml`. Data provenance and
checksums are in `data/README.md`.
