# fremtpl2-pricing

Motor third-party liability pricing on the public freMTPL2 data (French policies): claim
frequency, claim severity and pure premium, comparing GLMs with gradient boosting and
evaluating the results with actuarial criteria.

Status: milestone 1 (scaffold, download, data audit, cleaning rules, pre-registered split) is
complete, with the post-audit, pre-model amendments of 2026-11-07 applied (matched-row targets,
per-claim cap 100000, no ClaimNb clipping; see section 8 of `reports/data_audit.md`). No models or
results yet.

## Reproduce

```sh
make setup      # uv sync (installs the pinned Python and dependencies)
make download   # fetch freMTPL2freq / freMTPL2sev from OpenML into data/raw/
make audit      # results/audit.json + reports/data_audit.md + results/orphan_ledger.csv
make split      # verifies the frozen data/processed/split.parquet (never overwritten); results/split_summary.json,
                # results/targets_summary.json (about 2.5 minutes)
make test lint
```

The audit report documents every cleaning rule with measured counts; the judgement calls were decided
on 2026-11-07 and are listed in its section 8. Settings (paths, seed, dataset ids, cleaning rules, split) live in `configs/default.yaml`. Data provenance and
checksums are in `data/README.md`.
