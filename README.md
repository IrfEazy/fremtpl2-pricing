# fremtpl2-pricing

Motor third-party liability pricing on the public freMTPL2 data (French policies): claim
frequency, claim severity and pure premium, comparing GLMs with gradient boosting and
evaluating the results with actuarial criteria.

Status: milestone 1 (scaffold, download, data audit, cleaning rules, pre-registered split) is
complete pending human review of `reports/data_audit.md`. No models or results yet.

## Reproduce

```sh
make setup      # uv sync (installs the pinned Python and dependencies)
make download   # fetch freMTPL2freq / freMTPL2sev from OpenML into data/raw/
make audit      # results/audit.json + reports/data_audit.md (counts, leakage, cleaning impact)
make split      # data/processed/split.parquet + results/split_summary.json (about 2 minutes)
make test lint
```

The audit report documents every cleaning rule with measured counts; judgement calls are marked
"for human review". Settings (paths, seed, dataset ids, cleaning rules, split) live in `configs/default.yaml`. Data provenance and
checksums are in `data/README.md`.
