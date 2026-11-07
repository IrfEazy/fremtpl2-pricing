# fremtpl2-pricing

Motor third-party liability pricing on the public freMTPL2 data (French policies): claim
frequency, claim severity and pure premium, comparing GLMs with gradient boosting and
evaluating the results with actuarial criteria.

Status: milestone 1 in progress (scaffold and data download). No results yet.

## Reproduce

```sh
make setup      # uv sync (installs the pinned Python and dependencies)
make download   # fetch freMTPL2freq / freMTPL2sev from OpenML into data/raw/
make test lint
```

Settings (paths, seed, dataset ids) live in `configs/default.yaml`. Data provenance and
checksums are in `data/README.md`.
