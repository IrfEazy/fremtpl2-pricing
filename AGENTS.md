# Project conventions

- Layout: `src/fremtpl2_pricing/` (package, CLI `fremtpl2`), `configs/` (YAML), `tests/`,
  `data/raw` and `data/processed` (git-ignored, regenerated), `results/`, `reports/`.
- Config, not code: paths, dataset ids, seeds and thresholds come from `configs/*.yaml`; never hardcode them.
- Results only from pipeline files: every number in README or reports must come from a file
  written by a pipeline command; no hand-typed metrics.
- Commands: `make setup`, `make test` (`uv run pytest -q`), `make lint` (ruff check + format --check).
  Tests are offline; only `fremtpl2 download` touches the network.
- Commits: small and focused, one line `type(scope): summary`, no footer. Do not push unless explicitly asked.
