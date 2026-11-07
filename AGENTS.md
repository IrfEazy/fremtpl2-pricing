# Project conventions

- Layout: `src/fremtpl2_pricing/` (package, CLI `fremtpl2`: `download`, `audit`, `split`; modules
  `data` (verified load), `audit`, `clean` (single config-driven cleaning function), `groups`,
  `split`), `configs/` (YAML), `tests/` (offline, synthetic frames in `tests/synth.py`),
  `data/raw` and `data/processed` (git-ignored, regenerated), `results/` and `reports/` (tracked
  pipeline outputs: `audit.json`, `split_summary.json`, `data_audit.md`).
- Evaluation protocol: the grouped split (group = tuple of all rating factors) is the pre-registered
  primary; the random split is a labelled sensitivity. Do not change the choice after seeing test
  results. Cleaning runs on the full data before splitting; severity joins by IDpol after assignment.
- Target definitions (config `clean.targets`, for human review, see R5 in `reports/data_audit.md`):
  the severity sample is the policies with severity rows and the severity weight is the number of
  severity rows (not ClaimNb); the pure-premium / Tweedie observed cost is the sum of the capped
  severity amounts, so claims without severity rows count as zero cost, while frequency x severity
  counts them at the average severity. The two routes are not on the same target unless frequency is
  restricted to claims with severity rows (`frequency_count: sev_rows`); the default is
  `claim_nb`. The ClaimNb cap never changes severity rows. Use the `freq_count`, `sev_weight`, `cost`
  columns added by `clean`; do not re-derive weights ad hoc. Severity labels come from
  `split.attach_severity` (inner join by IDpol after assignment; orphan rows excluded).
- Config, not code: paths, dataset ids, seeds and thresholds come from `configs/*.yaml`; never hardcode them.
- Results only from pipeline files: every number in README or reports must come from a file
  written by a pipeline command; no hand-typed metrics.
- Commands: `make setup`, `make test` (`uv run pytest -q`), `make lint` (ruff check + format --check).
  Tests are offline; only `fremtpl2 download` touches the network.
- Commits: small and focused, one line `type(scope): summary`, no footer. Do not push unless explicitly asked.
