# Project conventions

- Layout: `src/fremtpl2_pricing/` (package, CLI `fremtpl2`: `download`, `audit`, `split`; modules
  `data` (verified load), `audit`, `clean` (single config-driven cleaning function), `targets`, `groups`,
  `split`), `configs/` (YAML), `tests/` (offline, synthetic frames in `tests/synth.py`),
  `data/raw` and `data/processed` (git-ignored, regenerated), `results/` and `reports/` (tracked
  pipeline outputs: `audit.json`, `split_summary.json`, `targets_summary.json`, `orphan_ledger.csv`, `data_audit.md`).
- Evaluation protocol: the grouped split (group = tuple of all rating factors) is the pre-registered
  primary; the random split is a labelled sensitivity. Do not change the choice after seeing test
  results. Cleaning runs on the full data before splitting; severity joins by IDpol after assignment.
- Target definitions (config `clean.targets`, fixed by the human decision of 2026-11-07, see section 8
  of `reports/data_audit.md`): one policy-level target table built by `targets.build_target_table`
  from matched severity rows: `N` (number of matched rows, the frequency target; `frequency_count:
  sev_rows` is the default, recorded ClaimNb is the sensitivity `claim_nb`), `S` (sum of amounts),
  `S_cap` (sum of per-claim capped amounts, cap `clean.large_loss.cap` 100000 applied before
  aggregation) and `excess = S - S_cap`; `ClaimNb_raw` and `claim_discrepancy` are audit-only, not
  predictors. N = 0 exactly when S = 0. There is no ClaimNb clipping. Amounts that are not finite
  and > 0 and exposures <= 0 are quarantined and counted; orphan severity rows (IDpol not in freq)
  are excluded and listed in `results/orphan_ledger.csv`. Exposure is capped at 1 and is only an
  offset / weight, never a rating factor. Use the columns added by `clean`; do not re-derive
  targets ad hoc. Severity labels come from `split.attach_severity` (inner join by IDpol after
  assignment).
- Sensitivities are declared in `clean.sensitivities` (rule overrides) and selected by name with
  `clean.clean_sensitivity`; they change cleaning/targets only, never the split. The excess loading
  (`clean.excess_loading`, `targets.excess_loading`) is computed from the primary train set only;
  its value is in `results/targets_summary.json`.
- The registered split is frozen: `fremtpl2 split` recomputes the assignment in memory, fails if it
  differs from `data/processed/split.parquet` and never overwrites that file. Do not change the
  stratification label or the group key.
- Config, not code: paths, dataset ids, seeds and thresholds come from `configs/*.yaml`; never hardcode them.
- Results only from pipeline files: every number in README or reports must come from a file
  written by a pipeline command; no hand-typed metrics.
- Commands: `make setup`, `make test` (`uv run pytest -q`), `make lint` (ruff check + format --check).
  Tests are offline; only `fremtpl2 download` touches the network.
- Commits: small and focused, one line `type(scope): summary`, no footer. Do not push unless explicitly asked.
