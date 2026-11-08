# Project conventions

- Layout: `src/fremtpl2_pricing/` (package, CLI `fremtpl2`: `download`, `audit`, `split`, `train`, `sensitivities`; modules
  `data` (verified load), `audit`, `clean` (single config-driven cleaning function), `targets`, `groups`,
  `split`, `features` (fold-fitted preprocessing), `glm` (glum GLMs), `metrics`, `train` (CV + final refits),
  `selection` (pure functions of protocol section 4: severity ladder, freq x sev vs Tweedie, with traces), `cv_report` (renders `reports/glm_cv.md` from `results/cv_glm.json`)), `configs/` (YAML), `tests/` (offline, synthetic frames in `tests/synth.py`),
  `data/raw` and `data/processed` (git-ignored, regenerated), `results/` and `reports/` (tracked
  pipeline outputs: `audit.json`, `split_summary.json`, `targets_summary.json`, `orphan_ledger.csv`, `data_audit.md`,
  `cv_glm.json`, `glm_relativities_*.csv`, `feature_map.json`, `glm_models.json`, `glm_cv.md`,
  `cv_sensitivities.json`, `glm_sensitivities.md`). `fremtpl2 train` runs the grouped CV of the GLMs on train only; `fremtpl2 gbm` (module `gbm`, report `gbm_report`; config `models.gbm`) tunes the LightGBM frequency/severity/Tweedie candidates on the same registered folds, refits on train, selects the GBM pure-premium candidate by protocol section 4 and runs the CV-only monotonicity diagnostic (outputs `cv_gbm.json`, `gbm_models.json`, `gbm_cv.md`, model files in git-ignored `data/processed/models/`);
  `fremtpl2 sensitivities` runs the declared `clean.sensitivities` in CV only.
- Evaluation protocol: the grouped split (group = tuple of all rating factors) is the pre-registered
  primary; the random split is a labelled sensitivity. Do not change the choice after seeing test
  results. `reports/evaluation_protocol.md` (2026-11-08) governs milestone 3 and the single test
  evaluation; it must not be changed after test access except by a dated, reasoned amendment made
  before the test evaluation. Cleaning runs on the full data before splitting; severity joins by IDpol after assignment.
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
- Milestone 2 (`fremtpl2 train`, config `models:`): CV only until the final evaluation. It loads
  the TRAIN rows of the frozen split (never a test row in any fit, preprocessing or metric; CV/fit functions raise
  on `cv_fold < 0`), runs grouped 5-fold CV of frequency/severity/Tweedie GLMs and baselines (features refitted
  per training part; severity has its own claims-only feature map; Tweedie power chosen by inner CV on the held-out
  Tweedie log-likelihood (dispersion by ML on the fitting part; Poisson deviance per power is a diagnostic, power 1.5 a
  fixed-power sensitivity; grid-edge/flatness and solver-convergence diagnostics are reported; `metrics.tweedie_logpdf`
  is the Dunn & Smyth series, scipy is a declared dependency), then refits on full train. The same CV also runs the pre-registered severity ladder (config `models.glm.severity_ladder`: constant < simplified < L2 full < full, frequency fixed at the full GLM; L2 penalty from strictly positive `models.glm.l2_grid`, chosen on the inner folds per outer fold and by plain CV for the final refit) and the section 4 rule, recorded under `selection` in `cv_glm.json`; every ladder candidate is refitted into `glm_models.json` as `severity_<id>`.
  Before the CV loop `train.check_level_coverage` fails fast if a feature map would not cover its validation part. Outputs: `results/cv_glm.json`, `results/glm_relativities_*.csv`,
  `results/feature_map.json`, `results/glm_models.json`, `reports/glm_cv.md`. The excess loading is only
  reported (separate block), never part of a CV target. The bootstrap block is recorded, not applied, until milestone 3.
- Commands: `make setup`, `make test` (`uv run pytest -q`), `make train` (CV of the GLMs on train only), `make gbm` (GBM tuning, train only; about 10-15 minutes, see `results/cv_gbm.json` timings), `make lint` (ruff check + format --check).
  Tests are offline; only `fremtpl2 download` touches the network.
- Commits: small and focused, one line `type(scope): summary`, no footer. Do not push unless explicitly asked.
