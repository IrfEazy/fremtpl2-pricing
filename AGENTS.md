# Project conventions

- Layout: `src/fremtpl2_pricing/` (package, CLI `fremtpl2`: `download`, `audit`, `split`, `train`, `sensitivities`, `gbm`, `interpret`, `freeze`, `evaluate`, `report`, `reproduce`, `docs`; modules
  `data` (verified load), `audit`, `clean` (single config-driven cleaning function), `targets`, `groups`,
  `split`, `features` (fold-fitted preprocessing), `glm` (glum GLMs), `metrics`, `train` (CV + final refits),
  `selection` (pure functions of protocol section 4: severity ladder, freq x sev vs Tweedie, with traces), `cv_report` (renders `reports/glm_cv.md` from `results/cv_glm.json`), `interpret` (train-only interpretation), `report` (test report), `docs_render` (generated documents), `figures` (deterministic matplotlib helpers)), `configs/` (YAML), `tests/` (offline, synthetic frames in `tests/synth.py`),
  `data/raw` and `data/processed` (git-ignored, regenerated), `results/` and `reports/` (tracked
  pipeline outputs: `audit.json`, `split_summary.json`, `targets_summary.json`, `orphan_ledger.csv`, `data_audit.md`,
  `cv_glm.json`, `glm_relativities_*.csv`, `feature_map.json`, `glm_models.json`, `glm_cv.md`,
  `cv_sensitivities.json`, `glm_sensitivities.md`). `fremtpl2 train` runs the grouped CV of the GLMs on train only; `fremtpl2 gbm` (module `gbm`, report `gbm_report`; config `models.gbm`) tunes the LightGBM frequency/severity/Tweedie candidates on the same registered folds, refits on train, selects the GBM pure-premium candidate by protocol section 4 and runs the CV-only monotonicity diagnostic (outputs `cv_gbm.json`, `gbm_models.json`, `gbm_cv.md`, model files in git-ignored `data/processed/models/`);
  `fremtpl2 sensitivities` runs the declared `clean.sensitivities` in CV only.
  `fremtpl2 interpret` (module `interpret`; config `interpret:`, `figures:`) runs BEFORE the freeze, on a fixed train sample
  (config size/seed, ids recorded, `cv_fold >= 0` asserted), loads the frozen artifacts through the freeze loaders and never
  refits: LightGBM `pred_contrib` contributions of each GBM component (per-row sum = raw score, tolerance recorded; the
  exposure offset is outside the contributions), group means of contributions against the log relativities (stored coefficients) of the GLM of the SELECTED GLM candidate for
  the same component, other GLMs as labelled extra columns, on the aligned log scale (never contributions as coefficients; the
  severity comparison uses claim rows only, weighted by N), an exploratory DrivAge x VehPower note, the BonusMalus pure-premium
  monotonicity sweep (distinct train BonusMalus values) of the six candidates (PM-M3-1: only the frequency and Tweedie GBMs are constrained; freq x sev is not
  guaranteed monotone; a limit, never retuned) and the constraint register from `cv_gbm.json`. Outputs
  `results/interpretation.json`, `reports/interpretation.md`, `reports/figures/interp_*.png`; commit them before `freeze`.
  Tracked outputs are run-state independent so a fresh clone regenerates them byte for byte: the test report states a fixed journal-check sentence (the git-ignored mirror is still checked when present, never rendered); `results/interpretation.json` `sources` hash a JSON input carrying telemetry as canonical JSON without `check.volatile_keys` of `configs/reproduce.yaml` (`content_sha256_excluding`), model files by bytes; `results/split_summary.json` holds no run-status text (printed to stdout only).
  `fremtpl2 report` (module `report`; config `report:`) renders `reports/test_evaluation.md` and `reports/figures/test_*.png`
  only from `results/metrics.json` and the completed attempt it points to (files checked against the COMPLETE hashes); with no
  completed attempt it exits non-zero and writes nothing. It runs only after the real `evaluate` (explicit, never in `make all`).
- Freeze and test evaluation (config `freeze:`, protocol section 6). `fremtpl2 freeze` (module `freeze`, train only,
  clean committed tree) recomputes both selections from the recorded per-fold losses, checks the config against the
  training records and the feature-map level coverage against `results/audit.json`, and writes
  `results/freeze_manifest.json` (six candidates with fit-free prediction recipes, `selected_glm_id`/`selected_gbm_id`,
  trace hashes, sha256 of every input/model/config/protocol/`uv.lock`/split/raw file, training commit, library versions,
  evaluation spec, train-row self-check predictions); it refuses an existing manifest unless `--force`, and `--force`
  once a test-state file exists. Commit the manifest, then `fremtpl2 evaluate` (module `evaluate`) is the ONLY code path
  that reads test rows (`evaluate.load_test`): lock, clean tree, hashes, runtime identity (python, packages, `uv.lock`,
  threads), selection and primary-contrast re-assertion and the self-check (edge cases, determinism) first; then
  `FROZEN`/`TEST_OPENED` are appended to `results/test_access.json` (the single authoritative journal; the mirror in
  `data/processed/` is only a check, any disagreement fails closed) before any test row is read; outputs are staged
  in `data/processed/evaluate_staging/`, validated and hashed, `COMPLETE` records the hashes, and only then are they
  published to `results/test/attempt-<n>/` and the pointer `results/metrics.json` (`--publish-only` republishes a
  COMPLETE attempt without recomputing). A second run is refused; only `--technical-rerun --reason` after an attempt
  with TEST_OPENED and no COMPLETE, with the same manifest and runtime. `evaluate.run_attempt` is the one code path:
  `fremtpl2 evaluate --rehearsal` runs it on train fold `freeze.rehearsal.fold` with its own journal and outputs under
  `data/processed/rehearsal/` (no test row, no `results/` write); run it before the real evaluation. Agents never run
  `freeze` or `evaluate` (rehearsal included) on the real project unless the lead says so; `make all` stops at `freeze`. A crash between the journal append and the mirror write makes `evaluate` fail closed; the operator inspects both files and restores the mirror from the authoritative journal only (never the reverse), recorded as a technical note.
- Reproduction (post-test verification policy in `configs/reproduce.yaml`, top-level `check:` and `reproduce:`; deliberately outside `configs/default.yaml`, whose sha256 must equal the `config` entry of `results/freeze_manifest.json`, guarded by `tests/test_manifest.py`; `freeze --check` and `reproduce` load it from `--reproduce-config`, default `DEFAULT_REPRODUCE_CONFIG`, merged onto a copy of the main config as `freeze.check` / `reproduce`; a change of it in the clone is a source change and fails `reproduce`). `fremtpl2 freeze --check` (module `freeze`, `run_freeze_check`) is
  strict and read-only (writes nothing, needs no clean tree, allowed after test access, exit 1 on any mismatch, structured JSON on
  stdout): the manifest is tracked and the HEAD blobs of its tracked files equal the manifest hashes; the manifest rebuilt in
  memory by `build_manifest` equals the committed one except the declared `run_specific_keys` (zero tolerance); untracked
  regenerated artifacts (models, split, raw) match the manifest sha256; tracked telemetry files (`telemetry_files`) are compared as
  parsed JSON with the declared `volatile_keys` removed; once a test-state file exists the published evidence is verified
  read-only with `report.load_completed` (journal COMPLETE, `results/metrics.json` pointer, every attempt file against the COMPLETE
  hashes). `compare_values` is the one pure comparison: a float difference is "numeric" only at the config-declared
  `check.numeric_paths` (per-fold losses, constants, stored predictions) and only outside `exact_keys`; every other
  difference (ids, strings, ints, bools, key sets, list lengths, hyperparameters, chosen powers/penalties, manifest fields) is
  structural and fails `reproduce`. `fremtpl2 reproduce [--force-clean]` (module `reproduce`) needs a clean source tree, clones
  HEAD into `paths.processed/reproduce.workdir` (real path strictly inside `paths.processed`, not a symlink, never the root or an
  ancestor; no `data/`, so no stale artifact), deletes every tracked file under `results/` and `reports/` of the clone except
  `reproduce.preserve_paths` (published evidence and inputs no step regenerates), runs `reproduce.steps` with per-step seconds
  and exit codes (stops at the first failure), then compares every tracked output with the HEAD blob: an output a step did not
  regenerate fails; "exact" is equal bytes or a JSON that differs only in declared volatile keys; parsed-equal files with other
  byte differences are "equivalent" and values within rtol/atol "tolerance" (portable at most); `byte_exact_paths` (renderer
  outputs of committed evidence, e.g. `reports/test_evaluation.md`) must be byte-identical; a `.png` whose bytes differ passes
  only as `pixel_equal` (portable, never exact) if the decoded pixels (`matplotlib.image.imread`) are identical, a pixel
  difference or an undecodable figure fails. Any other tracked change, any untracked file outside `results/` and `reports/`, and any
  symlinked or escaping artifact path fails. Each step may declare `outputs` (also under ignored paths such as `data/processed/`)
  that must exist as regular files after it exits 0, else FAIL. The comparison rules (tolerances, exact keys, volatile keys,
  byte-exact and preserved paths, the inventory of tracked outputs) are snapshotted from the SOURCE tree and commit before any
  step and never re-read from the clone. JSON is parsed strictly (duplicate keys are rejected); shape (keys, types, list lengths
  and order) is compared first and never tolerated, non-finite numbers pass only if identical to the committed ones, `exact_keys`
  cover their descendants, and a float passes iff `|a - b| <= atol + rtol * |a|` with `a` the committed value. A `freeze --check`
  failure whose mismatches are all numeric does not stop the run (strict check recorded as failed, `report` and the
  classification still run); integrity, structural or execution failures stop it. Verdicts:
  `EXACT_PASS`, `PORTABLE_NUMERIC_PASS` (strict check failed only numerically, outputs within tolerance or figures pixel-equal),
  `FAIL` (including a raw data checksum mismatch at download and any step failure not listed next), `UNSUPPORTED_ENVIRONMENT`
  (only exit 127 of a `reproduce.tool_commands` tool such as `uv`/`git`, or a `reproduce.transport_exit_codes` code, 75 for no
  network at download). The clone must be at the tested commit (`git rev-parse` in the clone), else FAIL. The record goes to
  `results/reproduction/run-<n>.json` of the source tree (n = highest existing + 1; the run history is immutable: files are
  created exclusively, an existing one is never overwritten, a modified committed record refuses the run, and the directory is
  excluded from the comparison and the purge) with the tested and clone commits, tool versions (uv, git, python), the preserved
  evidence identities (journal, pointer and manifest sha256, attempt id) and the recomputable inventory.
  Scope: `EXACT_PASS` means exact reproduction of the declared recomputable artifacts plus verification of preserved published
  evidence; held-out metrics are not recomputed. Amendment A1 (dated 2026-11-08, explicitly post-hoc, revealed by run-1 which
  failed the original strict policy; `amendments` in `configs/reproduce.yaml`, pinned to `original_policy_sha256` of that file at
  `b0008e1`): three declared deviations, each pinned to one artifact, field paths and values and verified from the actual files on
  every run (`freeze.verify_amendment`), nothing ignored by key name and never expanded after a run (a further difference needs a
  new dated amendment). D1 download metadata: `data/raw/manifest.json` `/datasets/0/retrieved` and `/datasets/1/retrieved`
  (committed 2026-11-07, new value an ISO `YYYY-MM-DD` date, both recorded); with the committed values put back and the producer's
  serialisation the file must hash to the frozen entry (every other field exact) and the raw parquet files stay byte-equal. D2
  GLM training config echo, three places found from the code (`train.config_summary` -> `results/cv_glm.json` `/config`,
  `freeze.check_training_record` -> manifest `/training_record/glm_config_sha256_at_training`, `cv_report` -> the "Config hash"
  line of `reports/glm_cv.md`; `glm_models.json` has none): the committed echo equals the current config restricted to the
  committed keys, the added keys are exactly `models.gbm`, `models.bootstrap.interval`, `models.bootstrap.quantile_method`,
  `models.evaluation.lorenz_points` with pinned canonical-JSON sha256 values, the old/new hashes are pinned (the new one is
  recomputed from the current main config with the producer's serialisation; the old one covers the whole config of the time and
  can only be compared with the pin) and everything else in the file is exact; the static evidence (a regex grep listing of the modules reading each added key and the GLM training modules matching none) is re-scanned from the tested source and recorded with a label as informational documentation only: a regex can miss another read idiom, so it is never a pass/fail input and not proof of irrelevance; D2 covers the hash line of `reports/glm_cv.md` only as a substitution that D3 applies before its comparison. D3 markdown telemetry (the original policy declared the JSON `timing_seconds` keys volatile but not their markdown renderings; found from the renderers): only `reports/glm_cv.md` and `reports/glm_sensitivities.md` (table `## Timing (seconds)`, second column, `cv_report`) and `reports/gbm_cv.md` (`## Checks and timings`, the `- <stage>: <n> s` bullets after `- passed:`, `gbm_report`) render timings; `audit`, `interpret` (timings only in its JSON), `report` (`opened (UTC)` of the committed journal, byte-exact) and `docs_render` (README / model card, not compared) render none. Within a pinned section only the pinned cells may differ, as finite non-negative decimals with the same padding and key, and each changed cell must equal the producer's rendering (`cv_report._f`, `gbm_report._f` with one decimal) of the corresponding `timing_seconds` value in the regenerated JSON named by the spec (`json`); the pinned heading, `after` marker and table header must exist exactly once; every other byte of the file is identical (`freeze.md_telemetry_differences`), and any other file or section is not covered. Each run record
  D4 derived provenance hashes (amendment key `derived_hashes`, dated 2026-11-08 after run-3; no observed value is pinned; not a `declared_deviations` kind, so `docs_render` has no description line for it): a difference at `/sources/<key>/sha256` of `results/interpretation.json` is covered only if (a) the clone value equals the hash recomputed by `interpret.source_entry_from_bytes` (the producer's own normalisation, the same code `interpret._sources` uses; never a copy) from the clone's input file, (b) the HEAD value equals the same recomputation from the HEAD blob of that input and (c) the input file is covered as a whole by a satisfied earlier deviation (e.g. D2 for `results/cv_glm.json`); every other byte of the artifact must equal HEAD once the covered hashes are put back, and a key not in `/sources` of both sides, an untracked input or a broken pin covers nothing. Each application is recorded (key, base, new, `covered_by`). Docs: the `docs` step (after `report`; `reproduce.docs_outputs` = `README.md`, `MODEL_CARD.md`, tracked outside `results/`/`reports/`) regenerates `reports/model_report.md` and the generated blocks of those two files in the clone; they are never purged, are compared byte for byte with HEAD (`byte_exact_paths`; any difference fails) and are not source changes.
  keeps `strict_verdict` (original policy, unchanged; `verdict` is its alias) and adds `amended_verdict`:
  `PASS_WITH_DECLARED_DEVIATIONS` only if every difference is matched by a satisfied deviation and every step and comparison
  completed (a skipped step, a missing file, a crash, a parser error or an unclassified difference gives the strict verdict). It is fail closed: neither passing verdict is possible unless every configured step ran and every expected comparison executed. The amended pass forbids any difference no deviation declares: within-tolerance numeric, parsed-equal reformatted and pixel-equal PNG re-encoded files are PORTABLE-class and make the amended verdict equal the strict one (a strict-check failure is suppressed inside `reproduce` only when every failed check is positively classified as covered or numeric-only, never on a generic non-zero exit). It is never an exact pass under
  the original policy. `reproduce.operational_retry: none`: a failed or incomplete run (an exception included) is recorded as its own `run-<n>` and never replaced or retried in place; the history must be contiguous (a gap refuses the run). Each record carries `decision_procedure`: the original and amended policy sha256, the sha256 of the verifier sources (`reproduce.py`, `freeze.py`), the tested commit and the exact rerun command. Known stale note: the Makefile comment on the `reproduce` target still says it writes `results/reproduction.json`; the record is `results/reproduction/run-<n>.json` (Makefile was outside the allowed paths of the amendment work). The record lists the applied deviations (base and new values), the original and amended policy hashes and
  the complete check inventory. A `freeze_check` step whose failures are all covered (or numeric) does not stop the run;
  `freeze --check` stays strict (exit code and `passed` unchanged) and only adds a `declared_deviations` report. `fremtpl2 reproduce` prints both verdicts and exits 0 only for `EXACT_PASS` or `PASS_WITH_DECLARED_DEVIATIONS` (non-zero for `PORTABLE_NUMERIC_PASS`, `FAIL` and `UNSUPPORTED_ENVIRONMENT`; the amended verdict is the strict one when no amendment applies). Reproduction rebuilds the pre-test artifacts and verifies them against the
  published freeze, validates (exact inventory: no missing or unexpected attempt entry, one COMPLETE, unique attempt identity, a
  pointer resolving inside `results/test/`, no symlink) and re-renders the published evaluation outputs (`report`), and never
  recomputes held-out metrics; `freeze` and `evaluate` are not steps. Accepted limits: volatile keys are matched by name at any
  depth (the declared list is verified to be telemetry only), and byte-different model files are structural (the `model_files`
  check is not numeric) and block portability until a semantic model comparison exists. Never run `reproduce`,
  `freeze --check` or `make all` on the real project unless the lead says so. `make all` ends with `freeze --check` instead of
  `freeze` once `results/test_access.json` exists, then `report` when `results/metrics.json` exists; `make freeze-check`,
  `make reproduce`.
- Generated documents (`fremtpl2 docs`, module `docs_render`, config `configs/docs.yaml` loaded by `config.load_docs_config`
  like `reproduce.yaml`; a `docs` key in `default.yaml` is a ConfigError, which stays frozen). It renders `reports/model_report.md`
  (fully generated) and the blocks between `<!-- docs:NAME:begin -->` / `<!-- docs:NAME:end -->` markers in `README.md`
  (`results`, `reproduction`, `drivers`) and `MODEL_CARD.md` (`metrics`) only from tracked pipeline files: `results/metrics.json`
  and the completed attempt it points to (verified by `report.load_completed`), `cv_glm/cv_gbm/cv_sensitivities/interpretation/
  audit/split_summary/targets_summary.json`, the GLM relativity tables, `gbm_models.json` and the run records `reproduction/run-<n>.json` when present
  (none -> "not yet recorded"; the block renders only the LATEST run record: tested commit, date, verdict(s) as recorded, total and per-step minutes,
  machine; plus the declared deviations of `configs/reproduce.yaml` as plain one-line policy items (text per deviation kind in `docs.yaml:deviation_items`);
  no history table, no run numbers, no narrative; earlier records stay in `results/reproduction/`). Rule: every number in README, model card and model report goes through a docs block or the
  model report, never typed by hand; text outside the markers is preserved byte for byte, a missing or duplicated marker is an
  error, output is deterministic and `fremtpl2 docs --check` writes nothing and exits non-zero on drift. Runs after `report`
  (`make docs`, explicit, never in `make all`); adding a block means one render function in `BLOCKS`. README.md and MODEL_CARD.md prose contains no digit outside the generated blocks, commands, file names, links, figure paths and the dataset name; text that conflicts with the rendered numbers is fixed after each `make docs`. `.github/workflows/ci.yml` (push and pull_request; ruff check and format, `pytest -q`, `fremtpl2 docs --check`; offline, no secrets) must stay in step with `make test lint docs`.
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
  reported (separate block), never part of a CV target. The bootstrap block is not applied in train/CV; `fremtpl2 evaluate` applies it to the test set (paired group bootstrap, `metrics.ratio_bootstrap`).
- Commands: `make setup`, `make test` (`uv run pytest -q`), `make train` (CV of the GLMs on train only), `make gbm` (GBM tuning, train only; about 10-15 minutes, see `results/cv_gbm.json` timings), `make lint` (ruff check + format --check), `make interpret` (train-only interpretation; before `freeze`), `make freeze` (train only), `make rehearsal` (the evaluate code path on a train fold, no test row), `make evaluate` (the one-shot test evaluation; never in `make all`), `make report` (test report after `evaluate`; explicit), `make docs` (model report and README / model card blocks after `report`; explicit), `make freeze-check`, `make reproduce` (see Reproduction).
  Tests are offline; only `fremtpl2 download` touches the network.
- Commits: small and focused, one line `type(scope): summary`, no footer. Do not push unless explicitly asked.
