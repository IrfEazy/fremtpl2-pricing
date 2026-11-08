# Evaluation protocol for milestone 3 and the final test evaluation

Date: 2026-11-08. Written before any test-set access. This document may only be amended with dated,
reasoned entries appended at the end, made before the test evaluation, except the technical-rerun
entry defined in section 6, which is logged after test access. No other edit is allowed after test
access. It contains no metric values: results are referred to by file name. Settings named
below live in `configs/default.yaml`, except those marked "(to be added)", which do not exist yet
and are added and committed in milestone 3 before the first tuning run that uses them.

## 1. Estimand and targets

For policy i with capped exposure E_i and matched claim amounts X_ij:

- S_cap_i = sum_j min(X_ij, cap), cap = `clean.large_loss.cap`, applied before aggregation.
- Annual capped cost Y_i = S_cap_i / E_i. This is the estimand: expected annual capped cost per
  policy. It is not an uncapped technical premium.
- Frequency target: matched claim count N_i (not recorded `ClaimNb`), Poisson with offset log E_i.
- Severity response: S_cap_i / N_i with weight N_i, on policies with N_i > 0 only.
- Primary loss for model m: L_m = sum_i E_i d_p(Y_i, mu_mi) / sum_i E_i, where d_p is the Tweedie
  unit deviance at p = `models.tweedie.reference_power` (fixed, common to all models) and mu_mi is
  the predicted annual capped cost. On CV, the per-fold loss is this exposure-weighted mean on the
  fold; the K folds are then averaged with equal weight.
- "Capped prediction" means the expected capped-claim cost; it is never a truncation of the annual
  prediction.
- Loaded premium = mu_mi + the training-only flat excess loading per exposure-year. It is reported
  separately, never compared with capped targets, and no claim of predictive adequacy is made for it.

Zero-claim retention, orphan severity rows, quarantine and the absence of ClaimNb clipping follow
`AGENTS.md` (target definitions) and are not restated here.

## 2. Split interpretation

The primary split is grouped on the tuple of the 9 raw rating factors. A test result therefore
measures generalisation to held-out raw rating profiles. It is not temporal or geographic
validation and not customer-level validation. After banding and pooling, transformed profiles of
train and test may coincide. The random split remains a labelled sensitivity only.

## 3. Candidate models

Reported on the test set (all prespecified, all reported regardless of outcome):

- pure premium null (training mean);
- BonusMalus-only;
- GLM frequency x selected severity candidate;
- GLM Tweedie;
- GBM frequency x GBM severity;
- GBM Tweedie (a fixed candidate, implemented in milestone 3).

Severity candidates, compared in CV only: constant (claim-weighted mean of the fitting part);
simplified GLM (BonusMalus capped, DrivAge band, VehAge band, VehGas); L2-regularised full GLM;
full GLM. The L2 penalty grid (to be added, e.g. `models.glm.l2_grid`) excludes zero. The penalty is
chosen inside each outer training part on its inner registered folds, with the primary loss as inner
loss. The final full-train refit uses the penalty chosen by plain CV over the registered folds of the
full train set, with the primary loss, as for the Tweedie power below. In the severity ladder the
frequency model is held fixed at the full GLM frequency, so the ladder isolates the severity component.

Tweedie power (GLM and GBM Tweedie): chosen by the held-out full Tweedie log-likelihood (series
density, dispersion MLE on the fitting part, prior weights) over `models.tweedie.power_grid`. It is
chosen per outer fold by inner CV and finally by plain CV on the full train set; the final value is
recorded in `results/glm_models.json` and is the single power used for GBM Tweedie. The reference
power of the primary loss stays fixed regardless.

A non-converged or failed fit fails the run: the cause is fixed technically and the run repeated
before freezing. No candidate is silently excluded.

## 4. Selection rule (CV within train only)

- Primary loss: section 1, at `reference_power`, for every comparison below (including frequency x
  severity versus Tweedie).
- Secondary: weighted normalised Gini (definition and tie handling as in `metrics.py`, ties
  aggregated), calibration (totals, by equal-exposure deciles and by factor), Poisson deviance on
  frequency only for models with a frequency component. Poisson deviance on the pure premium, if
  shown, is a Poisson-form mean score on monetary outcomes, not a likelihood.

### Selection algorithm

1. For each candidate compute the primary loss on each of the K = `split.cv_folds` registered folds.
2. Order candidates on an explicit complexity ladder. Severity ladder: constant < simplified GLM <
   L2-regularised full GLM < full GLM, judged on the pure-premium product (frequency x severity); the
   severity Gamma deviance is reported alongside.
3. Start with the simplest rung as incumbent. For the next rung let d_k = loss_incumbent,k -
   loss_next,k.
4. Paired-fold complexity heuristic: replace the incumbent iff mean(d) > sd(d, ddof=1) / sqrt(K).
   Exact ties keep the simpler candidate. This is a heuristic, not a significance test and not a
   valid standard error: training sets overlap across folds and K is small.
5. Comparisons are always against the current incumbent, never against the best-loss candidate.
   Continue up the ladder; the final incumbent is the selected candidate.
6. Frequency x severity versus Tweedie of the same family (GLM or GBM): with d_k = loss_tweedie,k -
   loss_freqxsev,k, the lower mean CV primary loss wins, but if |mean(d)| <= sd(d, ddof=1) / sqrt(K) frequency x severity is kept.
7. Differences within the heuristic are reported as inconclusive, not as wins.

## 5. GBM search space (milestone 3)

- LightGBM. Objectives: Poisson for frequency, Gamma for severity (weight N), Tweedie for pure
  premium (weight exposure, power from section 3).
- Exposure: log-exposure is the `init_score` in both training and validation sets. Predicted counts
  are computed explicitly as exp(raw score + log E), never relying on `predict()` to restore init
  scores; annual rates are exp(raw score). A synthetic test is required: doubling exposure doubles
  expected counts and leaves annual rates unchanged.
- The hyperparameter grid, seeds, round cap, early-stopping patience, minimum improvement, monitored
  metric, rounding of the mean best iteration (to the nearest integer) and behaviour at the round
  cap (cap reached: recorded, and the cap is used) are config keys (to be added, e.g. under
  `models.gbm`), committed before the first tuning run.
- Each component is tuned on the registered grouped CV folds inside train with its own objective
  deviance (frequency Poisson, severity Gamma weighted by N, Tweedie at the section 3 power weighted
  by exposure). The grid point with the lowest mean CV loss is chosen; the final number of rounds is
  the mean of per-fold best iterations; the model is refit on full train. The final GBM pure-premium
  candidate is chosen by section 4.
- Early stopping and grid scoring use the same registered fold; GBM CV scores are therefore
  optimistic tuning scores, not unbiased estimates. This does not affect test validity.
- Component-objective tuning is not end-to-end optimisation of the primary loss.
- Monotone constraints (frozen): non-decreasing in BonusMalus for the frequency (Poisson) and Tweedie
  GBMs; the severity (Gamma) GBM is unconstrained; no other constraints.
- No calibration step is fitted on the test set.

## 6. Test evaluation

- Run once, after every milestone 3 choice is frozen in config and committed. Tuning and model
  changes stop unconditionally after test access.
- A technical failure is narrowly a crash, I/O error or environment problem. A rerun restores the
  registered analysis only (no model, config or code-path change). Original outputs are preserved
  and the access is logged as an amendment. Sensitivities and failures never produce a new
  confirmatory claim on the same test set.
- Every prespecified model is reported, regardless of outcome.
- Metrics: Poisson, Gamma and Tweedie deviances; normalised Gini; lift and double lift;
  calibration totals, by decile and by factor. The loaded premium is reported separately.
- Uncertainty: paired group bootstrap. The resampling unit is the rating-profile group: whole groups
  are resampled with multiplicity weights shared by all models; policies are a labelled
  sensitivity. This amends the original plan wording ("resampling policies"); approved by the human
  on 2026-11-08. B, seed and level come from `models.bootstrap`; intervals are percentile intervals.
  Each replicate recomputes all denominators, Gini rankings and oracle normalisation with the
  multiplicity weights. Treating profiles as independent units is an assumption. Inference concerns
  held-out rating profiles of this dataset only, conditional on the fitted models; retraining and
  tuning variability are not included.
- Single primary contrast: Delta = L_GBM - L_GLM, the selected GBM and GLM pure-premium candidates
  (chosen in CV by section 4) on the primary loss. The Gini difference is secondary; all other pairs
  are descriptive.
- Decision: improvement is claimed iff the upper interval endpoint of Delta < 0; deterioration iff
  the lower endpoint > 0; otherwise no demonstrated difference.
- Report the number of groups, group sizes and exposure concentration.

## 7. Sensitivities

The declared `clean.sensitivities` are run in CV only (`results/cv_sensitivities.json`,
`reports/glm_sensitivities.md`, both produced by `fremtpl2 sensitivities`). Comparisons are made within each target definition, never
across targets. The test evaluation uses the primary definitions only.

## 8. Known limitations carried into the test

- The Tweedie likelihood of the capped target is a working likelihood: per-claim capping creates
  point masses at multiples of the cap.
- Sandwich standard errors are policy-level and conditional on the chosen specification.
- The severity signal is weak in CV (see `results/cv_glm.json` and `reports/glm_cv.md`); the severity
  candidates of section 3 exist for that reason.

## Amendments

None. Entry template (append only; before test access, or after it only for the technical rerun of
section 6):

- Date: YYYY-MM-DD. Reason: ... Author: ... Test access: before / after. Change: before -> after.
