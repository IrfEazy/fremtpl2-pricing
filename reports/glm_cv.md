# GLM cross-validation (milestone 2)

Rendered from `cv_glm.json` by `fremtpl2 train`; every number below comes from that file.

## Protocol

- Test set used: **False**. All fits, preprocessing and metrics use the primary training set only (registered `cv_fold`); the test set is untouched in this milestone.
- Split: `grouped`, 5 outer folds; features, rare-level grouping and every model are fitted on the other folds of each outer fold.
- Tweedie power: grid [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], chosen per outer fold by an inner CV over the remaining folds with the `tweedie_loglik` loss: the held-out Tweedie log-likelihood of S_cap / Exposure with prior weight = exposure, dispersion estimated by maximum likelihood on the fitting part only (Dunn & Smyth 2005 series density). The exposure-weighted Poisson deviance per power is kept as a labelled diagnostic. Common Tweedie deviance at the fixed reference power 1.5, which also enters the pure premium comparison as the fixed-power sensitivity `tweedie_ref`.
- Power selection by log-likelihood targets distributional fit. The exposure-weighted Poisson deviance of S_cap / Exposure per power is kept as a labelled diagnostic: it is a valid conditional-mean scoring rule, so a different optimum under it is not an error. Fixed-power Tweedie sensitivities [1.3, 1.7] (models `tweedie_fixed_<p>`) enter the pure premium comparison; the common deviance stays at the reference power.
- Severity model: the allowed levels per feature are those of the policy schema of the same fitting part; the severity map is derived from the claims-only fitting rows (minimum 100 claims per level): nominal levels below it merge into OTHER (then the reference), ordinal features (DrivAge, VehAge, VehPower) into the adjacent level with more claims (ties toward the reference). The mapping is stored in `feature_map.json`.
- Inference: coefficient standard errors in the relativity tables are policy-level sandwich standard errors, conditional on the chosen specification (they do not account for the power selection or the level pooling). Relativities are adjusted multiplicative effects versus the stated reference level, not causal effects.
- Targets: frequency count `sev_rows`, per-claim cap 100,000; the excess loading (`flat_per_exposure_year`) is in no CV target or prediction.
- Bootstrap confidence intervals are not part of this milestone (applied: False); they come in milestone 3.
- Config hash `fcf3a1b904659ff237f8dbe36af04149de2d95e2ebfc244ea61851c57fbcedd7`, seed 20260101.

## Data

| train policies | exposure-years | claims (N) | policies with claims |
|---|---|---|---|
| 542,411 | 286,622.9 | 21,123 | 19,966 |

| fold | policies | exposure | claims |
|---|---|---|---|
| 0 | 108,483 | 57,409.0 | 4,215 |
| 1 | 108,483 | 57,390.3 | 4,218 |
| 2 | 108,481 | 57,168.1 | 4,273 |
| 3 | 108,482 | 57,360.0 | 4,230 |
| 4 | 108,482 | 57,295.6 | 4,187 |

## Selected Tweedie power

| fold | selected | at grid edge | loglik spread (nats) |
|---|---|---|---|
| 0 | 1.5 | False | 44,115.7 |
| 1 | 1.5 | False | 45,382.3 |
| 2 | 1.5 | False | 44,978.9 |
| 3 | 1.5 | False | 45,152.7 |
| 4 | 1.5 | False | 44,554.7 |

Inner-CV held-out Tweedie log-likelihood per power (total nats; higher is better).

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -256,118.5 | -227,915.5 | -217,709.4 | -213,408.2 | -212,002.7 | -212,444.7 | -214,477.8 | -218,544.9 | -226,953.1 |
| 1 | -257,525.0 | -228,686.6 | -218,164.2 | -213,671.0 | -212,142.7 | -212,505.2 | -214,487.0 | -218,520.9 | -226,907.2 |
| 2 | -256,582.4 | -227,915.7 | -217,502.9 | -213,083.8 | -211,603.5 | -211,994.6 | -213,989.5 | -218,023.5 | -226,392.7 |
| 3 | -257,077.9 | -228,294.4 | -217,838.8 | -213,405.2 | -211,925.1 | -212,325.6 | -214,338.6 | -218,401.2 | -226,817.2 |
| 4 | -256,582.5 | -228,109.2 | -217,810.1 | -213,460.6 | -212,027.8 | -212,455.4 | -214,483.6 | -218,554.8 | -226,977.8 |

Mean maximum-likelihood dispersion over the inner splits.

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,005.7 | 797.74 | 584.86 | 423.55 | 312.13 | 239.04 | 195.52 | 180.3 | 222.12 |
| 1 | 1,018.4 | 808.7 | 592.35 | 428.26 | 314.88 | 240.61 | 196.43 | 180.81 | 222.43 |
| 2 | 1,015.3 | 806.24 | 590.71 | 427.32 | 314.52 | 240.6 | 196.65 | 181.23 | 223.16 |
| 3 | 1,009.9 | 801.71 | 587.28 | 424.83 | 312.66 | 239.23 | 195.56 | 180.28 | 222.08 |
| 4 | 1,003.5 | 796.03 | 583.63 | 422.75 | 311.53 | 238.59 | 195.2 | 180.04 | 221.87 |

Diagnostic only (not the selection criterion): inner-CV exposure-weighted Poisson deviance of S_cap / Exposure per exposure-year. It is a mean-prediction loss and does not identify the variance power.

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,137.2 | 1,137.4 | 1,137.6 | 1,137.9 | 1,138.4 | 1,138.9 | 1,139.6 | 1,140.5 | 1,141.6 |
| 1 | 1,166.7 | 1,166.8 | 1,167.0 | 1,167.3 | 1,167.8 | 1,168.4 | 1,169.2 | 1,170.2 | 1,171.4 |
| 2 | 1,153.1 | 1,153.3 | 1,153.6 | 1,153.9 | 1,154.5 | 1,155.1 | 1,156.0 | 1,157.1 | 1,158.5 |
| 3 | 1,151.2 | 1,151.3 | 1,151.5 | 1,151.7 | 1,152.1 | 1,152.7 | 1,153.4 | 1,154.4 | 1,155.7 |
| 4 | 1,134.2 | 1,134.2 | 1,134.4 | 1,134.6 | 1,135.0 | 1,135.5 | 1,136.1 | 1,137.0 | 1,138.1 |

Final power (plain CV over the grid on the registered folds): selected 1.5, interior to the grid; held-out log-likelihood spread over the grid 55,990.6 nats, best minus runner-up 513.04 nats.

| power | CV loglik (nats) | CV loglik per policy | diagnostic: Poisson deviance |
|---|---|---|---|
| 1.1 | -320,877.7 | -0.59158 | 1,147.4 |
| 1.2 | -285,161.6 | -0.52573 | 1,147.5 |
| 1.3 | -272,200.8 | -0.50183 | 1,147.7 |
| 1.4 | -266,711.0 | -0.49171 | 1,148.0 |
| 1.5 | -264,887.1 | -0.48835 | 1,148.4 |
| 1.6 | -265,400.1 | -0.4893 | 1,148.9 |
| 1.7 | -267,920.0 | -0.49394 | 1,149.6 |
| 1.8 | -272,994.4 | -0.5033 | 1,150.4 |
| 1.9 | -283,503.1 | -0.52267 | 1,151.5 |

Selection stability: the outer folds select [1.5, 1.5, 1.5, 1.5, 1.5]; 5 of 5 select the final power (0 selections at a grid edge).

Final-CV held-out log-likelihood per outer fold and power (nats).

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -64,827.9 | -57,300.2 | -54,538.0 | -53,343.6 | -52,919.8 | -52,985.4 | -53,466.0 | -54,466.5 | -56,559.0 |
| 1 | -63,445.5 | -56,544.3 | -54,094.7 | -53,090.8 | -52,789.1 | -52,933.4 | -53,464.5 | -54,496.7 | -56,608.7 |
| 2 | -64,384.8 | -57,314.1 | -54,755.7 | -53,677.8 | -53,327.7 | -53,443.1 | -53,960.6 | -54,992.5 | -57,122.0 |
| 3 | -63,875.7 | -56,915.2 | -54,394.0 | -53,326.6 | -52,974.6 | -53,081.3 | -53,584.2 | -54,593.8 | -56,685.6 |
| 4 | -64,343.8 | -57,087.8 | -54,418.4 | -53,272.2 | -52,875.9 | -52,957.0 | -53,444.8 | -54,444.9 | -56,527.7 |

Paired per-fold differences, selected power minus neighbour (nats).

| neighbour | total | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | folds agreeing |
|---|---|---|---|---|---|---|---|
| 1.4 | 1,824.0 | 423.79 | 301.72 | 350.18 | 351.96 | 396.3 | 5 of 5 |
| 1.6 | 513.04 | 65.55 | 144.31 | 115.4 | 106.67 | 81.107 | 5 of 5 |

- Selected 1.5 versus neighbour 1.4: total difference 1,824.0 nats (favours the selected power; 5 of 5 folds agree in sign with the total). Restricted to the policies with a capped claim the difference is 772.98 nats (they favour the selected power), for all other policies 1,051.0 nats.
- Selected 1.5 versus neighbour 1.6: total difference 513.04 nats (favours the selected power; 5 of 5 folds agree in sign with the total). Restricted to the policies with a capped claim the difference is -531.52 nats (they oppose the selected power), for all other policies 1,044.6 nats.

### Cap diagnostics (from the claim rows)

| cap | claims | capped claims | policies with a capped claim | their exposure | amount removed | removed / uncapped total | capped claims in S_cap | share of S_cap |
|---|---|---|---|---|---|---|---|---|
| 100,000 | 21,123 | 36 | 36 | 17.59 | 10,176,659.2 | 20.56% | 3,600,000.0 | 9.15% |

Reconciliation: claim-row S_cap 39,330,661.8 against policy-level S_cap 39,330,661.8; policies with a positive excess 36 against 36 with a capped claim.

The per-claim cap makes S_cap a censored amount: a capped claim sits exactly at the cap, an atom that the continuous Tweedie density does not model, so the likelihood is a working likelihood. 36 capped claims on 36 policies (0.01% of the exposure) carry 9.15% of S_cap. Their effect on the power choice is the capped-policy difference quoted above.

## Dispersion search

225 maximum-likelihood dispersion fits; 57 needed the bracket expanded (total expansions 57, maximum per fit 1); allowed: 5 (a fit still on the bound raises).

| stage | fold | inner fold | power | expansions |
|---|---|---|---|---|
| outer | 0 | n/a | 1.1 | 1 |
| outer | 0 | n/a | 1.2 | 1 |
| outer | 0 | n/a | 1.3 | 1 |
| inner | 0 | 1 | 1.1 | 1 |
| inner | 0 | 1 | 1.2 | 1 |
| inner | 0 | 2 | 1.1 | 1 |
| inner | 0 | 2 | 1.2 | 1 |
| inner | 0 | 3 | 1.1 | 1 |
| inner | 0 | 3 | 1.2 | 1 |
| inner | 0 | 3 | 1.3 | 1 |
| inner | 0 | 4 | 1.1 | 1 |
| inner | 0 | 4 | 1.2 | 1 |
| inner | 0 | 4 | 1.3 | 1 |
| outer | 1 | n/a | 1.1 | 1 |
| outer | 1 | n/a | 1.2 | 1 |
| inner | 1 | 0 | 1.1 | 1 |
| inner | 1 | 0 | 1.2 | 1 |
| inner | 1 | 2 | 1.1 | 1 |
| inner | 1 | 3 | 1.1 | 1 |
| inner | 1 | 3 | 1.2 | 1 |
| inner | 1 | 4 | 1.1 | 1 |
| inner | 1 | 4 | 1.2 | 1 |
| outer | 2 | n/a | 1.1 | 1 |
| outer | 2 | n/a | 1.2 | 1 |
| inner | 2 | 0 | 1.1 | 1 |
| inner | 2 | 0 | 1.2 | 1 |
| inner | 2 | 1 | 1.1 | 1 |
| inner | 2 | 3 | 1.1 | 1 |
| inner | 2 | 3 | 1.2 | 1 |
| inner | 2 | 4 | 1.1 | 1 |
| inner | 2 | 4 | 1.2 | 1 |
| outer | 3 | n/a | 1.1 | 1 |
| outer | 3 | n/a | 1.2 | 1 |
| outer | 3 | n/a | 1.3 | 1 |
| inner | 3 | 0 | 1.1 | 1 |
| inner | 3 | 0 | 1.2 | 1 |
| inner | 3 | 0 | 1.3 | 1 |
| inner | 3 | 1 | 1.1 | 1 |
| inner | 3 | 1 | 1.2 | 1 |
| inner | 3 | 2 | 1.1 | 1 |
| inner | 3 | 2 | 1.2 | 1 |
| inner | 3 | 4 | 1.1 | 1 |
| inner | 3 | 4 | 1.2 | 1 |
| inner | 3 | 4 | 1.3 | 1 |
| outer | 4 | n/a | 1.1 | 1 |
| outer | 4 | n/a | 1.2 | 1 |
| outer | 4 | n/a | 1.3 | 1 |
| inner | 4 | 0 | 1.1 | 1 |
| inner | 4 | 0 | 1.2 | 1 |
| inner | 4 | 0 | 1.3 | 1 |
| inner | 4 | 1 | 1.1 | 1 |
| inner | 4 | 1 | 1.2 | 1 |
| inner | 4 | 2 | 1.1 | 1 |
| inner | 4 | 2 | 1.2 | 1 |
| inner | 4 | 3 | 1.1 | 1 |
| inner | 4 | 3 | 1.2 | 1 |
| inner | 4 | 3 | 1.3 | 1 |

## Solver convergence

glum solver settings: max_iter 1000, gradient_tol 1e-07 (family tweedie: gradient_tol 1e-05). Fits: 244; **not converged: 0**; maximum iterations used 65.

### Solver accuracy check (real data)

The final full-train frequency, severity and Tweedie models and, per outer fold, the CV Tweedie (selected power), frequency and severity fits are refitted with gradient_tol 1e-08 (max_iter 1000). Maximum relative prediction difference 6.2846e-07 (limit 0.0001), relative deviance difference 8.8313e-16 (limit 1e-08), coefficient difference 3.6047e-07 (limit 0.001); violations: 0; passed: **True**.

| stage | fold | model | power | rel. prediction | rel. deviance | coef | iter |
|---|---|---|---|---|---|---|---|
| cv | 0 | tweedie | 1.5 | 3.7577e-07 | 4.4379e-16 | 2.3861e-07 | 3 / 4 |
| cv | 0 | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| cv | 0 | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| cv | 1 | tweedie | 1.5 | 6.2846e-07 | 6.598e-16 | 3.5192e-07 | 3 / 4 |
| cv | 1 | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| cv | 1 | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| cv | 2 | tweedie | 1.5 | 6.1359e-07 | 8.8313e-16 | 3.6047e-07 | 3 / 138 |
| cv | 2 | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| cv | 2 | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| cv | 3 | tweedie | 1.5 | 2.9345e-07 | 2.2155e-16 | 1.8633e-07 | 3 / 90 |
| cv | 3 | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| cv | 3 | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| cv | 4 | tweedie | 1.5 | 1.0954e-07 | 0 | 7.3605e-08 | 3 / 4 |
| cv | 4 | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| cv | 4 | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| final | n/a | frequency | n/a | 0 | 0 | 0 | 6 / 6 |
| final | n/a | severity | n/a | 0 | 0 | 0 | 4 / 4 |
| final | n/a | tweedie | 1.5 | 2.9285e-07 | 0 | 1.7505e-07 | 3 / 4 |


Combined out-of-fold pure premium (frequency x severity) per policy, pipeline solver against tight solver: maximum relative difference 0 (limit 0.0002) at prediction quantile 0.32684; maximum absolute difference 0 at prediction quantile 0.32684.

| fold | max rel. | at quantile | prediction there | max abs. | at quantile | prediction there | total rel. |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.32684 | 30.69 | 0 | 0.32684 | 30.69 | 0 |
| 1 | 0 | 0.16205 | 12.026 | 0 | 0.16205 | 12.026 | 0 |
| 2 | 0 | 0.088698 | 6.8204 | 0 | 0.088698 | 6.8204 | 0 |
| 3 | 0 | 0.91428 | 155.77 | 0 | 0.91428 | 155.77 | 0 |
| 4 | 0 | 0.12901 | 9.6237 | 0 | 0.12901 | 9.6237 | 0 |

Held-out log-likelihood difference (selected power minus grid neighbour) with the pipeline solver and with the tight solver; it must keep its sign and change by less than 1.00% (maximum relative change 4.6853e-08; signs kept: True).

| fold | neighbour | diff (config) | diff (tight) | abs. change | rel. change | ok |
|---|---|---|---|---|---|---|
| 0 | 1.4 | 423.79 | 423.79 | 3.785e-06 | 8.9314e-09 | True |
| 1 | 1.4 | 301.72 | 301.72 | 1.3442e-05 | 4.455e-08 | True |
| 2 | 1.4 | 350.18 | 350.18 | 5.7214e-06 | 1.6338e-08 | True |
| 3 | 1.4 | 351.96 | 351.96 | 5.6408e-06 | 1.6027e-08 | True |
| 4 | 1.4 | 396.3 | 396.3 | 1.1571e-05 | 2.9197e-08 | True |
| total | 1.4 | 1,824.0 | 1,824.0 | 1.8337e-06 | 1.0054e-09 | True |
| 0 | 1.6 | 65.55 | 65.55 | 1.4188e-06 | 2.1645e-08 | True |
| 1 | 1.6 | 144.31 | 144.31 | 3.4985e-06 | 2.4242e-08 | True |
| 2 | 1.6 | 115.4 | 115.4 | 5.4069e-06 | 4.6853e-08 | True |
| 3 | 1.6 | 106.67 | 106.67 | 6.1849e-07 | 5.7982e-09 | True |
| 4 | 1.6 | 81.107 | 81.107 | 3.6939e-06 | 4.5544e-08 | True |
| total | 1.6 | 513.04 | 513.04 | 3.1741e-06 | 6.1869e-09 | True |

Promotion gates of the model artifacts: solver_check True, cap_reconciliation True, convergence True; all passed: **True**.

## Rating-profile groups (train only; the bootstrap unit)

Groups are the tuples of all rating factors; sizes are policies per group. The largest groups are the top 1.00% by policy count.

| scope | groups | policies | size q0.5 | size q0.9 | size q0.99 | max size | top groups | top policy share | top exposure share |
|---|---|---|---|---|---|---|---|---|---|
| overall | 423,006 | 542,411 | 1 | 2 | 4 | 18 | 4,231 | 5.04% | 1.31% |
| fold 0 | 84,614 | 108,483 | 1 | 2 | 4 | 18 | 847 | 5.04% | 1.31% |
| fold 1 | 84,593 | 108,483 | 1 | 2 | 4 | 15 | 846 | 5.04% | 1.29% |
| fold 2 | 84,564 | 108,481 | 1 | 2 | 4 | 15 | 846 | 5.05% | 1.34% |
| fold 3 | 84,600 | 108,482 | 1 | 2 | 4 | 14 | 846 | 5.03% | 1.32% |
| fold 4 | 84,635 | 108,482 | 1 | 2 | 4 | 13 | 847 | 5.04% | 1.28% |

## Findings

- Severity full GLM, out-of-fold deviance explained versus the null: fold mean -0.35% (sd 0.97%), pooled -0.35%; negative in 3 of 5 folds. The full severity model predicts claims out of fold worse than the constant mean: the rating factors carry no usable severity signal at this sample size.
- Tweedie power: selected 1.5, interior to the grid; held-out log-likelihood spread over the grid 55,990.6 nats, best minus runner-up 513.04 nats.
- Selected 1.5 versus neighbour 1.4: total difference 1,824.0 nats (favours the selected power; 5 of 5 folds agree in sign with the total). Restricted to the policies with a capped claim the difference is 772.98 nats (they favour the selected power), for all other policies 1,051.0 nats.
- Selected 1.5 versus neighbour 1.6: total difference 513.04 nats (favours the selected power; 5 of 5 folds agree in sign with the total). Restricted to the policies with a capped claim the difference is -531.52 nats (they oppose the selected power), for all other policies 1,044.6 nats.

## Scalar metrics (fold mean ± sd, and pooled out-of-fold)

### Frequency

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | deviance_per_exposure | 0.47843 ± 0.0028362 | 0.47843 |
| null | deviance_explained | 0 ± 0 | 0 |
| null | gini | 0.0005088 ± 0.0011558 | -0.0038649 |
| null | gini_normalized | 0.00052706 ± 0.0011972 | -0.0040049 |
| bonus_malus_only | deviance_per_exposure | 0.46279 ± 0.00279 | 0.46279 |
| bonus_malus_only | deviance_explained | 0.032688 ± 0.00097661 | 0.032687 |
| bonus_malus_only | gini | 0.23147 ± 0.0075192 | 0.22871 |
| bonus_malus_only | gini_normalized | 0.23986 ± 0.0078221 | 0.23699 |
| full | deviance_per_exposure | 0.45726 ± 0.0030498 | 0.45726 |
| full | deviance_explained | 0.04426 ± 0.0017634 | 0.044258 |
| full | gini | 0.28388 ± 0.0074023 | 0.28376 |
| full | gini_normalized | 0.29416 ± 0.0075423 | 0.29404 |

### Severity (claims only, weight N)

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | deviance_per_claim | 1.3154 ± 0.069139 | 1.3152 |
| null | deviance_explained | 0 ± 0 | 0 |
| full | deviance_per_claim | 1.32 ± 0.069129 | 1.3198 |
| full | deviance_explained | -0.0035466 ± 0.009742 | -0.0035412 |

### Pure premium

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | poisson_deviance_per_exposure | 1,195.0 ± 62.38 | 1,195.0 |
| null | poisson_deviance_explained | 0 ± 0 | 0 |
| null | tweedie_ref_deviance_per_exposure | 77.75 ± 1.9453 | 77.75 |
| null | tweedie_ref_deviance_explained | 0 ± 0 | 0 |
| null | gini | -0.00056204 ± 0.0012568 | -0.022612 |
| null | gini_normalized | -0.00057147 ± 0.0012779 | -0.023003 |
| bonus_malus_only | poisson_deviance_per_exposure | 1,158.0 ± 55.344 | 1,158.0 |
| bonus_malus_only | poisson_deviance_explained | 0.030836 ± 0.0050166 | 0.031019 |
| bonus_malus_only | tweedie_ref_deviance_per_exposure | 74.986 ± 1.4921 | 74.986 |
| bonus_malus_only | tweedie_ref_deviance_explained | 0.035458 ± 0.0057059 | 0.035558 |
| bonus_malus_only | gini | 0.24565 ± 0.025599 | 0.24825 |
| bonus_malus_only | gini_normalized | 0.24989 ± 0.025949 | 0.25253 |
| freq_x_sev | poisson_deviance_per_exposure | 1,146.9 ± 51.071 | 1,146.9 |
| freq_x_sev | poisson_deviance_explained | 0.039946 ± 0.0076236 | 0.040256 |
| freq_x_sev | tweedie_ref_deviance_per_exposure | 74.044 ± 1.2295 | 74.044 |
| freq_x_sev | tweedie_ref_deviance_explained | 0.047506 ± 0.0088966 | 0.047672 |
| freq_x_sev | gini | 0.3044 ± 0.03348 | 0.30487 |
| freq_x_sev | gini_normalized | 0.30966 ± 0.033935 | 0.31013 |
| tweedie | poisson_deviance_per_exposure | 1,148.4 ± 51.057 | 1,148.4 |
| tweedie | poisson_deviance_explained | 0.038707 ± 0.0076194 | 0.039021 |
| tweedie | tweedie_ref_deviance_per_exposure | 74.097 ± 1.2615 | 74.097 |
| tweedie | tweedie_ref_deviance_explained | 0.046828 ± 0.0087122 | 0.046987 |
| tweedie | gini | 0.30396 ± 0.03165 | 0.30428 |
| tweedie | gini_normalized | 0.3092 ± 0.03208 | 0.30953 |
| tweedie_ref | poisson_deviance_per_exposure | 1,148.4 ± 51.057 | 1,148.4 |
| tweedie_ref | poisson_deviance_explained | 0.038707 ± 0.0076194 | 0.039021 |
| tweedie_ref | tweedie_ref_deviance_per_exposure | 74.097 ± 1.2615 | 74.097 |
| tweedie_ref | tweedie_ref_deviance_explained | 0.046828 ± 0.0087122 | 0.046987 |
| tweedie_ref | gini | 0.30396 ± 0.03165 | 0.30428 |
| tweedie_ref | gini_normalized | 0.3092 ± 0.03208 | 0.30953 |
| tweedie_fixed_1.3 | poisson_deviance_per_exposure | 1,147.7 ± 51.252 | 1,147.7 |
| tweedie_fixed_1.3 | poisson_deviance_explained | 0.039284 ± 0.0074604 | 0.03959 |
| tweedie_fixed_1.3 | tweedie_ref_deviance_per_exposure | 74.083 ± 1.2602 | 74.083 |
| tweedie_fixed_1.3 | tweedie_ref_deviance_explained | 0.047012 ± 0.0086598 | 0.047171 |
| tweedie_fixed_1.3 | gini | 0.30328 ± 0.032246 | 0.30363 |
| tweedie_fixed_1.3 | gini_normalized | 0.30851 ± 0.032684 | 0.30887 |
| tweedie_fixed_1.7 | poisson_deviance_per_exposure | 1,149.6 ± 50.717 | 1,149.6 |
| tweedie_fixed_1.7 | poisson_deviance_explained | 0.037703 ± 0.0079342 | 0.038031 |
| tweedie_fixed_1.7 | tweedie_ref_deviance_per_exposure | 74.134 ± 1.2568 | 74.134 |
| tweedie_fixed_1.7 | tweedie_ref_deviance_explained | 0.04635 ± 0.0088835 | 0.046512 |
| tweedie_fixed_1.7 | gini | 0.30429 ± 0.030897 | 0.30459 |
| tweedie_fixed_1.7 | gini_normalized | 0.30954 ± 0.031316 | 0.30985 |

| model | observed total (S_cap) | predicted total | observed / predicted |
|---|---|---|---|
| null | 39,330,661.8 | 39,330,749.5 | 1 |
| bonus_malus_only | 39,330,661.8 | 39,330,870.3 | 0.99999 |
| freq_x_sev | 39,330,661.8 | 39,324,494.2 | 1.0002 |
| tweedie | 39,330,661.8 | 39,668,551.9 | 0.99148 |
| tweedie_ref | 39,330,661.8 | 39,668,551.9 | 0.99148 |
| tweedie_fixed_1.3 | 39,330,661.8 | 39,439,547.0 | 0.99724 |
| tweedie_fixed_1.7 | 39,330,661.8 | 40,093,160.2 | 0.98098 |

## Lift tables (pooled out-of-fold)

### Pure premium: null

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,661.9 | 4,801,510.1 | 3,884,879.2 | 1.2359 |
| 1 | 28,662.8 | 3,458,600.4 | 3,885,020.4 | 0.89024 |
| 2 | 28,661.9 | 4,109,289.4 | 3,899,913.5 | 1.0537 |
| 3 | 28,662.6 | 4,017,929.7 | 3,900,001.8 | 1.0302 |
| 4 | 28,662.1 | 4,057,613.0 | 3,936,532.4 | 1.0308 |
| 5 | 28,662.5 | 3,781,659.5 | 3,936,660.7 | 0.96063 |
| 6 | 28,662.2 | 3,875,402.8 | 3,941,235.4 | 0.9833 |
| 7 | 28,661.8 | 3,918,554.0 | 3,941,338.9 | 0.99422 |
| 8 | 28,663.2 | 4,013,478.3 | 4,002,680.6 | 1.0027 |
| 9 | 28,661.8 | 3,296,624.6 | 4,002,486.5 | 0.82364 |

### Pure premium: bonus_malus_only

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.1 | 2,708,913.7 | 2,941,728.0 | 0.92086 |
| 1 | 28,662.8 | 2,732,489.5 | 2,944,081.6 | 0.92813 |
| 2 | 28,661.7 | 2,750,419.0 | 2,960,507.6 | 0.92904 |
| 3 | 28,662.6 | 2,355,432.1 | 2,977,265.1 | 0.79114 |
| 4 | 28,661.9 | 2,767,846.9 | 2,977,608.4 | 0.92955 |
| 5 | 28,662.2 | 2,691,998.4 | 3,023,307.8 | 0.89041 |
| 6 | 28,663.0 | 3,433,352.9 | 3,155,486.3 | 1.0881 |
| 7 | 28,661.8 | 4,962,161.8 | 3,846,367.5 | 1.2901 |
| 8 | 28,662.5 | 5,467,433.0 | 5,242,078.8 | 1.043 |
| 9 | 28,662.3 | 9,460,614.6 | 9,262,439.2 | 1.0214 |

### Pure premium: freq_x_sev

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.1 | 1,733,357.9 | 1,638,161.7 | 1.0581 |
| 1 | 28,662.6 | 2,164,699.0 | 2,112,403.1 | 1.0248 |
| 2 | 28,662.1 | 2,369,672.7 | 2,416,321.3 | 0.98069 |
| 3 | 28,662.4 | 2,427,116.9 | 2,689,720.3 | 0.90237 |
| 4 | 28,662.7 | 3,224,651.4 | 2,968,577.6 | 1.0863 |
| 5 | 28,661.9 | 2,971,128.3 | 3,276,999.4 | 0.90666 |
| 6 | 28,662.3 | 3,623,950.1 | 3,666,129.1 | 0.98849 |
| 7 | 28,662.1 | 5,026,055.6 | 4,242,613.6 | 1.1847 |
| 8 | 28,662.5 | 5,290,280.3 | 5,348,253.9 | 0.98916 |
| 9 | 28,662.3 | 10,499,749.6 | 10,965,314.1 | 0.95754 |

### Pure premium: tweedie

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.4 | 1,689,226.9 | 1,553,138.8 | 1.0876 |
| 1 | 28,662.5 | 2,229,723.4 | 2,033,696.9 | 1.0964 |
| 2 | 28,662.4 | 2,410,676.9 | 2,346,265.5 | 1.0275 |
| 3 | 28,661.9 | 2,527,504.2 | 2,629,698.6 | 0.96114 |
| 4 | 28,662.2 | 2,874,407.1 | 2,917,683.1 | 0.98517 |
| 5 | 28,662.2 | 3,166,972.7 | 3,238,011.2 | 0.97806 |
| 6 | 28,662.4 | 3,695,488.2 | 3,639,758.7 | 1.0153 |
| 7 | 28,662.9 | 4,812,144.7 | 4,234,993.9 | 1.1363 |
| 8 | 28,661.7 | 5,587,578.2 | 5,395,742.5 | 1.0356 |
| 9 | 28,662.4 | 10,336,939.5 | 11,679,562.8 | 0.88505 |

### Pure premium: tweedie_ref

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.4 | 1,689,226.9 | 1,553,138.8 | 1.0876 |
| 1 | 28,662.5 | 2,229,723.4 | 2,033,696.9 | 1.0964 |
| 2 | 28,662.4 | 2,410,676.9 | 2,346,265.5 | 1.0275 |
| 3 | 28,661.9 | 2,527,504.2 | 2,629,698.6 | 0.96114 |
| 4 | 28,662.2 | 2,874,407.1 | 2,917,683.1 | 0.98517 |
| 5 | 28,662.2 | 3,166,972.7 | 3,238,011.2 | 0.97806 |
| 6 | 28,662.4 | 3,695,488.2 | 3,639,758.7 | 1.0153 |
| 7 | 28,662.9 | 4,812,144.7 | 4,234,993.9 | 1.1363 |
| 8 | 28,661.7 | 5,587,578.2 | 5,395,742.5 | 1.0356 |
| 9 | 28,662.4 | 10,336,939.5 | 11,679,562.8 | 0.88505 |

### Pure premium: tweedie_fixed_1.3

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,661.8 | 1,750,459.1 | 1,595,487.7 | 1.0971 |
| 1 | 28,662.8 | 2,192,982.0 | 2,071,364.6 | 1.0587 |
| 2 | 28,662.5 | 2,411,596.6 | 2,378,899.1 | 1.0137 |
| 3 | 28,662.1 | 2,463,348.1 | 2,657,288.1 | 0.92702 |
| 4 | 28,662.0 | 2,877,787.7 | 2,941,373.8 | 0.97838 |
| 5 | 28,662.7 | 3,272,848.4 | 3,256,997.1 | 1.0049 |
| 6 | 28,662.0 | 3,645,593.0 | 3,652,201.8 | 0.99819 |
| 7 | 28,662.2 | 4,804,721.2 | 4,239,692.1 | 1.1333 |
| 8 | 28,662.5 | 5,547,172.5 | 5,370,798.6 | 1.0328 |
| 9 | 28,662.2 | 10,364,153.3 | 11,275,444.2 | 0.91918 |

### Pure premium: tweedie_fixed_1.7

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.4 | 1,568,528.3 | 1,506,714.0 | 1.041 |
| 1 | 28,662.5 | 2,408,880.5 | 1,993,671.3 | 1.2083 |
| 2 | 28,661.6 | 2,332,248.3 | 2,312,607.0 | 1.0085 |
| 3 | 28,662.4 | 2,623,070.0 | 2,601,364.0 | 1.0083 |
| 4 | 28,662.2 | 2,835,379.0 | 2,894,634.3 | 0.97953 |
| 5 | 28,662.4 | 3,136,155.1 | 3,220,566.0 | 0.97379 |
| 6 | 28,662.4 | 3,592,996.6 | 3,631,464.7 | 0.98941 |
| 7 | 28,662.7 | 4,965,269.4 | 4,239,455.3 | 1.1712 |
| 8 | 28,661.8 | 5,563,961.8 | 5,442,977.1 | 1.0222 |
| 9 | 28,662.4 | 10,304,172.9 | 12,249,706.6 | 0.84118 |

## Double lift: Tweedie vs frequency x severity

`pred_a` is the Tweedie, `pred_b` the frequency x severity prediction.

| bin | exposure | observed | pred_a | pred_b | ratio_a | ratio_b |
|---|---|---|---|---|---|---|
| 0 | 28,662.2 | 3,139,029.2 | 2,659,854.1 | 2,992,149.6 | 1.1802 | 1.0491 |
| 1 | 28,662.7 | 2,855,052.7 | 2,824,644.0 | 3,037,086.9 | 1.0108 | 0.94006 |
| 2 | 28,661.9 | 3,086,120.5 | 2,980,170.4 | 3,144,142.6 | 1.0356 | 0.98155 |
| 3 | 28,662.5 | 3,332,459.5 | 3,179,937.2 | 3,304,385.0 | 1.048 | 1.0085 |
| 4 | 28,662.5 | 3,243,648.2 | 3,370,032.6 | 3,452,928.3 | 0.9625 | 0.93939 |
| 5 | 28,662.1 | 3,547,217.4 | 3,614,629.1 | 3,649,814.7 | 0.98135 | 0.97189 |
| 6 | 28,662.1 | 4,186,148.7 | 3,878,023.2 | 3,852,957.4 | 1.0795 | 1.0865 |
| 7 | 28,662.3 | 4,946,071.1 | 4,254,518.5 | 4,146,622.9 | 1.1625 | 1.1928 |
| 8 | 28,662.4 | 4,639,983.4 | 4,870,531.0 | 4,627,231.0 | 0.95266 | 1.0028 |
| 9 | 28,662.2 | 6,354,931.1 | 8,036,211.7 | 7,117,175.7 | 0.79079 | 0.8929 |

## Calibration

### Frequency by decile: null

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.2 | 2,366.0 | 2,104.8 | 1.1241 |
| 1 | 28,662.5 | 1,915.0 | 2,104.9 | 0.9098 |
| 2 | 28,662.4 | 2,310.0 | 2,112.0 | 1.0938 |
| 3 | 28,662.3 | 1,920.0 | 2,112.0 | 0.90911 |
| 4 | 28,661.9 | 2,317.0 | 2,113.7 | 1.0962 |
| 5 | 28,662.2 | 1,897.0 | 2,113.7 | 0.89747 |
| 6 | 28,662.3 | 2,298.0 | 2,114.3 | 1.0869 |
| 7 | 28,662.5 | 1,908.0 | 2,114.3 | 0.90243 |
| 8 | 28,661.9 | 2,289.0 | 2,116.7 | 1.0814 |
| 9 | 28,662.6 | 1,903.0 | 2,116.8 | 0.89902 |

### Frequency by decile: bonus_malus_only

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.3 | 1,421.0 | 1,589.9 | 0.89376 |
| 1 | 28,662.7 | 1,594.0 | 1,594.8 | 0.99948 |
| 2 | 28,661.4 | 1,428.0 | 1,596.9 | 0.89423 |
| 3 | 28,662.8 | 1,497.0 | 1,597.5 | 0.93708 |
| 4 | 28,662.3 | 1,419.0 | 1,597.5 | 0.88825 |
| 5 | 28,662.5 | 1,433.0 | 1,602.7 | 0.89412 |
| 6 | 28,662.3 | 1,861.0 | 1,688.4 | 1.1022 |
| 7 | 28,662.4 | 2,933.0 | 2,066.8 | 1.4191 |
| 8 | 28,661.5 | 2,969.0 | 2,816.5 | 1.0541 |
| 9 | 28,662.7 | 4,568.0 | 4,972.0 | 0.91875 |

### Frequency by decile: full

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.3 | 937 | 1,018.7 | 0.91984 |
| 1 | 28,661.9 | 1,168.0 | 1,257.4 | 0.92887 |
| 2 | 28,662.7 | 1,355.0 | 1,408.2 | 0.96222 |
| 3 | 28,662.6 | 1,404.0 | 1,538.2 | 0.91277 |
| 4 | 28,661.6 | 1,690.0 | 1,675.5 | 1.0086 |
| 5 | 28,662.8 | 1,852.0 | 1,824.7 | 1.015 |
| 6 | 28,662.0 | 2,058.0 | 2,011.7 | 1.023 |
| 7 | 28,662.8 | 2,406.0 | 2,286.6 | 1.0522 |
| 8 | 28,662.1 | 3,209.0 | 2,817.6 | 1.1389 |
| 9 | 28,662.1 | 5,044.0 | 5,283.4 | 0.95468 |

### frequency by BonusMalus: full

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 50-59 | 369,106.0 | 11,653.0 | 12,203.6 | 0.95488 |
| 60-69 | 58,591.0 | 2,873.0 | 2,082.3 | 1.3797 |
| 70-79 | 35,758.0 | 1,730.0 | 1,528.1 | 1.1321 |
| 80-99 | 57,026.0 | 2,784.0 | 3,326.6 | 0.83688 |
| 100-124 | 20,129.0 | 1,709.0 | 1,571.7 | 1.0873 |
| 125+ | 1,801.0 | 374 | 409.68 | 0.9129 |

### pure_premium by BonusMalus: tweedie

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 50-59 | 369,106.0 | 21,050,341.3 | 21,567,526.2 | 0.97602 |
| 60-69 | 58,591.0 | 4,781,297.8 | 3,752,846.4 | 1.274 |
| 70-79 | 35,758.0 | 3,632,604.3 | 2,861,085.9 | 1.2697 |
| 80-99 | 57,026.0 | 5,120,405.4 | 6,801,448.0 | 0.75284 |
| 100-124 | 20,129.0 | 4,010,286.0 | 3,581,506.2 | 1.1197 |
| 125+ | 1,801.0 | 735,727.0 | 1,104,139.3 | 0.66634 |

### frequency by DrivAge: full

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 18-20 | 5,433.0 | 509 | 509.22 | 0.99956 |
| 21-25 | 25,697.0 | 1,424.0 | 1,424.2 | 0.99988 |
| 26-30 | 52,400.0 | 1,913.0 | 1,912.3 | 1.0003 |
| 31-40 | 136,119.0 | 4,639.0 | 4,638.2 | 1.0002 |
| 41-50 | 132,149.0 | 5,354.0 | 5,353.7 | 1.0001 |
| 51-60 | 108,145.0 | 4,124.0 | 4,124.3 | 0.99992 |
| 61-70 | 50,910.0 | 1,882.0 | 1,882.2 | 0.9999 |
| 71-80 | 26,595.0 | 1,063.0 | 1,063.0 | 0.99999 |
| 81+ | 4,963.0 | 215 | 214.85 | 1.0007 |

### pure_premium by DrivAge: tweedie

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 18-20 | 5,433.0 | 1,494,457.5 | 1,574,727.4 | 0.94903 |
| 21-25 | 25,697.0 | 2,895,173.8 | 2,803,914.9 | 1.0325 |
| 26-30 | 52,400.0 | 3,510,352.2 | 3,396,323.2 | 1.0336 |
| 31-40 | 136,119.0 | 8,374,343.0 | 8,530,089.0 | 0.98174 |
| 41-50 | 132,149.0 | 9,163,140.3 | 9,389,772.6 | 0.97586 |
| 51-60 | 108,145.0 | 7,252,248.1 | 7,425,026.2 | 0.97673 |
| 61-70 | 50,910.0 | 3,414,051.1 | 3,363,571.5 | 1.015 |
| 71-80 | 26,595.0 | 2,771,993.3 | 2,741,599.0 | 1.0111 |
| 81+ | 4,963.0 | 454,902.5 | 443,528.1 | 1.0256 |

### frequency by VehAge: full

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 0-0 | 46,111.0 | 939 | 939.12 | 0.99987 |
| 1-2 | 104,302.0 | 3,649.0 | 3,648.9 | 1 |
| 3-5 | 106,190.0 | 4,268.0 | 4,267.8 | 1.0001 |
| 6-10 | 137,345.0 | 6,386.0 | 6,385.7 | 1.0001 |
| 11-15 | 107,127.0 | 4,538.0 | 4,537.7 | 1.0001 |
| 16+ | 41,336.0 | 1,343.0 | 1,343.0 | 1 |

### pure_premium by VehAge: tweedie

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 0-0 | 46,111.0 | 1,762,388.3 | 1,834,136.2 | 0.96088 |
| 1-2 | 104,302.0 | 7,142,159.0 | 7,262,385.3 | 0.98345 |
| 3-5 | 106,190.0 | 8,081,060.1 | 8,266,751.0 | 0.97754 |
| 6-10 | 137,345.0 | 11,820,214.5 | 12,068,267.2 | 0.97945 |
| 11-15 | 107,127.0 | 8,077,305.3 | 7,916,219.6 | 1.0203 |
| 16+ | 41,336.0 | 2,447,534.6 | 2,320,792.5 | 1.0546 |

### frequency by Region: full

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| R11 | 55,880.0 | 2,072.0 | 2,072.2 | 0.99989 |
| R21 | 2,458.0 | 62 | 61.848 | 1.0025 |
| R22 | 6,535.0 | 267 | 267.13 | 0.9995 |
| R23 | 7,002.0 | 174 | 173.95 | 1.0003 |
| R24 | 128,313.0 | 5,182.0 | 5,181.7 | 1.0001 |
| R25 | 8,809.0 | 376 | 376.36 | 0.99905 |
| R26 | 8,345.0 | 284 | 284.08 | 0.99972 |
| R31 | 21,634.0 | 758 | 757.85 | 1.0002 |
| R41 | 10,411.0 | 370 | 369.92 | 1.0002 |
| R42 | 1,748.0 | 72 | 74.064 | 0.97213 |
| R43 | 1,073.0 | 26 | 33.253 | 0.78188 |
| R52 | 31,019.0 | 1,273.0 | 1,272.9 | 1.0001 |
| R53 | 33,703.0 | 1,498.0 | 1,497.9 | 1.0001 |
| R54 | 15,297.0 | 624 | 623.94 | 1.0001 |
| R72 | 24,832.0 | 836 | 836 | 1 |
| R73 | 13,669.0 | 281 | 280.99 | 1 |
| R74 | 3,669.0 | 156 | 156.08 | 0.99946 |
| R82 | 67,915.0 | 3,368.0 | 3,368.3 | 0.99992 |
| R83 | 4,176.0 | 119 | 118.23 | 1.0065 |
| R91 | 28,652.0 | 848 | 847.34 | 1.0008 |
| R93 | 63,629.0 | 2,376.0 | 2,375.5 | 1.0002 |
| R94 | 3,642.0 | 101 | 92.55 | 1.0913 |

### pure_premium by Region: tweedie

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| R11 | 55,880.0 | 3,689,526.5 | 3,895,731.4 | 0.94707 |
| R21 | 2,458.0 | 197,698.1 | 126,635.2 | 1.5612 |
| R22 | 6,535.0 | 492,543.0 | 504,216.7 | 0.97685 |
| R23 | 7,002.0 | 255,906.9 | 276,472.5 | 0.92561 |
| R24 | 128,313.0 | 10,219,268.2 | 9,753,939.5 | 1.0477 |
| R25 | 8,809.0 | 778,421.9 | 854,247.3 | 0.91124 |
| R26 | 8,345.0 | 496,768.5 | 510,003.6 | 0.97405 |
| R31 | 21,634.0 | 1,288,752.7 | 1,362,970.7 | 0.94555 |
| R41 | 10,411.0 | 698,446.7 | 721,522.8 | 0.96802 |
| R42 | 1,748.0 | 86,053.6 | 144,059.5 | 0.59735 |
| R43 | 1,073.0 | 42,798.4 | 66,884.8 | 0.63988 |
| R52 | 31,019.0 | 2,013,239.9 | 1,998,566.2 | 1.0073 |
| R53 | 33,703.0 | 2,994,345.1 | 2,940,773.6 | 1.0182 |
| R54 | 15,297.0 | 1,013,352.7 | 1,018,003.6 | 0.99543 |
| R72 | 24,832.0 | 1,407,901.0 | 1,402,023.7 | 1.0042 |
| R73 | 13,669.0 | 403,554.3 | 438,995.0 | 0.91927 |
| R74 | 3,669.0 | 236,380.6 | 242,501.9 | 0.97476 |
| R82 | 67,915.0 | 6,054,109.2 | 6,012,581.2 | 1.0069 |
| R83 | 4,176.0 | 151,414.2 | 209,226.0 | 0.72369 |
| R91 | 28,652.0 | 1,563,318.9 | 1,667,734.3 | 0.93739 |
| R93 | 63,629.0 | 5,057,034.6 | 5,326,970.1 | 0.94933 |
| R94 | 3,642.0 | 189,827.0 | 194,492.3 | 0.97601 |

## Loaded premium totals (separate; reporting only)

loaded premium totals (reporting only; not part of any CV metric)

| mode | observed uncapped cost | predicted (capped Tweedie) | loading total | predicted loaded | observed / loaded |
|---|---|---|---|---|---|
| flat_per_exposure_year | 49,507,321.0 | 39,668,551.9 | 10,176,393.5 | 49,844,945.4 | 0.99323 |

Loading per exposure-year on the full training set: 35.505.

## Final models

Refitted on the full training set (Tweedie power 1.5); files: `glm_relativities_frequency.csv`, `glm_relativities_severity.csv`, `glm_relativities_tweedie.csv`, `feature_map.json`, `glm_models.json`.

## Timing (seconds)

| stage | seconds |
|---|---|
| coverage_check | 13.542 |
| cv_fits | 214.75 |
| cv_metrics | 5.3277 |
| solver_check_cv | 108.9 |
| final_fit | 25.577 |
| total | 368.96 |
| load_and_clean | 0.77785 |
