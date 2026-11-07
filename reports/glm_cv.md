# GLM cross-validation (milestone 2)

Rendered from `cv_glm.json` by `fremtpl2 train`; every number below comes from that file.

## Protocol

- Test set used: **False**. All fits, preprocessing and metrics use the primary training set only (registered `cv_fold`); the test set is untouched in this milestone.
- Split: `grouped`, 5 outer folds; features, rare-level grouping and every model are fitted on the other folds of each outer fold.
- Tweedie power: grid [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], chosen per outer fold by an inner CV over the remaining folds with the `tweedie_loglik` loss: the held-out Tweedie log-likelihood of S_cap / Exposure with prior weight = exposure, dispersion estimated by maximum likelihood on the fitting part only (Dunn & Smyth 2005 series density). The exposure-weighted Poisson deviance per power is kept as a labelled diagnostic. Common Tweedie deviance at the fixed reference power 1.5, which also enters the pure premium comparison as the fixed-power sensitivity `tweedie_ref`.
- Targets: frequency count `sev_rows`, per-claim cap 100,000; the excess loading (`flat_per_exposure_year`) is in no CV target or prediction.
- Bootstrap confidence intervals are not part of this milestone (applied: False); they come in milestone 3.
- Config hash `65b0ba08bf46d1d0d9188890a7ae39fdd457c4a439a5742fbd73a43001099d5d`, seed 20260101.

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

| fold | selected | at grid edge | loglik spread (nats) | weakly identified |
|---|---|---|---|---|
| 0 | 1.5 | False | 53,623.5 | False |
| 1 | 1.5 | False | 49,535.0 | False |
| 2 | 1.5 | False | 50,323.2 | False |
| 3 | 1.5 | False | 55,320.5 | False |
| 4 | 1.5 | False | 55,542.2 | False |

Inner-CV held-out Tweedie log-likelihood per power (total nats; higher is better).

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -265,626.2 | -228,587.0 | -217,691.5 | -213,408.2 | -212,002.7 | -212,444.7 | -214,477.8 | -218,544.9 | -226,953.1 |
| 1 | -261,677.6 | -228,691.6 | -218,164.2 | -213,671.0 | -212,142.7 | -212,505.2 | -214,487.0 | -218,520.9 | -226,907.2 |
| 2 | -261,926.7 | -228,054.4 | -217,502.9 | -213,083.8 | -211,603.5 | -211,994.6 | -213,989.5 | -218,023.5 | -226,392.7 |
| 3 | -267,245.7 | -228,991.4 | -217,827.5 | -213,405.2 | -211,925.1 | -212,325.6 | -214,338.6 | -218,401.2 | -226,817.2 |
| 4 | -267,570.0 | -229,072.3 | -217,812.0 | -213,460.6 | -212,027.8 | -212,455.4 | -214,483.6 | -218,554.8 | -226,977.8 |

Mean maximum-likelihood dispersion over the inner splits.

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,500.0 | 916.95 | 594.31 | 423.55 | 312.13 | 239.04 | 195.52 | 180.3 | 222.12 |
| 1 | 1,319.2 | 828.64 | 592.35 | 428.26 | 314.88 | 240.61 | 196.43 | 180.81 | 222.43 |
| 2 | 1,363.5 | 868.12 | 590.71 | 427.32 | 314.52 | 240.6 | 196.65 | 181.23 | 223.16 |
| 3 | 1,530.7 | 928.57 | 597.82 | 424.83 | 312.66 | 239.23 | 195.56 | 180.28 | 222.08 |
| 4 | 1,541.9 | 939.78 | 603.58 | 422.75 | 311.53 | 238.59 | 195.2 | 180.04 | 221.87 |

Diagnostic only (not the selection criterion): inner-CV exposure-weighted Poisson deviance of S_cap / Exposure per exposure-year. It is a mean-prediction loss and does not identify the variance power.

| fold | p=1.1 | p=1.2 | p=1.3 | p=1.4 | p=1.5 | p=1.6 | p=1.7 | p=1.8 | p=1.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,137.2 | 1,137.4 | 1,137.6 | 1,137.9 | 1,138.4 | 1,138.9 | 1,139.6 | 1,140.5 | 1,141.6 |
| 1 | 1,166.7 | 1,166.8 | 1,167.0 | 1,167.3 | 1,167.8 | 1,168.4 | 1,169.2 | 1,170.2 | 1,171.4 |
| 2 | 1,153.1 | 1,153.3 | 1,153.6 | 1,153.9 | 1,154.5 | 1,155.1 | 1,156.0 | 1,157.1 | 1,158.5 |
| 3 | 1,151.2 | 1,151.3 | 1,151.5 | 1,151.7 | 1,152.1 | 1,152.7 | 1,153.4 | 1,154.4 | 1,155.7 |
| 4 | 1,134.2 | 1,134.2 | 1,134.4 | 1,134.6 | 1,135.0 | 1,135.5 | 1,136.1 | 1,137.0 | 1,138.1 |

Final power (plain CV over the grid on the registered folds): selected 1.5, interior to the grid; held-out log-likelihood spread over the grid 67,346.7 nats, best minus runner-up 513.04 nats; the power is identified by the likelihood (spread above the threshold).

| power | CV loglik (nats) | CV loglik per policy | diagnostic: Poisson deviance |
|---|---|---|---|
| 1.1 | -332,233.8 | -0.61251 | 1,147.4 |
| 1.2 | -285,867.0 | -0.52703 | 1,147.5 |
| 1.3 | -272,189.9 | -0.50181 | 1,147.7 |
| 1.4 | -266,711.0 | -0.49171 | 1,148.0 |
| 1.5 | -264,887.1 | -0.48835 | 1,148.4 |
| 1.6 | -265,400.1 | -0.4893 | 1,148.9 |
| 1.7 | -267,920.0 | -0.49394 | 1,149.6 |
| 1.8 | -272,994.4 | -0.5033 | 1,150.4 |
| 1.9 | -283,503.1 | -0.52267 | 1,151.5 |

Per outer fold: 0 of 5 selections at a grid edge, 0 weakly identified (threshold 10 nats).

### Caveat: point mass at the cap

The per-claim cap makes S_cap a censored amount: policies whose S_cap sits exactly at a positive multiple of the cap carry a point mass that the continuous Tweedie density does not model. The likelihood is therefore a working likelihood, adequate for comparing powers but not an exact model of the capped response.

| cap | policies with claim cost | S_cap at a cap multiple | share | with a capped claim |
|---|---|---|---|---|
| 100,000.0 | 19,966 | 31 | 0.16% | 36 |

## Solver convergence

glum solver settings: max_iter 1000, gradient_tol 1e-05. Fits: 244; **not converged: 0**; maximum iterations used 65.

## Findings

- Severity full GLM, out-of-fold deviance explained versus the null: fold mean -0.40% (sd 0.99%), pooled -0.40%; negative in 3 of 5 folds. The full severity model predicts claims out of fold worse than the constant mean: the rating factors carry no usable severity signal at this sample size.
- Tweedie power: selected 1.5, interior to the grid; held-out log-likelihood spread over the grid 67,346.7 nats, best minus runner-up 513.04 nats; the power is identified by the likelihood (spread above the threshold).

## Scalar metrics (fold mean ± sd, and pooled out-of-fold)

### Frequency

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | deviance_per_exposure | 0.47843 ± 0.0028362 | 0.47843 |
| null | deviance_explained | 0 ± 0 | 0 |
| null | gini | 0.0005088 ± 0.0011558 | -0.0038649 |
| null | gini_normalized | 0.00052706 ± 0.0011972 | -0.0040049 |
| bonus_malus_only | deviance_per_exposure | 0.46279 ± 0.00279 | 0.46279 |
| bonus_malus_only | deviance_explained | 0.032688 ± 0.00097667 | 0.032687 |
| bonus_malus_only | gini | 0.22957 ± 0.0059458 | 0.22832 |
| bonus_malus_only | gini_normalized | 0.23788 ± 0.0061084 | 0.23659 |
| full | deviance_per_exposure | 0.45726 ± 0.0030498 | 0.45726 |
| full | deviance_explained | 0.04426 ± 0.0017634 | 0.044258 |
| full | gini | 0.28388 ± 0.0074022 | 0.28376 |
| full | gini_normalized | 0.29416 ± 0.0075422 | 0.29404 |

### Severity (claims only, weight N)

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | deviance_per_claim | 1.3154 ± 0.069139 | 1.3152 |
| null | deviance_explained | 0 ± 0 | 0 |
| full | deviance_per_claim | 1.3206 ± 0.069035 | 1.3204 |
| full | deviance_explained | -0.0039909 ± 0.0099311 | -0.0039845 |

### Pure premium

| model | metric | fold mean ± sd | pooled |
|---|---|---|---|
| null | poisson_deviance_per_exposure | 1,195.0 ± 62.38 | 1,195.0 |
| null | poisson_deviance_explained | 0 ± 0 | 0 |
| null | tweedie_ref_deviance_per_exposure | 77.75 ± 1.9453 | 77.75 |
| null | tweedie_ref_deviance_explained | 0 ± 0 | 0 |
| null | gini | -0.00056204 ± 0.0012568 | -0.022612 |
| null | gini_normalized | -0.00057147 ± 0.0012779 | -0.023003 |
| bonus_malus_only | poisson_deviance_per_exposure | 1,158.0 ± 55.343 | 1,158.0 |
| bonus_malus_only | poisson_deviance_explained | 0.030836 ± 0.0050173 | 0.03102 |
| bonus_malus_only | tweedie_ref_deviance_per_exposure | 74.985 ± 1.492 | 74.986 |
| bonus_malus_only | tweedie_ref_deviance_explained | 0.035459 ± 0.0057065 | 0.035559 |
| bonus_malus_only | gini | 0.2553 ± 0.022907 | 0.25012 |
| bonus_malus_only | gini_normalized | 0.25971 ± 0.023239 | 0.25444 |
| freq_x_sev | poisson_deviance_per_exposure | 1,147.0 ± 51.018 | 1,147.0 |
| freq_x_sev | poisson_deviance_explained | 0.039887 ± 0.0076577 | 0.0402 |
| freq_x_sev | tweedie_ref_deviance_per_exposure | 74.049 ± 1.2247 | 74.049 |
| freq_x_sev | tweedie_ref_deviance_explained | 0.047433 ± 0.0089243 | 0.0476 |
| freq_x_sev | gini | 0.304 ± 0.033416 | 0.30448 |
| freq_x_sev | gini_normalized | 0.30925 ± 0.033866 | 0.30974 |
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

| model | observed total (S_cap) | predicted total | observed / predicted |
|---|---|---|---|
| null | 39,330,661.8 | 39,330,749.5 | 1 |
| bonus_malus_only | 39,330,661.8 | 39,331,753.7 | 0.99997 |
| freq_x_sev | 39,330,661.8 | 39,324,477.1 | 1.0002 |
| tweedie | 39,330,661.8 | 39,668,551.9 | 0.99148 |
| tweedie_ref | 39,330,661.8 | 39,668,551.9 | 0.99148 |

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
| 0 | 28,662.0 | 2,618,272.7 | 2,941,694.5 | 0.89006 |
| 1 | 28,662.6 | 2,753,189.0 | 2,944,036.3 | 0.93517 |
| 2 | 28,662.1 | 2,687,381.7 | 2,960,526.9 | 0.90774 |
| 3 | 28,662.2 | 2,558,159.6 | 2,977,194.8 | 0.85925 |
| 4 | 28,662.4 | 2,693,039.8 | 2,977,629.5 | 0.90442 |
| 5 | 28,662.0 | 2,591,306.1 | 3,023,262.5 | 0.85712 |
| 6 | 28,663.0 | 3,561,173.5 | 3,155,472.4 | 1.1286 |
| 7 | 28,662.2 | 4,940,091.8 | 3,846,474.5 | 1.2843 |
| 8 | 28,662.4 | 5,481,131.8 | 5,242,284.0 | 1.0456 |
| 9 | 28,662.0 | 9,446,915.7 | 9,263,178.3 | 1.0198 |

### Pure premium: freq_x_sev

| bin | exposure | observed | predicted | ratio |
|---|---|---|---|---|
| 0 | 28,662.2 | 1,730,787.4 | 1,636,135.8 | 1.0579 |
| 1 | 28,662.7 | 2,033,721.9 | 2,111,344.1 | 0.96324 |
| 2 | 28,661.9 | 2,542,189.9 | 2,415,627.9 | 1.0524 |
| 3 | 28,662.4 | 2,421,215.0 | 2,689,334.8 | 0.9003 |
| 4 | 28,662.3 | 3,193,997.9 | 2,968,525.5 | 1.076 |
| 5 | 28,662.2 | 2,966,081.6 | 3,277,393.4 | 0.90501 |
| 6 | 28,662.4 | 3,610,086.9 | 3,666,580.3 | 0.98459 |
| 7 | 28,662.3 | 5,040,698.1 | 4,243,066.2 | 1.188 |
| 8 | 28,662.0 | 5,295,681.7 | 5,348,558.7 | 0.99011 |
| 9 | 28,662.5 | 10,496,201.6 | 10,967,910.4 | 0.95699 |

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

## Double lift: Tweedie vs frequency x severity

`pred_a` is the Tweedie, `pred_b` the frequency x severity prediction.

| bin | exposure | observed | pred_a | pred_b | ratio_a | ratio_b |
|---|---|---|---|---|---|---|
| 0 | 28,662.0 | 3,106,811.1 | 2,654,284.9 | 2,976,661.1 | 1.1705 | 1.0437 |
| 1 | 28,663.0 | 2,856,497.5 | 2,821,307.2 | 3,033,612.8 | 1.0125 | 0.94162 |
| 2 | 28,661.6 | 3,102,052.1 | 2,981,980.5 | 3,146,458.6 | 1.0403 | 0.98589 |
| 3 | 28,662.7 | 3,303,675.0 | 3,187,896.7 | 3,313,177.7 | 1.0363 | 0.99713 |
| 4 | 28,661.9 | 3,241,611.0 | 3,369,015.0 | 3,452,546.7 | 0.96218 | 0.9389 |
| 5 | 28,662.4 | 3,607,702.2 | 3,620,714.4 | 3,656,828.6 | 0.99641 | 0.98657 |
| 6 | 28,662.6 | 4,202,411.5 | 3,877,878.0 | 3,853,733.0 | 1.0837 | 1.0905 |
| 7 | 28,662.0 | 4,920,648.0 | 4,260,670.1 | 4,153,640.6 | 1.1549 | 1.1847 |
| 8 | 28,662.2 | 4,727,350.2 | 4,865,336.6 | 4,623,603.7 | 0.97164 | 1.0224 |
| 9 | 28,662.5 | 6,261,903.2 | 8,029,468.4 | 7,114,214.4 | 0.77987 | 0.8802 |

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
| 0 | 28,661.9 | 1,504.0 | 1,589.9 | 0.94599 |
| 1 | 28,662.6 | 1,457.0 | 1,594.8 | 0.91359 |
| 2 | 28,662.3 | 1,505.0 | 1,596.9 | 0.94243 |
| 3 | 28,662.8 | 1,482.0 | 1,597.5 | 0.9277 |
| 4 | 28,662.0 | 1,402.0 | 1,597.5 | 0.87763 |
| 5 | 28,662.2 | 1,466.0 | 1,602.7 | 0.91472 |
| 6 | 28,662.5 | 1,837.0 | 1,688.4 | 1.088 |
| 7 | 28,661.8 | 2,931.0 | 2,066.8 | 1.4182 |
| 8 | 28,662.4 | 2,972.0 | 2,816.7 | 1.0551 |
| 9 | 28,662.4 | 4,567.0 | 4,972.4 | 0.91848 |

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
| 8 | 28,661.8 | 3,209.0 | 2,817.6 | 1.1389 |
| 9 | 28,662.4 | 5,044.0 | 5,283.5 | 0.95467 |

### frequency by BonusMalus: full

| level | weight | observed | predicted | ratio |
|---|---|---|---|---|
| 50-59 | 369,106.0 | 11,653.0 | 12,203.6 | 0.95488 |
| 60-69 | 58,591.0 | 2,873.0 | 2,082.3 | 1.3797 |
| 70-79 | 35,758.0 | 1,730.0 | 1,528.1 | 1.1321 |
| 80-99 | 57,026.0 | 2,784.0 | 3,326.7 | 0.83688 |
| 100-124 | 20,129.0 | 1,709.0 | 1,571.7 | 1.0873 |
| 125+ | 1,801.0 | 374 | 409.69 | 0.91289 |

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
| 18-20 | 5,433.0 | 509 | 509.27 | 0.99946 |
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
| 6-10 | 137,345.0 | 6,386.0 | 6,385.7 | 1 |
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
| R42 | 1,748.0 | 72 | 74.065 | 0.97212 |
| R43 | 1,073.0 | 26 | 33.253 | 0.78187 |
| R52 | 31,019.0 | 1,273.0 | 1,272.9 | 1.0001 |
| R53 | 33,703.0 | 1,498.0 | 1,497.9 | 1.0001 |
| R54 | 15,297.0 | 624 | 623.95 | 1.0001 |
| R72 | 24,832.0 | 836 | 836.01 | 0.99999 |
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
| coverage_check | 16.862 |
| cv_fits | 283.57 |
| cv_metrics | 5.9074 |
| final_fit | 15.433 |
| total | 323.3 |
| load_and_clean | 1.5183 |
