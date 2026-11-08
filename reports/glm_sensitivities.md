# GLM cleaning/target sensitivities (grouped CV, train set only)

Rendered from `results/cv_sensitivities.json` by `fremtpl2 sensitivities`; no hand-typed numbers. The test set is untouched: sensitivities are CV-only by protocol (`reports/evaluation_protocol.md` section 7); the test evaluation uses the primary definitions only.

Outer registered folds only (5, grouped split, frozen); Tweedie power fixed at 1.5 (final power of `results/cv_glm.json`), no nested selection, no final fit. Policies a sensitivity drops simply drop out; the split assignment of the others never changes.

**Comparisons are made within each target definition only. Losses of different definitions (different targets, caps, exposures or policy sets) are not comparable and are not ranked.**

## Row counts (train)

| sensitivity | overrides | policies | exposure | claims (N) | frequency events | S_cap |
|---|---|---|---|---|---|---|
| primary | none (primary rules) | 542,411 | 286,622.9 | 21,123 | 21,123.0 | 39,330,661.8 |
| exclude_high_count | high_count: exclude_n_above: 5 | 542,409 | 286,622.7 | 21,101 | 21,101.0 | 39,287,389.6 |
| frequency_recorded_claim_nb | targets: frequency_count: claim_nb | 542,411 | 286,622.9 | 21,123 | 28,811.0 | 39,330,661.8 |
| cap_50000 | large_loss: cap: 50000 | 542,411 | 286,622.9 | 21,123 | 21,123.0 | 36,890,541.8 |
| cap_200000 | large_loss: cap: 200000 | 542,411 | 286,622.9 | 21,123 | 21,123.0 | 41,876,059.7 |
| cap_uncapped | large_loss: cap: None | 542,411 | 286,622.9 | 21,123 | 21,123.0 | 49,507,321.0 |
| raw_exposure | exposure_cap: None | 542,411 | 286,735.8 | 21,123 | 21,123.0 | 39,330,661.8 |
| exclude_short_records_1day | exposure_min: 0.00274 | 539,920 | 286,616.1 | 21,120 | 21,120.0 | 39,327,113.8 |
| exclude_short_records_0_05 | exposure_min: 0.05 | 508,560 | 285,808.9 | 20,841 | 20,841.0 | 38,307,079.9 |

## Convergence

| sensitivity | fits | not converged | max iterations |
|---|---|---|---|
| primary | 20 | 0 | 6 |
| exclude_high_count | 20 | 0 | 6 |
| frequency_recorded_claim_nb | 20 | 0 | 6 |
| cap_50000 | 20 | 0 | 6 |
| cap_200000 | 20 | 0 | 6 |
| cap_uncapped | 20 | 0 | 6 |
| raw_exposure | 20 | 0 | 6 |
| exclude_short_records_1day | 20 | 0 | 6 |
| exclude_short_records_0_05 | 20 | 0 | 6 |

## Frequency (full GLM vs null, within definition)

| sensitivity | deviance / exposure-year (null → full, pooled) | explained pooled | explained fold mean ± sd | Gini normalised pooled |
|---|---|---|---|---|
| primary | 0.47843 → 0.45726 | 4.43% | 4.43% ± 0.18% | 0.29404 |
| exclude_high_count | 0.47742 → 0.45616 | 4.45% | 4.45% ± 0.20% | 0.295 |
| frequency_recorded_claim_nb | 0.62438 → 0.593 | 5.03% | 5.03% ± 0.21% | 0.29131 |
| cap_50000 | 0.47843 → 0.45726 | 4.43% | 4.43% ± 0.18% | 0.29404 |
| cap_200000 | 0.47843 → 0.45726 | 4.43% | 4.43% ± 0.18% | 0.29404 |
| cap_uncapped | 0.47843 → 0.45726 | 4.43% | 4.43% ± 0.18% | 0.29404 |
| raw_exposure | 0.47826 → 0.4571 | 4.43% | 4.43% ± 0.18% | 0.29397 |
| exclude_short_records_1day | 0.47828 → 0.45711 | 4.43% | 4.43% ± 0.18% | 0.29403 |
| exclude_short_records_0_05 | 0.46841 → 0.44742 | 4.48% | 4.48% ± 0.16% | 0.29464 |

## Severity (full GLM vs null, claims only, within definition)

| sensitivity | Gamma deviance / claim (null → full, pooled) | explained pooled | explained fold mean ± sd |
|---|---|---|---|
| primary | 1.3152 → 1.3198 | -0.35% | -0.35% ± 0.97% |
| exclude_high_count | 1.3165 → 1.3211 | -0.35% | -0.35% ± 0.96% |
| frequency_recorded_claim_nb | 1.3152 → 1.3198 | -0.35% | -0.35% ± 0.97% |
| cap_50000 | 1.1905 → 1.1911 | -0.05% | -0.06% ± 0.62% |
| cap_200000 | 1.4391 → 1.4455 | -0.44% | -0.45% ± 1.05% |
| cap_uncapped | 1.7921 → 1.6332 | 8.87% | 7.04% ± 10.10% |
| raw_exposure | 1.3152 → 1.3198 | -0.35% | -0.35% ± 0.97% |
| exclude_short_records_1day | 1.3153 → 1.32 | -0.35% | -0.35% ± 0.98% |
| exclude_short_records_0_05 | 1.3001 → 1.3051 | -0.39% | -0.37% ± 0.94% |

## Pure premium (common Tweedie deviance at reference power 1.5, within definition)

The `tweedie` model is fitted at the fixed power 1.5; the deviance is always evaluated at the reference power.

### primary

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 77.75 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.023003 | 39,330,661.8 | 39,330,749.5 | 1 |
| bonus_malus_only | 74.986 | 3.56% | 3.55% ± 0.57% | 3.10% | 0.25253 | 39,330,661.8 | 39,330,870.3 | 0.99999 |
| freq_x_sev | 74.044 | 4.77% | 4.75% ± 0.89% | 4.03% | 0.31013 | 39,330,661.8 | 39,324,494.2 | 1.0002 |
| tweedie | 74.097 | 4.70% | 4.68% ± 0.87% | 3.90% | 0.30953 | 39,330,661.8 | 39,668,551.9 | 0.99148 |

### exclude_high_count

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 77.701 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.022971 | 39,287,389.6 | 39,287,485.6 | 1 |
| bonus_malus_only | 74.927 | 3.57% | 3.56% ± 0.57% | 3.12% | 0.25426 | 39,287,389.6 | 39,287,603.5 | 0.99999 |
| freq_x_sev | 73.967 | 4.81% | 4.79% ± 0.87% | 4.06% | 0.3115 | 39,287,389.6 | 39,281,545.3 | 1.0001 |
| tweedie | 74.018 | 4.74% | 4.72% ± 0.85% | 3.93% | 0.311 | 39,287,389.6 | 39,628,350.2 | 0.9914 |

### frequency_recorded_claim_nb

Frequency x severity not computed: frequency count (recorded ClaimNb) differs from the severity denominator (matched severity rows): frequency x severity does not estimate the capped pure premium, so it and its comparison with the Tweedie are not computed; frequency metrics are compared only with their own null.

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 78.88 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.024937 | 39,330,661.8 | 53,646,113.8 | 0.73315 |
| bonus_malus_only | 76.475 | 3.05% | 3.04% ± 0.43% | 2.82% | 0.25118 | 39,330,661.8 | 53,646,279.5 | 0.73315 |
| tweedie | 74.097 | 6.06% | 6.06% ± 0.44% | 5.07% | 0.30953 | 39,330,661.8 | 39,668,551.9 | 0.99148 |

### cap_50000

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 74.878 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.017388 | 36,890,541.8 | 36,890,636.1 | 1 |
| bonus_malus_only | 72.272 | 3.48% | 3.48% ± 0.35% | 3.16% | 0.24945 | 36,890,541.8 | 36,890,734.0 | 0.99999 |
| freq_x_sev | 71.24 | 4.86% | 4.85% ± 0.55% | 4.25% | 0.31436 | 36,890,541.8 | 36,899,208.7 | 0.99977 |
| tweedie | 71.264 | 4.83% | 4.82% ± 0.55% | 4.15% | 0.31533 | 36,890,541.8 | 37,193,590.0 | 0.99185 |

### cap_200000

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 80.68 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.028974 | 41,876,059.7 | 41,876,149.6 | 1 |
| bonus_malus_only | 77.723 | 3.67% | 3.65% ± 0.74% | 3.08% | 0.25311 | 41,876,059.7 | 41,876,297.7 | 0.99999 |
| freq_x_sev | 76.891 | 4.70% | 4.67% ± 1.23% | 3.87% | 0.30306 | 41,876,059.7 | 41,814,203.6 | 1.0015 |
| tweedie | 77.027 | 4.53% | 4.50% ± 1.15% | 3.68% | 0.29905 | 41,876,059.7 | 42,218,869.1 | 0.99188 |

### cap_uncapped

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 89.295 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.10138 | 49,507,321.0 | 49,507,143.0 | 1 |
| bonus_malus_only | 83.804 | 6.15% | 5.83% ± 3.32% | 4.46% | 0.33477 | 49,507,321.0 | 49,507,015.2 | 1 |
| freq_x_sev | 81.008 | 9.28% | 8.66% ± 6.39% | 7.66% | 0.37942 | 49,507,321.0 | 48,389,805.3 | 1.0231 |
| tweedie | 81.309 | 8.94% | 8.33% ± 6.30% | 7.29% | 0.37324 | 49,507,321.0 | 49,354,584.5 | 1.0031 |

### raw_exposure

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 77.736 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.022861 | 39,330,661.8 | 39,330,734.9 | 1 |
| bonus_malus_only | 74.971 | 3.56% | 3.55% ± 0.57% | 3.10% | 0.25283 | 39,330,661.8 | 39,330,858.9 | 0.99999 |
| freq_x_sev | 74.031 | 4.77% | 4.75% ± 0.89% | 4.03% | 0.31008 | 39,330,661.8 | 39,324,691.1 | 1.0002 |
| tweedie | 74.084 | 4.70% | 4.68% ± 0.87% | 3.90% | 0.30948 | 39,330,661.8 | 39,668,551.4 | 0.99148 |

### exclude_short_records_1day

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 77.747 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.021713 | 39,327,113.8 | 39,327,202.7 | 1 |
| bonus_malus_only | 74.983 | 3.56% | 3.54% ± 0.57% | 3.10% | 0.25434 | 39,327,113.8 | 39,327,329.6 | 0.99999 |
| freq_x_sev | 74.041 | 4.77% | 4.75% ± 0.89% | 4.03% | 0.31012 | 39,327,113.8 | 39,320,805.5 | 1.0002 |
| tweedie | 74.094 | 4.70% | 4.68% ± 0.87% | 3.90% | 0.30951 | 39,327,113.8 | 39,664,788.5 | 0.99149 |

### exclude_short_records_0_05

| model | Tweedie dev. / exposure-year | explained pooled | explained fold mean ± sd | Poisson explained pooled | Gini normalised | observed S_cap | predicted | obs / pred |
|---|---|---|---|---|---|---|---|---|
| null | 76.663 | 0.00% | 0.00% ± 0.00% | 0.00% | -0.025023 | 38,307,079.9 | 38,307,017.0 | 1 |
| bonus_malus_only | 74.042 | 3.42% | 3.41% ± 0.55% | 3.04% | 0.24429 | 38,307,079.9 | 38,307,102.5 | 1 |
| freq_x_sev | 73.068 | 4.69% | 4.67% ± 0.87% | 4.01% | 0.30804 | 38,307,079.9 | 38,298,164.6 | 1.0002 |
| tweedie | 73.124 | 4.62% | 4.60% ± 0.86% | 3.88% | 0.30704 | 38,307,079.9 | 38,640,130.6 | 0.99138 |

## Differences inside each definition

Tweedie minus frequency x severity (negative deviance difference favours the Tweedie), pooled and fold mean ± sd of the per-fold differences.

| sensitivity | Tweedie - freq x sev: deviance pooled | fold mean ± sd | Gini norm. pooled | frequency full - null: deviance pooled | fold mean ± sd | severity full - null: deviance pooled | fold mean ± sd |
|---|---|---|---|---|---|---|---|
| primary | 0.053247 | 0.053294 ± 0.064468 | -0.00060001 | -0.021174 | -0.021174 ± 0.00081616 | 0.0046573 | 0.0046171 ± 0.013075 |
| exclude_high_count | 0.050376 | 0.050423 ± 0.062753 | -0.00050215 | -0.021261 | -0.021261 ± 0.0008952 | 0.0046315 | 0.0046047 ± 0.012961 |
| frequency_recorded_claim_nb | not computed | not computed | not computed | -0.031376 | -0.031378 ± 0.0012885 | 0.0046573 | 0.0046171 ± 0.013075 |
| cap_50000 | 0.0238 | 0.023842 ± 0.04683 | 0.00096919 | -0.021174 | -0.021174 ± 0.00081616 | 0.00058782 | 0.00057341 ± 0.0072974 |
| cap_200000 | 0.1358 | 0.13586 ± 0.12063 | -0.0040111 | -0.021174 | -0.021174 ± 0.00081616 | 0.0063377 | 0.0063013 ± 0.015993 |
| cap_uncapped | 0.30143 | 0.3015 ± 0.17292 | -0.0061739 | -0.021174 | -0.021174 ± 0.00081616 | -0.15893 | -0.16003 ± 0.24652 |
| raw_exposure | 0.053275 | 0.05332 ± 0.064529 | -0.00060107 | -0.021164 | -0.021164 ± 0.00081687 | 0.0046573 | 0.0046171 ± 0.013075 |
| exclude_short_records_1day | 0.053326 | 0.053373 ± 0.064596 | -0.00060147 | -0.021171 | -0.021171 ± 0.00081473 | 0.0046276 | 0.0045862 ± 0.013102 |
| exclude_short_records_0_05 | 0.055481 | 0.055502 ± 0.054825 | -0.0010019 | -0.020994 | -0.020994 ± 0.00074187 | 0.0050289 | 0.0050182 ± 0.012336 |


`frequency_recorded_claim_nb`: frequency count (recorded ClaimNb) differs from the severity denominator (matched severity rows): frequency x severity does not estimate the capped pure premium, so it and its comparison with the Tweedie are not computed; frequency metrics are compared only with their own null.

## Bootstrap unit diagnostic (train only, primary definition)

diagnostic only (train, primary definition); the primary unit is unchanged.

Paired bootstrap of the primary out-of-fold predictions; loss: Tweedie deviance per exposure-year at power 1.5; 1000 replicates, seed 20260102, 95.00% percentile intervals. The rating-profile groups are resampled because identical profiles may belong to the same policyholder (renewals, duplicates), not merely because they share covariates.

| quantity | unit | units | point | SE | lower | upper | width |
|---|---|---|---|---|---|---|---|
| freq_x_sev | group | 423,006 | 74.044 | 0.96822 | 72.215 | 76.041 | 3.826 |
| freq_x_sev | policy | 542,411 | 74.044 | 0.96025 | 72.221 | 75.959 | 3.7378 |
| tweedie | group | 423,006 | 74.097 | 0.9818 | 72.26 | 76.135 | 3.8749 |
| tweedie | policy | 542,411 | 74.097 | 0.97135 | 72.227 | 75.997 | 3.7699 |
| tweedie_minus_freq_x_sev | group | 423,006 | 0.053247 | 0.036219 | -0.014424 | 0.12615 | 0.14057 |
| tweedie_minus_freq_x_sev | policy | 542,411 | 0.053247 | 0.03474 | -0.013678 | 0.12368 | 0.13736 |

| quantity | SE ratio group / policy | interval width ratio group / policy |
|---|---|---|
| freq_x_sev | 1.0083 | 1.0236 |
| tweedie | 1.0108 | 1.0279 |
| tweedie_minus_freq_x_sev | 1.0426 | 1.0234 |

## Timing (seconds)

| stage | seconds |
|---|---|
| primary | 24.769 |
| exclude_high_count | 12.699 |
| frequency_recorded_claim_nb | 13.108 |
| cap_50000 | 13.278 |
| cap_200000 | 13.854 |
| cap_uncapped | 14.576 |
| raw_exposure | 13.401 |
| exclude_short_records_1day | 13.69 |
| exclude_short_records_0_05 | 12.54 |
| total | 132.15 |
