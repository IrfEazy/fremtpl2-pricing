# Model card: freMTPL2 pure-premium models

## Model details

Two families predict the motor TPL pure premium per exposure-year as frequency times severity, or
directly with a Tweedie model: GLMs (glum) and LightGBM gradient boosting. One GLM and one GBM
pure-premium candidate were selected in grouped cross-validation on the training set by the rule in
[`reports/evaluation_protocol.md`](reports/evaluation_protocol.md); their ids are named in the
evaluation table below and in `results/freeze_manifest.json`. Everything is fitted on the training
set only and frozen before the test evaluation.

## Intended use

A portfolio and methodology demonstration: comparing a GLM with gradient boosting under a
protocol written before test access, with uncertainty. **It is not a production tariff.**

## Out-of-scope uses

- Pricing, underwriting or any decision about real customers.
- Price elasticity, conversion or price optimisation (no such data).
- Large-loss modelling beyond the documented per-claim cap.
- Regulatory or fairness conclusions.
- Other markets or periods than the one in the data.

## Data

freMTPL2 from OpenML (frequency and severity tables); provenance in [`data/README.md`](data/README.md).
The data covers one historical French portfolio and no period information is given that would allow
a time-based split, so drift over time cannot be assessed. Cleaning rules, quarantined rows and the
per-claim cap are in [`reports/data_audit.md`](reports/data_audit.md). The split is grouped by
rating profile, so identical profiles never appear in both train and test.

## Evaluation

Single primary contrast and descriptive candidates, from the one-shot test evaluation
([`reports/test_evaluation.md`](reports/test_evaluation.md), protocol in
[`reports/evaluation_protocol.md`](reports/evaluation_protocol.md)). Generated from
`results/metrics.json`.

<!-- docs:metrics:begin -->
Population: the primary grouped test fold, 135 602 policies, 71737.2 exposure-years, 5 321 claims on 4 978 policies, 105 759 rating-profile groups.

Primary contrast Delta = L(`gbm_freq_x_sev`) - L(`glm_freq_x_sev[simplified]`) = -1.37922 (95% group-bootstrap CI [-2.12406, -0.77571]): **improvement**.

Gini difference (normalised, secondary) = 0.05891 (95% CI [0.02616, 0.10615]).

| Candidate | Primary loss (Tweedie 1.5) | 95% CI (group bootstrap) | Normalised Gini | Gini 95% CI | Observed / predicted |
| --- | --- | --- | --- | --- | --- |
| `pp_null` | 78.197 | [74.500, 81.958] | 0.0000 | n/a | 1.0130 |
| `bonus_malus_only` | 75.510 | [71.955, 79.230] | 0.2653 | n/a | 1.0129 |
| `glm_freq_x_sev[simplified]` | 74.567 | [70.849, 78.504] | 0.3136 | [0.2577, 0.3647] | 1.0148 |
| `glm_tweedie` | 74.415 | [70.739, 78.298] | 0.3192 | n/a | 1.0081 |
| `gbm_freq_x_sev` | 73.188 | [69.631, 76.972] | 0.3725 | [0.3271, 0.4161] | 1.0303 |
| `gbm_tweedie` | 73.504 | [69.812, 77.360] | 0.3572 | n/a | 1.0731 |
<!-- docs:metrics:end -->

## Limits and caveats

The intervals are conditional on the frozen models and do not include retraining or selection
variability. Calibration totals, the grid edges of the tuning and the effect of the cap are
reported in the limits section of [`reports/model_report.md`](reports/model_report.md). Differences
between the other candidates are descriptive only.

## Ethical considerations

The data has no protected attributes. Rating factors such as Region, Density and VehBrand may act as
proxies for protected characteristics, and no fairness or regulatory analysis was carried out beyond
this note. Do not use these models for decisions about people.

## Reproducibility

Dependencies are locked with `uv`, seeds and settings are in `configs/`, and `make all` rebuilds the
pre-test artifacts and verifies them against the published freeze. `make reproduce` rebuilds the
pre-test pipeline in a fresh clone and compares its outputs with the committed ones
([`results/reproduction/`](results/reproduction/)). The comparison is strict, except for the
deviations declared by a dated amendment of the verification policy in
[`configs/reproduce.yaml`](configs/reproduce.yaml); each deviation is pinned to specific fields or
values and verified from the files on every run. The latest check fails the strict policy and passes
under the amendment; the deviations and both verdicts are listed in the reproduction block of the
README. Held-out metrics are never recomputed: the published test report is re-rendered and its
files are verified against their recorded hashes.

## Contact

The repository owner, through the repository. The code licence is in [`LICENSE`](LICENSE).
