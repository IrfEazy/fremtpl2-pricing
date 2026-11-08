.PHONY: setup lint test download audit split train sensitivities gbm interpret freeze freeze-check reproduce rehearsal evaluate report docs all

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

download:
	uv run fremtpl2 download

audit:
	uv run fremtpl2 audit

split:
	uv run fremtpl2 split

train:
	uv run fremtpl2 train

sensitivities:
	uv run fremtpl2 sensitivities

gbm:
	uv run fremtpl2 gbm

# Train-only interpretation of the frozen candidates (SHAP vs GLM, BonusMalus monotonicity, constraint
# register): results/interpretation.json, reports/interpretation.md, reports/figures/interp_*.png. Never
# fits, never reads a test row; commit its outputs before `freeze` (which needs a clean tree).
interpret:
	uv run fremtpl2 interpret

# Train only; needs a clean committed work tree. Commit results/freeze_manifest.json afterwards.
freeze:
	uv run fremtpl2 freeze

# The evaluate code path on the train fold freeze.rehearsal.fold (no test row, no results/ write).
rehearsal:
	uv run fremtpl2 evaluate --rehearsal

# The single, one-shot test evaluation (protocol section 6). Never part of `all`.
evaluate:
	uv run fremtpl2 evaluate

# Renders reports/test_evaluation.md and figures only from results/metrics.json and the completed
# attempt it points to; exits non-zero (writing nothing) if no completed attempt exists. Explicit,
# never part of `all`.
report:
	uv run fremtpl2 report

# Renders reports/model_report.md and the generated blocks of README.md / MODEL_CARD.md only from the
# tracked pipeline files (settings: configs/docs.yaml); `uv run fremtpl2 docs --check` writes nothing and
# exits non-zero on drift. Explicit, after `report`, never part of `all`.
docs:
	uv run fremtpl2 docs

# Strict, read-only verification of the committed freeze manifest against the regenerated artifacts
# (writes nothing, needs no clean tree; usable after test access). Settings: configs/reproduce.yaml `check:`.
freeze-check:
	uv run fremtpl2 freeze --check

# Rebuilds the pre-test pipeline in an isolated clean clone of HEAD and compares it with the committed
# baseline. Appends an immutable record results/reproduction/run-<n>.json (the run history) with a strict
# verdict and a post-hoc amended verdict; the exit code is 0 only for an exact pass or the amended pass
# (a PORTABLE_NUMERIC_PASS or any failure exits non-zero, but the record is still written). Never recomputes held-out metrics. Settings: configs/reproduce.yaml.
reproduce:
	uv run fremtpl2 reproduce

# After the test access (results/test_access.json exists) `freeze` is refused, so the strict check
# replaces it; `report` runs only when a completed attempt exists (results/metrics.json).
all: download audit split train gbm sensitivities interpret
	@if [ -f results/test_access.json ]; then $(MAKE) freeze-check; else $(MAKE) freeze; fi
	@if [ -f results/metrics.json ]; then $(MAKE) report; fi
