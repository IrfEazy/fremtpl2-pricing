.PHONY: setup lint test download audit split train sensitivities gbm freeze rehearsal evaluate all

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

# Train only; needs a clean committed work tree. Commit results/freeze_manifest.json afterwards.
freeze:
	uv run fremtpl2 freeze

# The evaluate code path on the train fold freeze.rehearsal.fold (no test row, no results/ write).
rehearsal:
	uv run fremtpl2 evaluate --rehearsal

# The single, one-shot test evaluation (protocol section 6). Never part of `all`.
evaluate:
	uv run fremtpl2 evaluate

all: download audit split train gbm sensitivities freeze
