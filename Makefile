.PHONY: setup lint test download audit split train all

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

all: download audit split train
