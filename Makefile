.PHONY: setup lint test download all

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

download:
	uv run fremtpl2 download

all: download
