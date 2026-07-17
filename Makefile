PYTHON = uv run python3 


all: install

install:
	uv sync

ingest:
	uv run python3 scripts/ingest.py