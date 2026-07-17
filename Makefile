PYTHON = uv run python3 


all: install

install:
	uv sync
	uv run python3 scripts/download_embedding_model.py
	docker compose up -d

ingest:
	uv run python3 scripts/ingest.py

up:
	docker compose up -d

down:
	docker compose down