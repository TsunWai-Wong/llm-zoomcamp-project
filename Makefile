PYTHON = uv run python3

.PHONY: all install ingest ground-truth evaluate run down

all: install

install:
	uv sync
	$(PYTHON) scripts/download_embedding_model.py
	docker compose up -d --wait

ingest: install
	$(PYTHON) scripts/ingest.py --skip-etl

ground-truth: ingest
	$(PYTHON) -m src.evals.ground_truth_builder

evaluate: ingest ground-truth
	$(PYTHON) -m src.evals.evaluator

run: install ingest
	uv run streamlit run app.py

down:
	docker compose down
