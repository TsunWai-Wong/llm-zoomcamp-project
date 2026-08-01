from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.data_loader import DataLoader
from src.search.embedder import Embedder
from src.search.text_search import TextSearch
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the song lyrics dataset and transform it into a curated Parquet file."
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help=(
            "Skip the download and transform steps and reuse the existing curated "
            "Parquet, going straight to indexing and embedding."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download, re-transform and re-index even if the local files or "
            "the Elasticsearch index already exist."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = DataLoader()

    if args.skip_etl:
        if not loader.curated_parquet.exists():
            logger.error(
                "Curated dataset not found at %s. Re-run without --skip-etl "
                "to fetch and transform the dataset from Kaggle.",
                loader.curated_parquet,
            )
            return 1
        logger.info(
            "Skipping download and transform (--skip-etl); "
            "using curated dataset at %s.",
            loader.curated_parquet,
        )
    else:
        try:
            raw_path = loader.download_data(force=args.force)
            logger.info("Raw dataset is available at %s", raw_path)
        except Exception:
            logger.exception("Failed to download dataset from Kaggle.")
            return 1

        try:
            parquet_path = loader.transform_data(force=args.force)
            logger.info("Curated dataset is available at %s", parquet_path)
        except Exception:
            logger.exception("Failed to transform dataset into Parquet.")
            return 1

    try:
        search = TextSearch()
        existing = 0
        if search.es_client.indices.exists(index=search.index_name):
            search.es_client.indices.refresh(index=search.index_name)
            existing = search.es_client.count(index=search.index_name)["count"]

        # The index lives in a Docker volume, so it survives restarts. An index
        # that exists but is empty still gets rebuilt.
        if existing and not args.force:
            logger.info(
                "Elasticsearch index '%s' already holds %d documents, "
                "skipping indexing. Use --force to rebuild it.",
                search.index_name,
                existing,
            )
        else:
            search.build_index(loader.load_data())
            search.es_client.indices.refresh(index=search.index_name)
            count = search.es_client.count(index=search.index_name)["count"]
            logger.info(
                "Elasticsearch index '%s' now holds %d documents.",
                search.index_name,
                count,
            )
    except Exception:
        logger.exception(
            "Failed to index documents into Elasticsearch. "
            "Is the server running? Start it with 'make install'."
        )
        return 1

    try:
        vector_search = VectorSearch(Embedder())
        vector_search.embedd_documents(loader.load_data())
        logger.info(
            "Lyrics embeddings are available at %s", vector_search.embeddings_parquet
        )
    except Exception:
        logger.exception(
            "Failed to embed documents. Is the embedding model downloaded? "
            "Run 'uv run python3 scripts/download_embedding_model.py' first."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # SystemExit
