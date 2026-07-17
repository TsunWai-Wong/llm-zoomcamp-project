from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.data_loader import DEFAULT_DATASET, DEFAULT_TOP_N, DataLoader
from src.search.text_search import TextSearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the song lyrics dataset and transform it into a curated Parquet file."
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET,
        help="Kaggle dataset path.",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Stop after the transform step; do not index into Elasticsearch.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top songs by views to keep in the curated dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-transform even if local files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = DataLoader(dataset_path=args.dataset_path, top_n=args.top_n)

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

    if args.skip_indexing:
        logger.info("Skipping Elasticsearch indexing (--skip-indexing).")
        return 0

    try:
        search = TextSearch()
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
            "Is the server running? Start it with 'make up'."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # SystemExit
