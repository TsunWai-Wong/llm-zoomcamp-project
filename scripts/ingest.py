from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.data_loader import DEFAULT_DATASET, DEFAULT_TOP_N, DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # SystemExit
