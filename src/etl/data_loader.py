from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb
import kagglehub


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"
DEFAULT_TOP_N = 10_000
DEFAULT_CURATED_FILENAME = "songs.parquet"


class DataLoader:
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET,
        raw_dir: str | Path = PROJECT_ROOT / "data" / "raw",
        curated_dir: str | Path = PROJECT_ROOT / "data" / "curated",
        top_n: int = DEFAULT_TOP_N,
        curated_filename: str = DEFAULT_CURATED_FILENAME,
    ) -> None:
        self.dataset_path = dataset_path
        self.raw_dir = Path(raw_dir)
        self.curated_dir = Path(curated_dir)
        self.top_n = top_n
        self.raw_csv = self.raw_dir / "song_lyrics.csv"
        # Overridable so other curated tables (the ground truth set) can be read
        # through the same loader. Only load_data() is meaningful for those;
        # download_data() and transform_data() still target the song corpus.
        self.curated_parquet = self.curated_dir / curated_filename

    def download_data(self, force: bool = False) -> Path:
        """Download the dataset from Kaggle and copy the CSV into the raw folder.

        Skips the copy if the CSV is already present, unless force is True.
        """
        if self.raw_csv.exists() and not force:
            logger.info("Raw CSV already exists at %s, skipping download.", self.raw_csv)
            return self.raw_csv

        download_dir = Path(kagglehub.dataset_download(self.dataset_path))
        csv_files = sorted(download_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in Kaggle download at {download_dir}")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Copying %s to %s (this may take a few minutes)...",
            csv_files[0],
            self.raw_csv,
        )
        shutil.copy2(csv_files[0], self.raw_csv)
        return self.raw_csv

    def transform_data(self, force: bool = False) -> Path:
        """Filter the raw CSV and write the curated Parquet file.

        Keeps English songs, drops the 'misc' tag (non-song entries), and
        selects the top N songs by views. DuckDB streams the CSV, so the
        raw file is never held in memory.
        """
        if self.curated_parquet.exists() and not force:
            logger.info(
                "Curated Parquet already exists at %s, skipping transform.",
                self.curated_parquet,
            )
            return self.curated_parquet

        if not self.raw_csv.exists():
            raise FileNotFoundError(
                f"Raw CSV not found at {self.raw_csv}. Run download_data() first."
            )

        self.curated_dir.mkdir(parents=True, exist_ok=True)
        duckdb.sql(
            f"""
            COPY (
                SELECT id, title, artist, tag, year, views, lyrics
                FROM read_csv_auto('{self.raw_csv.as_posix()}')
                WHERE language = 'en'
                  AND tag <> 'misc'
                ORDER BY views DESC
                LIMIT {self.top_n}
            ) TO '{self.curated_parquet.as_posix()}' (FORMAT PARQUET)
            """
        )
        logger.info("Wrote curated Parquet to %s", self.curated_parquet)
        return self.curated_parquet

    def load_data(self) -> duckdb.DuckDBPyRelation:
        """Load the curated Parquet file through DuckDB.

        Returns a lazy relation: downstream consumers can select only the
        columns they need or pull rows in batches instead of materialising
        the whole dataset at once.
        """
        if not self.curated_parquet.exists():
            raise FileNotFoundError(
                f"Curated Parquet not found at {self.curated_parquet}. "
                "Run transform_data() first."
            )
        return duckdb.read_parquet(str(self.curated_parquet))
