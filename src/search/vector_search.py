import logging
from pathlib import Path

import duckdb
import minsearch
import numpy as np
from tqdm import tqdm

from src.search.embedder import Embedder


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_PARQUET = PROJECT_ROOT / "data" / "curated" / "embeddings.parquet"

BATCH_SIZE = 32


class VectorSearch:
    embedder: Embedder
    embeddings_parquet: Path
    vector_index: minsearch.VectorSearch | None

    def __init__(
        self,
        embedder: Embedder,
        embeddings_parquet: str | Path = DEFAULT_EMBEDDINGS_PARQUET,
    ) -> None:
        self.embedder = embedder
        self.embeddings_parquet = Path(embeddings_parquet)
        self.vector_index = None

    def embedd_documents(self, data: duckdb.DuckDBPyRelation) -> None:
        """Embed all lyrics and cache them in a parquet file keyed by id.

        Skips the whole computation if the cache file already exists, so the
        expensive embedding run happens only once per corpus/model.
        """
        if self.embeddings_parquet.exists():
            logger.info(
                "Embeddings already exist at %s, skipping embedding.",
                self.embeddings_parquet,
            )
            return

        rows = data.select("id, lyrics").fetchall()
        ids = [row[0] for row in rows]
        lyrics = [row[1] for row in rows]

        records: list[tuple[int, list[float]]] = []
        for start in tqdm(range(0, len(lyrics), BATCH_SIZE), desc="Embedding lyrics"):
            batch = lyrics[start : start + BATCH_SIZE]
            vectors = self.embedder.encode_batch(batch)
            records.extend(
                (doc_id, vector.tolist())
                for doc_id, vector in zip(ids[start : start + BATCH_SIZE], vectors)
            )

        self.embeddings_parquet.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute("CREATE TABLE embeddings (id BIGINT, embedding FLOAT[])")
        con.executemany("INSERT INTO embeddings VALUES (?, ?)", records)
        con.execute(
            f"COPY embeddings TO '{self.embeddings_parquet.as_posix()}' (FORMAT PARQUET)"
        )
        con.close()
        logger.info(
            "Wrote %d embeddings to %s", len(records), self.embeddings_parquet
        )

    def build_index(self, data: duckdb.DuckDBPyRelation) -> None:
        """Fit the in-memory vector index from cached embeddings and song payloads.

        Vectors and documents are matched by id, so the two sources do not
        need to be in the same row order.
        """
        if not self.embeddings_parquet.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self.embeddings_parquet}. "
                "Run embedd_documents() first."
            )

        embedding_by_id = dict(
            duckdb.sql(
                f"SELECT id, embedding FROM '{self.embeddings_parquet.as_posix()}'"
            ).fetchall()
        )

        columns = data.columns
        docs = [dict(zip(columns, row)) for row in data.fetchall()]
        docs = [doc for doc in docs if doc["id"] in embedding_by_id]
        vectors = np.array(
            [embedding_by_id[doc["id"]] for doc in docs], dtype=np.float32
        )

        self.vector_index = minsearch.VectorSearch()
        self.vector_index.fit(vectors, docs)
        logger.info("Vector index fitted with %d documents.", len(docs))

    def vector_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Semantic search over the lyrics embeddings, best matches first."""
        if self.vector_index is None:
            raise RuntimeError("Vector index not built. Run build_index() first.")

        query_vector = self.embedder.encode(query)
        return self.vector_index.search(query_vector, num_results=num_results)
