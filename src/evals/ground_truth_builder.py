import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Literal

import duckdb
from pydantic import BaseModel
from tqdm import tqdm

from src.agent.llm_service import LLMService
from src.agent.prompts import Prompt
from src.etl.data_loader import DataLoader


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH_PARQUET = PROJECT_ROOT / "data" / "curated" / "ground_truth.parquet"

DEFAULT_SAMPLE_SIZE = 500
# Fixed seed so a rerun evaluates the same songs; without it, hit rate moves
# between runs because the sample changed, not because retrieval changed.
SAMPLE_SEED = 42
# The embedder truncates at 256 tokens, so anything past roughly this many
# words is never indexed. Generating questions from text the retriever cannot
# see would score it on content it was never given.
LYRICS_WORD_LIMIT = 180
# Each song is one independent, network-bound API call, so threads help even
# under the GIL. Kept modest: every worker competes for the same rate limit,
# and 429s cost more wall time in retries than the extra concurrency buys.
DEFAULT_MAX_WORKERS = 8


class GroundTruthQuestion(BaseModel):
    question: str
    type: Literal["CONTENT", "THEME", "SITUATION"]

class GroundTruthQuestionSet(BaseModel):
    questions: List[GroundTruthQuestion]

class GroundTruthBuilder():
    llm: LLMService
    data: DataLoader
    sample_size: int
    ground_truth_parquet: Path
    max_workers: int

    def __init__(
        self,
        llm: LLMService,
        data: DataLoader,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        ground_truth_parquet: str | Path = DEFAULT_GROUND_TRUTH_PARQUET,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        self.llm = llm
        self.data = data
        self.sample_size = sample_size
        self.ground_truth_parquet = Path(ground_truth_parquet)
        self.max_workers = max_workers

    def generate_ground_truth_set(self, force: bool = False) -> Path:
        """Build the question set used to score retrieval, and cache it.

        Skips the whole run if the parquet already exists, so the expensive
        LLM pass happens once per corpus. A song whose generation fails is
        logged and skipped instead of aborting the batch.
        """
        if self.ground_truth_parquet.exists() and not force:
            logger.info(
                "Ground truth already exists at %s, skipping generation.",
                self.ground_truth_parquet,
            )
            return self.ground_truth_parquet

        songs = self._sample_data(self.sample_size)

        rows: list[tuple] = []
        failures = 0

        # Only the worker calls fan out; rows and failures are touched solely
        # by this loop, which runs on the main thread, so no locking is needed.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._generate_question, song): song for song in songs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating ground truth",
            ):
                song = futures[future]
                try:
                    questions = future.result()
                except Exception as error:
                    failures += 1
                    logger.warning(
                        "Question generation failed for song %s (%s): %s",
                        song["id"],
                        type(error).__name__,
                        error,
                    )
                    continue

                rows.extend(
                    (
                        row["song_id"],
                        row["title"],
                        row["artist"],
                        row["question"],
                        row["type"],
                    )
                    for row in questions
                )

        # Completion order is nondeterministic, which would otherwise leak into
        # the file and make two runs of the same seed produce different parquets.
        # The sort is stable, so each song keeps its own question order.
        rows.sort(key=lambda row: row[0])

        if not rows:
            raise RuntimeError(
                "No ground truth questions were generated; every song failed."
            )

        self.ground_truth_parquet.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE ground_truth ("
            "song_id BIGINT, title VARCHAR, artist VARCHAR, "
            "question VARCHAR, type VARCHAR)"
        )
        con.executemany("INSERT INTO ground_truth VALUES (?, ?, ?, ?, ?)", rows)
        con.execute(
            "COPY ground_truth TO "
            f"'{self.ground_truth_parquet.as_posix()}' (FORMAT PARQUET)"
        )
        con.close()

        logger.info(
            "Wrote %d questions from %d/%d songs to %s (%d failures)",
            len(rows),
            len(songs) - failures,
            len(songs),
            self.ground_truth_parquet,
            failures,
        )
        return self.ground_truth_parquet

    def _sample_data(self, sample_size: int) -> List[Dict[str, str]]:
        """Draw a reproducible random sample of songs to generate questions for.

        Lyrics are trimmed to the window the embedder actually indexes, so the
        generator can only ask about text the retriever is able to find.
        """
        relation = self.data.load_data()
        rows = duckdb.sql(
            f"""
            SELECT id, title, artist, tag, year, lyrics
            FROM relation
            USING SAMPLE reservoir({int(sample_size)} ROWS) REPEATABLE ({SAMPLE_SEED})
            """
        ).fetchall()

        songs = []
        for song_id, title, artist, tag, year, lyrics in rows:
            songs.append(
                {
                    "id": song_id,
                    "title": title,
                    "artist": artist,
                    "tag": tag,
                    "year": year,
                    "lyrics": " ".join(lyrics.split()[:LYRICS_WORD_LIMIT]),
                }
            )

        logger.info("Sampled %d songs for ground truth generation.", len(songs))
        return songs

    def _generate_question(self, song: Dict[str, str]):
        instruction = Prompt.get_ground_truth_instruction()

        response = self.llm.chat(
            messages=[self.llm.user_message(json.dumps(song, ensure_ascii=False))],
            system=instruction,
            text_format=GroundTruthQuestionSet,
        )
        output = response.parsed

        results = []

        for q in output.questions:
            results.append({
                "song_id": song["id"],
                "title": song["title"],
                "artist": song["artist"],
                "question": q.question,
                "type": q.type,
            })

        return results


if __name__ == "__main__":
    # Run from the project root as: python -m src.evals.ground_truth_builder
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    llm = LLMService()
    data = DataLoader()
    ground_truth_builder = GroundTruthBuilder(llm, data, sample_size=500)

    ground_truth_builder.generate_ground_truth_set()
