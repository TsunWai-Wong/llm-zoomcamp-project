"""The retrieval layer as the agent sees it.

Search and reading are split into two tools on purpose. Returning whole songs
from a search puts roughly 675 tokens of lyrics into the context per hit, and
the agent searches several times per question — so a single turn used to cost
tens of thousands of tokens, nearly all of it lyrics the model skimmed once and
never referred to again.

search_song therefore returns only what is needed to triage a candidate, and
get_lyrics fetches the full text for the few songs that survive triage. The
docstrings below are the contract the model reads, so they are written for it
rather than for us: ToolRegistry turns them into the tool schemas verbatim.
"""

from .hybrid_search import HybridSearch
from .text_search import TextSearch


# Enough lyrics to tell what a song is about, far short of enough to answer
# from. Answering is what get_lyrics is for.
SNIPPET_CHARS = 200

MAX_LYRICS_IDS = 15
MAX_LYRICS_CHARS = 10_000


class SearchTools:
    """Bundles the searchers into the two callables registered as tools."""

    hybrid_search: HybridSearch
    text_search: TextSearch

    def __init__(
        self, hybrid_search: HybridSearch, text_search: TextSearch
    ) -> None:
        self.hybrid_search = hybrid_search
        self.text_search = text_search

    def search_song(self, query: str, num_results: int = 10) -> str:
        """Search the song corpus and return a shortlist of candidates.

        Each result gives the song's id, title, artist, year, genre and the
        opening words of its lyrics — enough to judge whether the song is
        worth reading, not enough to answer from. Pass the ids of the songs
        that look promising to get_lyrics, and recommend only songs whose
        full lyrics you have read.

        Args:
            query: Natural-language description of the songs to find.
            num_results: Maximum number of songs to return.
        """
        documents = self.hybrid_search.hybrid_search(query, num_results)
        if not documents:
            return "No songs matched that query."

        lines = []
        for document in documents:
            # Collapsing the newlines keeps every hit to two lines, which is
            # both cheaper and easier for the model to scan than raw verses.
            flat = " ".join((document.get("lyrics") or "").split())
            snippet = flat[:SNIPPET_CHARS]
            if len(flat) > SNIPPET_CHARS:
                snippet += "…"
            lines.append(f"{self._header(document)}\n  {snippet}")

        return "\n".join(lines)

    def get_lyrics(self, song_ids: list[int]) -> str:
        """Read the full lyrics of songs found with search_song.

        Use this before recommending a song, so the recommendation rests on
        the whole song rather than on its opening lines. Ask only for the
        songs you are seriously considering — at most 15 at a time, and
        usually far fewer.

        Args:
            song_ids: Ids of the songs to read, as returned by search_song.
        """
        notes = []

        requested = list(song_ids)
        if len(requested) > MAX_LYRICS_IDS:
            # Truncate rather than refuse, and say so: the model can ask for
            # the rest, whereas an error would leave it with nothing.
            notes.append(
                f"Only the first {MAX_LYRICS_IDS} of {len(requested)} ids were "
                f"read. Ask again for the rest if you still need them."
            )
            requested = requested[:MAX_LYRICS_IDS]

        documents = self.text_search.get_by_ids(requested)

        # Compared as strings so an id that arrives from the model as "4212"
        # is not reported missing against an indexed integer 4212.
        found = {str(document.get("id")) for document in documents}
        missing = [song_id for song_id in requested if str(song_id) not in found]
        if missing:
            notes.append(
                "No song found for id(s): "
                + ", ".join(str(song_id) for song_id in missing)
                + "."
            )

        blocks = []
        for document in documents:
            lyrics = document.get("lyrics") or ""
            if len(lyrics) > MAX_LYRICS_CHARS:
                lyrics = lyrics[:MAX_LYRICS_CHARS] + "\n…[truncated]"
            blocks.append(f"{self._header(document)}\n{lyrics}")

        if not blocks:
            return "\n".join(notes) or "No songs found for those ids."

        body = "\n\n---\n\n".join(blocks)
        return f"{body}\n\n{chr(10).join(notes)}" if notes else body

    @staticmethod
    def _header(document: dict) -> str:
        """One identifying line per song, leading with the id.

        The id comes first because it is the handle get_lyrics takes; a title
        the model has to guess the id for is not much use to it.
        """
        meta = ", ".join(
            str(value)
            for value in (document.get("year"), document.get("tag"))
            if value
        )
        header = (
            f"[id={document.get('id')}] {document.get('title')} "
            f"— {document.get('artist')}"
        )
        return f"{header} ({meta})" if meta else header
