from .text_search import TextSearch
from .vector_search import VectorSearch

class HybridSearch():
    text_searcher: TextSearch
    vector_searcher: VectorSearch

    def __init__(self, text_searcher: TextSearch,
                 vector_searcher: VectorSearch) -> None:
        self.text_searcher = text_searcher
        self.vector_searcher = vector_searcher

    def _rrf(self, result_lists, k=60, num_results=5):
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["id"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]

    def hybrid_search(self, query: str, num_results: int = 10) -> list[dict]:
        """Search the song corpus with combined full-text and semantic search.

        Runs Elasticsearch full-text search and vector similarity search,
        then merges both rankings with Reciprocal Rank Fusion.

        Args:
            query: Natural-language description of the songs to find.
            num_results: Maximum number of songs to return.
        """
        pool_size = max(2 * num_results, 10)
        text_results = self.text_searcher.text_search(query, num_results=pool_size)
        vector_results = self.vector_searcher.vector_search(query, num_results=pool_size)
        return self._rrf([text_results, vector_results], num_results=num_results)