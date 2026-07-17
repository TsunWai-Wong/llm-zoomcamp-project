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

    def hybrid_search(self, query, k=60):
        text_results = self.text_searcher.text_search(query, num_results=10)
        vector_results = self.vector_searcher.vector_search(query, num_results=10)
        return self._rrf([text_results, vector_results], k=k)