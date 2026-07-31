from pydantic import BaseModel
from typing import Literal, Callable, List
from tqdm import tqdm

from src.search.text_search import TextSearch
from src.search.vector_search import VectorSearch
from src.search.hybrid_search import HybridSearch
from src.etl.data_loader import DataLoader


class LLMJudgement(BaseModel):
    reasoning: str
    rating: Literal["OK", "Not OK"] 


class Evalutator():
    ground_truth: DataLoader
    text_search: TextSearch
    vector_search: VectorSearch
    hybrid_search: HybridSearch

    def __init__(self, ground_truth: DataLoader, text_search: TextSearch,
                 vector_search: VectorSearch, hybrid_search: HybridSearch):
        data = ground_truth.load_data()
        self.ground_truth = [dict(zip(data.columns, row)) for row in data.fetchall()]
        self.text_search = text_search
        self.vector_search = vector_search
        self.hybrid_search = hybrid_search

    def evaluate_search(self):
        text_results = self._conduct_search(self.text_search.text_search)
        vector_results = self._conduct_search(self.vector_search.vector_search)
        hybrid_results = self._conduct_search(self.hybrid_search.hybrid_search)
        print(f"Hit rate (text search): {self._calculate_hit_rate(text_results)}")
        print(f"Hit rate (vector search): {self._calculate_hit_rate(vector_results)}")
        print(f"Hit rate (hybrid search): {self._calculate_hit_rate(hybrid_results)}")

        print(f"mrr (text search): {self._calculate_mrr(text_results)}")
        print(f"mrr (vector search): {self._calculate_mrr(vector_results)}")
        print(f"mrr (hybrid search): {self._calculate_mrr(hybrid_results)}")

    def _conduct_search(self, search_method: Callable) -> List:
        results = []

        for i in tqdm(range(len(self.ground_truth)),
                              desc=f"Conducting searches ({search_method.__name__})"
                              ):
                    query = self.ground_truth[i]["question"]
                    standard_answer = self.ground_truth[i]["song_id"]
                    results.append([int(ans['id'] == standard_answer) for ans in search_method(query, 5)])

        return results

    def _calculate_hit_rate(self, results: List) -> float:
            
        count = sum(1 for ans in results if 1 in ans)
        hit_rate = count / len(results)

        return hit_rate

    def _calculate_mrr(self, results: List) -> float:

        total_score = 0.0

        for line in results:
            for rank in range(len(line)):
                if line[rank] == 1:
                    total_score = total_score + 1 / (rank + 1)
                    break

        return total_score / len(results)


if __name__ == "__main__":
    from src.evals.evals_initializer import DEFAULT_GROUND_TRUTH_PARQUET
    from src.search.embedder import Embedder

    loader = DataLoader()
    data = loader.load_data()

    # The ground truth is a different curated table, so point a loader at that
    # file rather than the song corpus. Evalutator only calls load_data().
    ground_truth = DataLoader(curated_filename=DEFAULT_GROUND_TRUTH_PARQUET.name)

    text_search = TextSearch()
    embedder = Embedder()
    vector_search = VectorSearch(embedder)
    vector_search.build_index(data)
    
    hybrid_search = HybridSearch(text_search, vector_search)

    evaluator = Evalutator(ground_truth, text_search, vector_search, hybrid_search)
    evaluator.evaluate_search()