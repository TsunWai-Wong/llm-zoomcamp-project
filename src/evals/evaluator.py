from collections import Counter
from pydantic import BaseModel
from typing import Literal, Callable, List
from tqdm import tqdm
import json
import random

from src.search.text_search import TextSearch
from src.search.vector_search import VectorSearch
from src.search.hybrid_search import HybridSearch
from src.etl.data_loader import DataLoader
from src.agent.prompts import Prompt
from src.agent.conversation import Conversation
from src.agent.llm_service import LLMService


# Fixed seed so two runs judge the same questions; otherwise a change in the
# OK rate could just be a different slice of the ground truth.
JUDGE_SAMPLE_SEED = 42


class LLMJudgement(BaseModel):
    reasoning: str
    recommended_songs: list[str]
    relevance: Literal["RELEVANT", "PARTIAL", "IRRELEVANT"]


class Evalutator():
    ground_truth: DataLoader
    text_search: TextSearch
    vector_search: VectorSearch
    hybrid_search: HybridSearch
    conversation: Conversation
    llm_judge: LLMService

    def __init__(self, ground_truth: DataLoader, text_search: TextSearch,
                 vector_search: VectorSearch, hybrid_search: HybridSearch,
                 agent: Conversation):
        data = ground_truth.load_data()
        self.ground_truth = [dict(zip(data.columns, row)) for row in data.fetchall()]
        self.text_search = text_search
        self.vector_search = vector_search
        self.hybrid_search = hybrid_search
        self.conversation = agent
        self.llm_judge = LLMService()

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

    def evaluate_llm(self, sample_size: int) -> None:
        """Judge the agent's answers on a random slice of the ground truth.

        Each question is judged independently; a question whose agent call or
        judge call fails is counted and skipped rather than aborting the run.
        """
        questions = random.Random(JUDGE_SAMPLE_SEED).sample(
            self.ground_truth, k=min(sample_size, len(self.ground_truth))
        )

        relevance_counts = Counter()
        ungrounded = 0
        ok = 0
        judged = 0
        failures = 0

        for row in tqdm(questions, desc="Judging agent answers"):
            try:
                grounded, relevance = self._judge_response(row["question"])
            except Exception as error:
                failures += 1
                print(
                    f"Judging failed for {row['question'][:60]!r}: "
                    f"{type(error).__name__}: {error}"
                )
                continue

            judged += 1
            relevance_counts[relevance] += 1
            ungrounded += int(not grounded)
            ok += int(grounded and relevance == "RELEVANT")

        if not judged:
            print("No answers were judged successfully.")
            return

        print(f"\nJudged {judged}/{len(questions)} answers ({failures} failures)")
        print(f"  Ungrounded: {ungrounded} ({ungrounded / judged:.1%})")
        for level in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
            print(f"  {level}: {relevance_counts[level]}")
        print(f"OK (grounded and RELEVANT): {ok} ({ok / judged:.1%})")

    def _judge_response(self, question: str) -> tuple[bool, str]:
        """Return (grounded, relevance) so the two failures can be counted apart.

        Groundedness is decided here in code rather than by the judge: whether a
        recommended title is in search_results is an exact check, not a matter
        of opinion.
        """
        instruction = Prompt.get_judge_instruction()
        search_results = [
            {"title": doc["title"], "artist": doc["artist"]}
            for doc in self.hybrid_search.hybrid_search(question, 5)
        ]
        # Each judged question must start from an empty history, or question N
        # sees the previous N-1 answers and can reply from that context instead
        # of searching — which the judge would then score as ungrounded.
        self.conversation.reset()
        agent_answer = self.conversation.ask(question)
        user_prompt = json.dumps({
                "question": question,
                "search_results": search_results,
                "answer": agent_answer,
            }, ensure_ascii=False)
        judgement = self.llm_judge.chat(
            messages=[self.llm_judge.user_message(user_prompt)],
            system=instruction,
            text_format=LLMJudgement,
        ).parsed
        titles = {song["title"].lower() for song in search_results}
        grounded = all(s.lower() in titles for s in judgement.recommended_songs)
        return grounded, judgement.relevance


if __name__ == "__main__":
    from src.evals.ground_truth_builder import DEFAULT_GROUND_TRUTH_PARQUET
    from src.search.embedder import Embedder
    from src.agent.tool_registry import ToolRegistry
    from src.agent.rag_agent import RAGAgent

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

    tools = ToolRegistry()
    tools.register("search", hybrid_search.hybrid_search)
    agent = RAGAgent(tools, LLMService(), Prompt.get_agent_instruction())
    conversation = Conversation(agent)


    evaluator = Evalutator(ground_truth, text_search, vector_search, hybrid_search, conversation)
    # evaluator.evaluate_search()

    evaluator.evaluate_llm(sample_size=10)