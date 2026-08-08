from src.agent.rag_agent import RAGAgent

class Conversation:
    agent: RAGAgent
    history: list

    def __init__(self, agent: RAGAgent):
        self.agent = agent
        self.history = []

    def reset(self) -> None:
        """Drop the conversation history.

        The agent's instruction is not part of it, so a reset conversation still
        behaves the same way — it has only forgotten what was said.
        """
        self.history = []

    def ask(self, question: str) -> str:
        answer, self.history = self.agent.agentic_loop(
            question, history=self.history
        )
        return answer
