from typing import List, Dict

from src.agent.rag_agent import RAGAgent

class Conversation:
    agent: RAGAgent
    messages: List[Dict[str, str]]

    def __init__(self, agent: RAGAgent, instruction: str):
        self.agent = agent
        self.messages = [{"role": "system", "content": instruction}]

    def reset(self) -> None:
        """Drop the conversation history, keeping the system instruction."""
        self.messages = self.messages[:1]

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        answer = self.agent.agentic_loop(self.messages)
        return answer