from src.agent.rag_agent import RAGAgent

class Conversation:
    agent: RAGAgent
    instruction: str
    history: list

    def __init__(self, agent: RAGAgent, instruction: str):
        self.agent = agent
        # Held as plain text rather than as the first message, because only the
        # provider knows where the instruction belongs: OpenAI takes it as a
        # message, Gemini as system_instruction on the request config.
        self.instruction = instruction
        self.history = []

    def reset(self) -> None:
        """Drop the conversation history. The instruction is not part of it."""
        self.history = []

    def ask(self, question: str) -> str:
        answer, self.history = self.agent.agentic_loop(
            question, history=self.history, system=self.instruction
        )
        return answer
