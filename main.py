from src.etl.data_loader import DataLoader
from src.search.text_search import TextSearch
from src.search.vector_search import VectorSearch
from src.search.embedder import Embedder
from src.search.hybrid_search import HybridSearch
from src.agent.tool_registry import ToolRegistry
from src.agent.llm_service import LLMService
from src.agent.rag_agent import RAGAgent
from src.agent.prompts import Prompt
from src.monitoring.tracing import setup_tracing

def main():
    setup_tracing()
    loader = DataLoader()
    data = loader.load_data()
    # print(loader.load_data().limit(5).fetchall())

    text_search = TextSearch()
    # print(text_search.text_search("ice cream"))
    embedder = Embedder()
    vector_search = VectorSearch(embedder)
    vector_search.build_index(data)
    
    hybrid_search = HybridSearch(text_search, vector_search)
    # print(hybrid_search.hybrid_search("Food that I eat"))
    tools = ToolRegistry()
    tools.register("search", hybrid_search.hybrid_search)

    llm = LLMService()
    agent = RAGAgent(tools, llm)

    instruction = Prompt.get_agent_instruction()
    messages = [
        {
            "role": "system",
            "content": instruction
        },
    ]

    messages.append({
        "role": "user",
        "content": "Find me some songs which can be played in a Funeral in the Winter. It should be a sad song."
    })
    print(agent.agentic_loop(messages))
    # print(tools.tools)

if __name__ == "__main__":
    main()
