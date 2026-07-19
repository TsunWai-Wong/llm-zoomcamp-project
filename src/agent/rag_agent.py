import json

from .llm_service import LLMService
from .tool_registry import ToolRegistry

class AgentLoopError(Exception):
    pass

class RAGAgent:
    tools: ToolRegistry
    llm: LLMService


    def __init__(self, tools: ToolRegistry, llm: LLMService):
        self.tools = tools
        self.llm = llm

    def agentic_loop(self, messages: list, max_turns: int = 25) -> str:
        """Run the agentic loop until the model produces a final text response."""
        tool_schemas = self.tools.get_schemas()
        last_answer = None

        for _ in range(max_turns):
            response = self.llm.chat(messages=messages, tools=tool_schemas)
            messages.extend(response.output)

            has_function_calls = False
            for item in response.output:
                if item.type == "function_call":
                    has_function_calls = True
                    arguments = json.loads(item.arguments)
                    result = self.tools.dispatch(item.name, arguments)
                    messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    })
                elif item.type == "message":
                    last_answer = item.content[0].text

            if not has_function_calls:
                return last_answer

        raise AgentLoopError(f"Agent did not complete within {max_turns} turns")
