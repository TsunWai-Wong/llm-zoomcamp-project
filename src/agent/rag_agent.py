import json

from openinference.semconv.trace import SpanAttributes

from src.monitoring import get_tracer

from .llm_service import LLMService
from .tool_registry import ToolRegistry

tracer = get_tracer(__name__)

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
        with tracer.start_as_current_span(
            "agentic_loop", openinference_span_kind="agent"
        ) as agent_span:
            agent_span.set_input(str(messages[-1].get("content", "")))

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
                        with tracer.start_as_current_span(
                            item.name, openinference_span_kind="tool"
                        ) as tool_span:
                            tool_span.set_attribute(SpanAttributes.TOOL_NAME, item.name)
                            tool_span.set_input(item.arguments)
                            result = self.tools.dispatch(item.name, arguments)
                            tool_span.set_output(result)
                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result,
                        })
                    elif item.type == "message":
                        last_answer = item.content[0].text

                if not has_function_calls:
                    agent_span.set_output(last_answer or "")
                    return last_answer

            raise AgentLoopError(f"Agent did not complete within {max_turns} turns")
