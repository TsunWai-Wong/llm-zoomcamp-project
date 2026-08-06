import json

from openinference.semconv.trace import SpanAttributes

from src.monitoring import get_tracer

from .llm_service import LLMService
from .providers import ToolResult
from .tool_registry import ToolRegistry

tracer = get_tracer(__name__)

class AgentLoopError(Exception):
    pass

class RAGAgent:
    tools: ToolRegistry
    llm: LLMService
    instruction: str


    def __init__(self, tools: ToolRegistry, llm: LLMService, instruction: str):
        self.tools = tools
        self.llm = llm
        # The instruction describes the agent, not any one conversation, so it
        # sits alongside the tools and the model. Being immutable is what lets
        # an agent be built once and shared while histories stay per session.
        self.instruction = instruction

    def agentic_loop(
        self,
        question: str,
        history: list | None = None,
        max_turns: int = 25,
    ) -> tuple[str, list]:
        """Run the agentic loop until the model produces a final text response.

        Returns the answer together with the extended history. That history is
        in whatever shape the active provider uses; it is only ever passed back
        in, never read here, which is what keeps this loop provider-agnostic.
        """
        with tracer.start_as_current_span(
            "agentic_loop", openinference_span_kind="agent"
        ) as agent_span:
            agent_span.set_input(question)

            messages = [*(history or []), self.llm.user_message(question)]
            tool_schemas = self.tools.get_schemas()

            for _ in range(max_turns):
                response = self.llm.chat(
                    messages=messages,
                    system=self.instruction,
                    tools=tool_schemas,
                )

                if not response.tool_calls:
                    # Record the final turn too, or the next question in the
                    # conversation would not see the answer it follows from.
                    messages = self.llm.extend(messages, response, [])
                    agent_span.set_output(response.text or "")
                    return response.text, messages

                results = []
                for call in response.tool_calls:
                    with tracer.start_as_current_span(
                        call.name, openinference_span_kind="tool"
                    ) as tool_span:
                        tool_span.set_attribute(SpanAttributes.TOOL_NAME, call.name)
                        tool_span.set_input(json.dumps(call.arguments))
                        output = self.tools.dispatch(call.name, call.arguments)
                        tool_span.set_output(output)
                    results.append(
                        ToolResult(call_id=call.id, name=call.name, output=output)
                    )

                messages = self.llm.extend(messages, response, results)

            raise AgentLoopError(f"Agent did not complete within {max_turns} turns")
