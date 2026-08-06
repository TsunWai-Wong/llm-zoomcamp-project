import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from src.agent.tool_registry import ToolSchema

from .base import ChatResponse, ToolCall, ToolResult


DEFAULT_MODEL = "gemini-2.5-flash"

# google-genai raises one APIError for every HTTP failure and carries the status
# on .code, so unlike OpenAI there is no exception class per failure mode to
# match on. 429 is rate limiting and 5xx is a server fault; 4xx is a caller bug.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class GeminiProvider:
    """Adapter for the Google Gemini API (google-genai SDK)."""

    client: genai.Client
    default_model: str

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self.client = genai.Client(api_key=api_key)
        self.default_model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: list,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        config: dict[str, Any] = {}
        if system is not None:
            # Gemini has no system role in contents; the instruction rides on
            # the request config instead.
            config["system_instruction"] = system
        if tools is not None:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        self._to_function_declaration(tool) for tool in tools
                    ]
                )
            ]
        if text_format is not None:
            # Gemini rejects a response schema combined with tools. No call site
            # asks for both — the agent uses tools, the evaluators use schemas.
            config["response_mime_type"] = "application/json"
            config["response_schema"] = text_format

        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=messages,
            config=types.GenerateContentConfig(**config),
        )
        return self._to_chat_response(response)

    def user_message(self, text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    def extend(
        self,
        messages: list,
        response: ChatResponse,
        results: list[ToolResult],
    ) -> list:
        """Append the model turn and its tool outputs to the history."""
        history = [*messages, response.raw.candidates[0].content]
        if not results:
            return history

        # Function results go back as a user turn: Gemini has no dedicated tool
        # role, and an empty parts list would be rejected, hence the guard above.
        return [
            *history,
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=result.name,
                        response={"result": result.output},
                    )
                    for result in results
                ],
            ),
        ]

    def is_retryable(self, error: Exception) -> bool:
        return (
            isinstance(error, errors.APIError)
            and error.code in RETRYABLE_STATUS_CODES
        )

    @staticmethod
    def _to_function_declaration(schema: ToolSchema) -> types.FunctionDeclaration:
        """Convert a neutral schema into a Gemini function declaration."""
        return types.FunctionDeclaration(
            name=schema.name,
            description=schema.description,
            parameters=schema.parameters.model_dump(),
        )

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        """Flatten the candidate's parts into a ChatResponse."""
        candidates = response.candidates or []
        content = candidates[0].content if candidates else None
        parts = (content.parts if content else None) or []

        texts = []
        tool_calls = []

        for index, part in enumerate(parts):
            if part.function_call is not None:
                call = part.function_call
                tool_calls.append(
                    ToolCall(
                        # Gemini usually leaves the id unset and matches results
                        # by function name, so one is synthesized to satisfy the
                        # normalized type.
                        id=call.id or f"call_{index}",
                        name=call.name,
                        arguments=dict(call.args or {}),
                    )
                )
            elif part.text:
                texts.append(part.text)

        return ChatResponse(
            text="".join(texts) if texts else None,
            tool_calls=tool_calls,
            # Only set when a response_schema was requested.
            parsed=getattr(response, "parsed", None),
            raw=response,
        )
