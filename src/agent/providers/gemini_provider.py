import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from src.agent.tool_registry import ToolSchema

from .base import ChatResponse, ToolCall, ToolResult, Usage


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

    def split_turns(
        self, messages: list, keep_last_turns: int
    ) -> tuple[list, list]:
        if keep_last_turns <= 0:
            return list(messages), []

        boundaries = [
            index
            for index, content in enumerate(messages)
            if self._is_user_turn(content)
        ]
        if len(boundaries) <= keep_last_turns:
            return [], list(messages)

        cut = boundaries[-keep_last_turns]
        return list(messages[:cut]), list(messages[cut:])

    def compact(self, messages: list, keep_last_turns: int = 1) -> list:
        older, recent = self.split_turns(messages, keep_last_turns)

        kept = []
        for content in older:
            # Rebuilding from the text parts alone drops function calls and
            # function responses while preserving a model turn that mixed text
            # with a call. A turn left with nothing is dropped outright: an
            # empty parts list is rejected, as extend() already notes.
            texts = [part for part in self._parts(content) if part.text]
            if texts:
                kept.append(types.Content(role=content.role, parts=texts))

        return [*kept, *recent]

    def render_transcript(self, messages: list) -> str:
        lines = []
        for content in messages:
            text = "".join(
                part.text for part in self._parts(content) if part.text
            )
            if not text:
                continue
            speaker = "User" if content.role == "user" else "Assistant"
            lines.append(f"{speaker}: {text}")
        return "\n\n".join(lines)

    def is_retryable(self, error: Exception) -> bool:
        return (
            isinstance(error, errors.APIError)
            and error.code in RETRYABLE_STATUS_CODES
        )

    @staticmethod
    def _parts(content: Any) -> list:
        return getattr(content, "parts", None) or []

    @classmethod
    def _is_user_turn(cls, content: Any) -> bool:
        """Whether this content is a real user question, not a tool result.

        Function results also ride under the user role — Gemini has no tool
        role — so the role alone cannot tell the two apart, and cutting the
        history at a tool result would strand the call it belongs to.
        """
        if getattr(content, "role", None) != "user":
            return False
        return not any(
            part.function_response is not None for part in cls._parts(content)
        )

    @staticmethod
    def _to_function_declaration(schema: ToolSchema) -> types.FunctionDeclaration:
        """Convert a neutral schema into a Gemini function declaration."""
        return types.FunctionDeclaration(
            name=schema.name,
            description=schema.description,
            parameters=schema.parameters.model_dump(exclude_none=True),
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

        usage = getattr(response, "usage_metadata", None)

        return ChatResponse(
            text="".join(texts) if texts else None,
            tool_calls=tool_calls,
            # Every count is optional on the SDK model, so none of them can be
            # passed through to the non-optional normalized type unguarded.
            usage=Usage(
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
                total_tokens=usage.total_token_count or 0,
            )
            if usage
            else None,
            # Only set when a response_schema was requested.
            parsed=getattr(response, "parsed", None),
            raw=response,
        )
