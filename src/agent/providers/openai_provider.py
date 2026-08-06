import json
import os
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.agent.tool_registry import ToolSchema

from .base import ChatResponse, ToolCall, ToolResult


DEFAULT_MODEL = "gpt-5.4-mini"

# The SDK raises typed exceptions instead of surfacing HTTP status codes:
# 429 -> RateLimitError, 5xx -> InternalServerError, and network problems or
# timeouts -> APIConnectionError. Anything else (401 auth, 400 bad request,
# 404 model not found) is a caller bug and must not be retried.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class OpenAIProvider:
    """Adapter for the OpenAI Responses API."""

    client: OpenAI
    default_model: str

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        # max_retries=0 disables the SDK's built-in retries so that the retry
        # loop in LLMService is the only one running.
        self.client = OpenAI(max_retries=0)
        self.default_model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: list,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        # The Responses API takes the system prompt as an ordinary message, so
        # it is prepended to a copy here. Keeping it out of the stored history
        # is what lets Gemini, which passes it as config instead, share one loop.
        payload = list(messages)
        if system is not None:
            payload.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": payload,
        }
        if tools is not None:
            kwargs["tools"] = [self._to_function_tool(tool) for tool in tools]

        if text_format is not None:
            response = self.client.responses.parse(
                text_format=text_format, **kwargs
            )
        else:
            response = self.client.responses.create(**kwargs)

        return self._to_chat_response(response)

    def user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def extend(
        self,
        messages: list,
        response: ChatResponse,
        results: list[ToolResult],
    ) -> list:
        """Append the model turn and its tool outputs to the history.

        The Responses API accepts its own output items back verbatim, so the
        raw items are reused rather than rebuilt from the normalized response.
        """
        return [
            *messages,
            *response.raw.output,
            *(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.output,
                }
                for result in results
            ),
        ]

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, RETRYABLE_ERRORS)

    @staticmethod
    def _to_function_tool(schema: ToolSchema) -> dict:
        """Wrap a neutral schema in the Responses API function envelope."""
        return {"type": "function", **schema.model_dump()}

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        """Flatten the Responses output list into a ChatResponse."""
        text = None
        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        # call_id, not id: call_id is the one the API matches a
                        # function_call_output against.
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments or "{}"),
                    )
                )
            elif item.type == "message":
                text = item.content[0].text

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            # Only responses.parse() produces this attribute.
            parsed=getattr(response, "output_parsed", None),
            raw=response,
        )
