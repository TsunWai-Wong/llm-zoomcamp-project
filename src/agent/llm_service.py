import logging
import os
import time
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0

# The SDK raises typed exceptions instead of surfacing HTTP status codes:
# 429 -> RateLimitError, 5xx -> InternalServerError, and network problems or
# timeouts -> APIConnectionError. Anything else (401 auth, 400 bad request,
# 404 model not found) is a caller bug and must not be retried.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class LLMService:
    client: OpenAI
    model: str
    max_attempts: int
    backoff_seconds: float

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_seconds: float = BACKOFF_SECONDS,
    ) -> None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        # max_retries=0 disables the SDK's built-in retries so that the
        # retry loop in chat() is the only one running.
        self.client = OpenAI(max_retries=0)
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        text_format: Any | None = None,
        model: str | None = None,
    ) -> Response:
        """Call the Responses API with exponential-backoff retries.

        Pass text_format (a Pydantic model) for structured output; the
        parsed object is available on the returned response as
        response.output_parsed. Pass tools for agentic tool calling.
        """
        model = model or self.model

        kwargs: dict[str, Any] = {"model": model, "input": messages}
        if tools is not None:
            kwargs["tools"] = tools

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if text_format is not None:
                    return self.client.responses.parse(
                        text_format=text_format, **kwargs
                    )
                return self.client.responses.create(**kwargs)
            except RETRYABLE_ERRORS as error:
                last_error = error
                if attempt == self.max_attempts:
                    break
                delay = self.backoff_seconds * 2 ** (attempt - 1)
                logger.warning(
                    "OpenAI call failed (%s), attempt %d/%d, retrying in %.1fs",
                    type(error).__name__,
                    attempt,
                    self.max_attempts,
                    delay,
                )
                time.sleep(delay)

        raise last_error
