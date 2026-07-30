import logging
import os

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
PROJECT_NAME = "lyrics-rag"


def setup_tracing(project_name: str = PROJECT_NAME) -> None:
    """Send OpenTelemetry traces to Phoenix and auto-instrument OpenAI calls.

    Call this once at application startup, before the first LLM call.
    If Phoenix is not running, the app keeps working: spans are dropped
    with a warning instead of raising.
    """
    load_dotenv()
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_COLLECTOR_ENDPOINT)

    from phoenix.otel import register

    register(project_name=project_name, auto_instrument=True)
    logger.info(
        "Tracing enabled: project '%s' -> %s",
        project_name,
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
    )
