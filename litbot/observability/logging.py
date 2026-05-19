import logging
import sys
from datetime import datetime
from typing import Any, TextIO

import structlog


def configure_logging(stream: TextIO = sys.stdout, renderer: str = "json") -> None:
    """Configure structured logs for API, ingestion, retrieval, and CLI events."""

    logging.basicConfig(format="%(message)s", stream=stream, level=logging.INFO, force=True)
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if renderer == "console":
        processors.append(_render_console_event)
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )


def _render_console_event(
    logger: object,
    method_name: str,
    event_dict: dict[str, Any],
) -> str:
    del logger, method_name
    event = str(event_dict.pop("event", "event"))
    timestamp = _short_time(str(event_dict.pop("timestamp", "")))
    level = str(event_dict.pop("level", "info")).upper()
    layer = _event_layer(event)
    message = _event_message(event)
    details = " ".join(
        f"{key}={_format_value(value)}"
        for key, value in sorted(event_dict.items())
        if value not in (None, "", [], {})
    )
    prefix = f"{timestamp}  {level:<5}  {layer:<10}  {message}"
    return f"{prefix}  {details}" if details else prefix


def _short_time(timestamp: str) -> str:
    if not timestamp:
        return "--:--:--"
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        return timestamp[:8]


def _event_layer(event: str) -> str:
    if event.startswith("document_"):
        return "ingestion"
    if event.startswith("retrieval_"):
        return "retrieval"
    if event.startswith("generation_"):
        return "generation"
    if event.endswith("_request") or event.endswith("_exception"):
        return "api"
    return "litbot"


def _event_message(event: str) -> str:
    return event.replace("_", " ")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return repr(value) if isinstance(value, (dict, list, tuple, set)) else str(value)
