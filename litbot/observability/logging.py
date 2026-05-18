import logging
import sys
from typing import TextIO

import structlog


def configure_logging(stream: TextIO = sys.stdout) -> None:
    """Configure JSON-friendly structured logs for API, ingestion, and retrieval events."""

    logging.basicConfig(format="%(message)s", stream=stream, level=logging.INFO, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )
