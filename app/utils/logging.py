"""
Logging configuration using structlog.
"""

import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO", format_type: str = "json") -> Any:
    """
    Configure structured logging with structlog.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("json" or "text")

    Returns:
        Configured logger instance
    """

    # Set up processors based on format type
    if format_type.lower() == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Set logging level
    log_level = getattr(logging, level.upper())
    logging.basicConfig(format="%(message)s", level=log_level, handlers=[handler])

    # Get structured logger
    logger = structlog.get_logger()

    return logger


def create_request_logger(request_id: str) -> Any:
    """
    Create a logger bound to a specific request ID.

    Args:
        request_id: Unique request identifier

    Returns:
        Logger bound with request_id
    """
    logger = structlog.get_logger()
    return logger.bind(request_id=request_id)
