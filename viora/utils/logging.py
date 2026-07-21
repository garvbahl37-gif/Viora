"""Centralised logging.

Library code uses ``logging.getLogger(__name__)`` and never ``print``. This
module configures a single rich-formatted root handler for the ``viora``
namespace so scripts get readable output while library consumers keep control.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

_CONFIGURED = False


def get_logger(name: str = "viora") -> logging.Logger:
    """Return a namespaced logger. Cheap and idempotent."""
    return logging.getLogger(name)


def configure_logging(
    level: str | int = "INFO",
    *,
    rich: bool = True,
    logfile: str | None = None,
) -> None:
    """Configure the ``viora`` logger once. Safe to call repeatedly.

    Args:
        level: Log level name or value.
        rich: Use rich formatting for the console handler when available.
        logfile: Optional path to also write plain-text logs to.
    """
    global _CONFIGURED
    logger = logging.getLogger("viora")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated configuration.
    logger.handlers.clear()
    logger.propagate = False

    if rich and _HAS_RICH:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=False
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(file_handler)

    _CONFIGURED = True


def log_dict(logger: logging.Logger, title: str, data: dict[str, Any], level: int = logging.INFO) -> None:
    """Log a dictionary as aligned ``key: value`` lines (config / metric dumps)."""
    if not data:
        logger.log(level, "%s: <empty>", title)
        return
    width = max(len(str(k)) for k in data)
    lines = [f"{title}:"]
    lines.extend(f"  {str(k):<{width}} : {v}" for k, v in data.items())
    logger.log(level, "\n".join(lines))
