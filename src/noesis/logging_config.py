"""Central logging configuration for Noesis (local-only, stderr-only).

Called once at each entry point (``noesis.app.create_app`` and
``noesis.mcp.__main__.main``) before any runtime work. Configures the
``noesis`` logger namespace with a single stderr handler at
``NOESIS_LOG_LEVEL`` (default INFO). Prior to this the ``noesis`` loggers had
no handler and the stdlib ``lastResort`` handler is WARNING-level, so every
``logger.info(...)`` in the package printed nothing — the runtime looked like
a black box even though the log calls existed.

Design constraints:
- **stderr only.** stdout carries the stdio MCP JSON-RPC stream; a log line on
  stdout corrupts it (CLAUDE.md rule 2, ``noesis.mcp.__main__``). We pin
  ``sys.stderr`` explicitly rather than relying on the implicit default.
- **No new runtime deps** (CLAUDE.md rule 3): stdlib ``logging`` + ``json``.
- **``propagate`` stays True** on the ``noesis`` logger so pytest's ``caplog``
  (a handler on the *root* logger) still captures records — the existing
  ``tests/test_reranker.py`` caplog assertions depend on this.
- This module only sets up transport/format. The content rule (ADR-25: never
  log query text or file contents) is enforced at the individual call sites.
"""

from __future__ import annotations

import json
import logging
import os
import sys

LEVEL_ENV = "NOESIS_LOG_LEVEL"
FORMAT_ENV = "NOESIS_LOG_FORMAT"
LOGGER_NAME = "noesis"

# Marks our handler so repeat configure_logging() calls (create_app runs at
# import time, and the test suite builds many apps) never stack duplicates.
_HANDLER_TAG = "noesis-stderr"

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Stdlib only. Carries exception text when
    present so tracebacks are not lost in json mode."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _resolve_level(raw: str | None) -> int:
    """Level name → int; unknown/empty → INFO. A typo in the env var must
    never crash logging setup or silence the whole app."""
    if not raw:
        return logging.INFO
    level = logging.getLevelName(raw.strip().upper())
    return level if isinstance(level, int) else logging.INFO


def _make_formatter() -> logging.Formatter:
    fmt = os.environ.get(FORMAT_ENV, "text").strip().lower()
    return _JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT)


def configure_logging() -> logging.Logger:
    """Idempotently configure the ``noesis`` logger. Safe to call repeatedly:
    the level and formatter are refreshed from the environment on every call,
    but at most one handler is ever attached."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(os.environ.get(LEVEL_ENV)))
    formatter = _make_formatter()

    for handler in logger.handlers:
        if getattr(handler, "_noesis_tag", None) == _HANDLER_TAG:
            handler.setFormatter(formatter)  # env may have changed (tests)
            return logger

    handler = logging.StreamHandler(sys.stderr)
    handler._noesis_tag = _HANDLER_TAG  # type: ignore[attr-defined]
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # propagate deliberately left at its default (True) — see module docstring.
    return logger
