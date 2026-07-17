"""Structured JSON logging with a request-scoped requestId.

Every log line carries the current requestId (via contextvar), so access logs,
audit logs and error logs for one HTTP request can be correlated end-to-end.
"""

import logging
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

# Set by the requestId middleware at the start of each request.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return request_id_ctx.get()


class _RequestIdFilter(logging.Filter):
    """Inject the current requestId into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.requestId = request_id_ctx.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install a single JSON handler on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove pre-existing handlers so we don't double-log under uvicorn --reload.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(requestId)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
            timestamp=True,
        )
    )
    root.addHandler(handler)

    # Route uvicorn's own loggers through the root handler (avoid duplicate lines).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
