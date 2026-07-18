"""Designed error codes + uniform error envelope + global exception handlers.

Every error response has the shape:
    {"error": {"code": "HPP-xxxx", "message": "...", "details": [...]}}
so clients can branch on a stable machine-readable code instead of parsing prose.
Request correlation remains exclusively in the X-Request-ID response header.
"""

from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger("app.errors")


class ErrorCode(StrEnum):
    """Stable machine-readable errors consumed by browser and gateway clients."""
    VALIDATION = "HPP-1001"       # bad/malformed input features
    MODEL_NOT_LOADED = "HPP-1002"  # artifacts missing/corrupt at startup
    INFERENCE_FAILED = "HPP-1003"  # model raised during predict
    NOT_FOUND = "HPP-1004"         # unknown route/resource
    INTERNAL = "HPP-5000"          # uncaught server error


_CODE_STATUS = {
    ErrorCode.VALIDATION: 422,
    ErrorCode.MODEL_NOT_LOADED: 503,
    ErrorCode.INFERENCE_FAILED: 500,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.INTERNAL: 500,
}


class HppError(Exception):
    """Application error carrying a designed code + optional structured details."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: list[dict] | None = None,
        status_code: int | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or []
        self.status_code = status_code or _CODE_STATUS.get(code, 500)
        super().__init__(message)


def _header_safe(text: str) -> str:
    """HTTP header values must be latin-1 and single-line."""
    return text.replace("\n", " ").replace("\r", " ").encode("latin-1", "replace").decode("latin-1")


def _error_response(status_code: int, code: str, message: str, details: list) -> JSONResponse:
    """Build the uniform error body AND set the gateway headers the frontend reads."""
    body = {"error": {"code": code, "message": message, "details": details}}
    headers = {
        "X-Error-Code": code,
        "X-Error-Message": _header_safe(message),
    }
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Translate framework and application exceptions into one public contract."""
    @app.exception_handler(HppError)
    async def _hpp(_: Request, exc: HppError):
        logger.warning("handled error %s: %s", exc.code.value, exc.message)
        return _error_response(exc.status_code, exc.code.value, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        logger.warning("request validation failed: %d issue(s)", len(details))
        return _error_response(
            422, ErrorCode.VALIDATION.value, "Request validation failed", details
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INTERNAL
        return _error_response(exc.status_code, code.value, str(exc.detail), [])

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled exception: %s", exc)
        return _error_response(500, ErrorCode.INTERNAL.value, "Internal server error", [])
