"""Error handling for the prediction service.

Every failure leaves the service as a structured body carrying the request id, so a
caller reporting "it returned 500 at 14:03" can be traced to a specific log line.
Unhandled exceptions are never allowed to leak a stack trace to the client: the
trace goes to the log, a correlation id goes to the caller.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.predictor import ModelNotLoadedError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Name the offending field rather than returning a bare 422."""
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"][1:]) or None,
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        logger.info("Rejected malformed request %s: %s", _request_id(request), errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": "The request body failed validation.",
                "request_id": _request_id(request),
                "errors": errors,
            },
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        # 503 rather than 500: the request was fine, the service is not ready yet.
        logger.error("Model unavailable for %s: %s", _request_id(request), exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "model_unavailable",
                "detail": str(exc),
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "detail": str(exc.detail),
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error for request %s", _request_id(request))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred. Quote the request_id when reporting.",
                "request_id": _request_id(request),
            },
        )
