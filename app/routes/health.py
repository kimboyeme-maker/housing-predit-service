"""Health check endpoint."""

import time

from fastapi import APIRouter

from app.ml.loader import get_bundle
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    description=(
        "Returns `ok` when the regression model artifacts are loaded and ready to serve "
        "predictions, or `degraded` if they are missing/corrupt. Also reports the loaded "
        "model version and process uptime. Used by Docker/compose healthchecks."
    ),
)
def health() -> HealthResponse:
    """Return service readiness, model version, and process uptime."""
    bundle = get_bundle()
    return HealthResponse(
        status="ok" if bundle.is_loaded else "degraded",
        model_loaded=bundle.is_loaded,
        model_version=bundle.version,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
    )
