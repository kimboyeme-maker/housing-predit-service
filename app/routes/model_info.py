"""Model metadata + performance metrics endpoint."""

from fastapi import APIRouter

from app.core.errors import ErrorCode, HppError
from app.ml.loader import get_bundle
from app.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model coefficients and performance metrics",
    description=(
        "Returns the linear-regression model's learned **coefficients** (weight per feature) "
        "and **intercept**, the **performance metrics** measured on a held-out split "
        "(R², MAE, RMSE, cross-validated R²), per-feature training statistics used for the UI "
        "and range hints, and provenance (trained_at, version, dataset size)."
    ),
)
def model_info() -> ModelInfoResponse:
    bundle = get_bundle()
    if not bundle.is_loaded:
        raise HppError(
            ErrorCode.MODEL_NOT_LOADED,
            "Model metadata is not available",
            details=[{"reason": bundle.error}] if bundle.error else [],
        )
    return ModelInfoResponse(**bundle.metadata)
