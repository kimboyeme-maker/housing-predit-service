"""Prediction endpoint — request and response are always arrays."""

from fastapi import APIRouter, Body

from app.ml.loader import get_bundle
from app.ml.predictor import predict as run_predict
from app.schemas import ErrorEnvelope, HouseFeatures, PredictionItem, PredictResponse

router = APIRouter(tags=["predict"])

_EXAMPLE_BATCH = [
    {
        "square_footage": 1550,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1997,
        "lot_size": 6800,
        "distance_to_city_center": 4.1,
        "school_rating": 7.6,
    },
    {
        "square_footage": 2200,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "year_built": 2008,
        "lot_size": 9600,
        "distance_to_city_center": 7.0,
        "school_rating": 8.8,
    },
]


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict housing prices (single or batch)",
    description=(
        "Accepts a **JSON array** of one or more property feature sets and returns a predicted "
        "price for each, in the same order. A single prediction is just a one-element array. "
        "Inputs are validated against domain ranges; out-of-range values return `HPP-1001`. "
        "If the model is unavailable the endpoint returns `HPP-1002`."
    ),
    responses={
        422: {"model": ErrorEnvelope, "description": "Validation error (HPP-1001)"},
        503: {"model": ErrorEnvelope, "description": "Model not loaded (HPP-1002)"},
    },
)
def predict(
    items: list[HouseFeatures] = Body(..., min_length=1, examples=[_EXAMPLE_BATCH]),  # noqa: B008
) -> PredictResponse:
    """Validate and predict a non-empty batch while preserving input order."""
    rows = [it.model_dump() for it in items]
    prices = run_predict(rows)
    predictions = [
        PredictionItem(price=price, inputs=item) for price, item in zip(prices, items, strict=True)
    ]
    return PredictResponse(
        predictions=predictions,
        model_version=get_bundle().version,
    )
