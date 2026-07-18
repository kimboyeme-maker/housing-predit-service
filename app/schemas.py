"""Pydantic request/response models with rich OpenAPI metadata (descriptions + examples)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """A single property's features. All fields are required."""

    square_footage: float = Field(
        ..., gt=0, description="Interior living area in square feet.", examples=[1550]
    )
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms.", examples=[3])
    bathrooms: float = Field(
        ..., ge=0, description="Number of bathrooms (half-baths allowed, e.g. 2.5).", examples=[2]
    )
    year_built: int = Field(
        ..., ge=1800, le=2100, description="Year the property was constructed.", examples=[1997]
    )
    lot_size: float = Field(..., ge=0, description="Lot size in square feet.", examples=[6800])
    distance_to_city_center: float = Field(
        ..., ge=0, description="Distance to the city centre in kilometres.", examples=[4.1]
    )
    school_rating: float = Field(
        ..., ge=0, le=10, description="Nearby school rating (0–10).", examples=[7.6]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "square_footage": 1550,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "year_built": 1997,
                    "lot_size": 6800,
                    "distance_to_city_center": 4.1,
                    "school_rating": 7.6,
                }
            ]
        }
    }


class PredictionItem(BaseModel):
    """One prediction paired with its normalized source features."""
    price: float = Field(
        ..., description="Predicted price in currency units.", examples=[241300.55]
    )
    inputs: HouseFeatures = Field(..., description="Echo of the input features for traceability.")


class PredictResponse(BaseModel):
    """Batch prediction envelope with model traceability."""
    model_config = {"protected_namespaces": ()}

    predictions: list[PredictionItem]
    model_version: str | None = Field(None, description="Version of the model that produced these.")


class FeatureStat(BaseModel):
    """Training distribution summary for a single feature."""
    min: float
    max: float
    mean: float


class Metrics(BaseModel):
    """Held-out and cross-validation metrics emitted during training."""
    r2: float
    mae: float
    rmse: float
    cv_r2_mean: float
    n_train: int
    n_test: int


class ModelInfoResponse(BaseModel):
    """Public model provenance, parameters, metrics, and feature ranges."""
    model_config = {"protected_namespaces": ()}

    model_type: str = Field(..., examples=["LinearRegression"])
    target: str = Field(..., examples=["price"])
    features: list[str]
    coefficients: dict[str, float] = Field(..., description="Learned weight per feature.")
    intercept: float
    metrics: Metrics
    feature_stats: dict[str, FeatureStat]
    trained_at: str
    version: str
    dataset_rows: int


class HealthResponse(BaseModel):
    """Liveness/readiness payload consumed by Docker and the portal."""
    model_config = {"protected_namespaces": ()}

    status: str = Field(
        ..., description="'ok' when model is loaded, else 'degraded'.", examples=["ok"]
    )
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ErrorDetail(BaseModel):
    """Machine code, safe user message, and optional structured context."""
    code: str = Field(..., examples=["HPP-1001"])
    message: str
    details: list[dict] = []


class ErrorEnvelope(BaseModel):
    """Uniform error shape returned for every non-2xx response."""

    error: ErrorDetail
