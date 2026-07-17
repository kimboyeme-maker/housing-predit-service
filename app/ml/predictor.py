"""Inference wrapper: entry-guard validation + batch prediction."""

from __future__ import annotations

import numpy as np

from app.core.errors import ErrorCode, HppError
from app.core.logging import get_logger
from app.ml.loader import ModelBundle, get_bundle

logger = get_logger("app.ml.predictor")

# Hard domain bounds independent of the training data (sanity guards).
_HARD_BOUNDS = {
    "square_footage": (100, 100_000),
    "bedrooms": (0, 30),
    "bathrooms": (0, 30),
    "year_built": (1800, 2100),
    "lot_size": (0, 1_000_000),
    "distance_to_city_center": (0, 500),
    "school_rating": (0, 10),
}


def _validate(rows: list[dict], bundle: ModelBundle) -> None:
    """Reject out-of-domain feature values before they reach the model."""
    details: list[dict] = []
    for i, row in enumerate(rows):
        for feat, (lo, hi) in _HARD_BOUNDS.items():
            val = row.get(feat)
            if val is None:
                continue
            if not (lo <= val <= hi):
                details.append(
                    {
                        "index": i,
                        "field": feat,
                        "value": val,
                        "msg": f"{feat} must be between {lo} and {hi}",
                    }
                )
    if details:
        raise HppError(ErrorCode.VALIDATION, "Feature values out of allowed range", details)


def predict(rows: list[dict]) -> list[float]:
    """Predict prices for a batch of feature dicts. Always returns a list."""
    bundle = get_bundle()
    if not bundle.is_loaded:
        raise HppError(
            ErrorCode.MODEL_NOT_LOADED,
            "Prediction model is not available",
            details=[{"reason": bundle.error}] if bundle.error else [],
        )

    _validate(rows, bundle)

    features = bundle.features
    try:
        # Build the matrix in the exact feature order the model was trained on.
        matrix = np.array([[row[f] for f in features] for row in rows], dtype=float)
        preds = bundle.model.predict(matrix)
        results = [round(float(p), 2) for p in preds]
    except HppError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("inference failed")
        raise HppError(ErrorCode.INFERENCE_FAILED, "Model inference failed") from exc

    logger.info("prediction", extra={"n_inputs": len(rows), "version": bundle.version})
    return results
