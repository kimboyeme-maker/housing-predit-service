"""Train the housing-price regression model and write committed artifacts.

USER SEAM
=========
This is the one file to edit if you want to change the model. Its only contract is:
it must produce `app/ml/artifacts/model.pkl` (a fitted estimator with `.predict`) and
`app/ml/artifacts/metadata.json` matching the shape the API expects (see METADATA below).
The FastAPI service loads those two files at startup and needs no code changes.

Run:  uv run python train.py   (or: python train.py)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

FEATURES = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]
TARGET = "price"

ROOT = Path(__file__).resolve().parent
DATASET = Path(os.environ.get("HPP_DATASET", ROOT / "data" / "House Price Dataset.csv"))
ARTIFACT_DIR = ROOT / "app" / "ml" / "artifacts"
MODEL_VERSION = "1"


def main() -> None:
    """Train, evaluate, and write the deployable model artifacts."""
    df = pd.read_csv(DATASET)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, preds)), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, preds)), 2),
        "cv_r2_mean": round(
            float(cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()), 4
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    coefficients = {f: round(float(c), 4) for f, c in zip(FEATURES, model.coef_, strict=True)}
    feature_stats = {
        f: {
            "min": round(float(X[f].min()), 2),
            "max": round(float(X[f].max()), 2),
            "mean": round(float(X[f].mean()), 2),
        }
        for f in FEATURES
    }

    metadata = {
        "model_type": "LinearRegression",
        "target": TARGET,
        "features": FEATURES,
        "coefficients": coefficients,
        "intercept": round(float(model.intercept_), 4),
        "metrics": metrics,
        "feature_stats": feature_stats,
        "trained_at": datetime.now(UTC).isoformat(),
        "version": MODEL_VERSION,
        "dataset_rows": int(len(df)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved model.pkl + metadata.json to {ARTIFACT_DIR}")
    print(f"Metrics: {json.dumps(metrics)}")
    # Sanity: refuse to silently ship a broken model.
    if not np.isfinite(model.intercept_) or metrics["r2"] < 0:
        raise SystemExit("Model looks invalid (non-finite intercept or negative R²).")


if __name__ == "__main__":
    main()
