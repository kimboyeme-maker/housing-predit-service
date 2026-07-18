"""Model artifact loader.

Loads the committed `model.pkl` + `metadata.json` once at startup and holds them
in a process-wide singleton. This is the seam users extend: swap in your own
artifacts (produced by any train.py) and the API serves them with no code change,
as long as metadata.json keeps the documented shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("app.ml.loader")


@dataclass
class ModelBundle:
    """In-memory estimator plus metadata or a captured startup failure."""
    model: Any | None = None
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Report readiness only when both estimator and metadata are present."""
        return self.model is not None and bool(self.metadata)

    @property
    def version(self) -> str | None:
        """Return artifact version for responses and audit logs."""
        return self.metadata.get("version")

    @property
    def features(self) -> list[str]:
        """Preserve training feature order for inference matrix construction."""
        return list(self.metadata.get("features", []))

    @property
    def feature_stats(self) -> dict[str, dict]:
        """Expose training ranges used by clients for hints and validation."""
        return self.metadata.get("feature_stats", {})


_bundle: ModelBundle | None = None


def load_bundle(force: bool = False) -> ModelBundle:
    """Load (once) and return the model bundle. Never raises — failures are captured."""
    global _bundle
    if _bundle is not None and not force:
        return _bundle

    settings = get_settings()
    model_path = Path(settings.model_path)
    metadata_path = Path(settings.metadata_path)

    try:
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"missing artifacts: model={model_path.exists()} metadata={metadata_path.exists()}"
            )
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text())
        _bundle = ModelBundle(model=model, metadata=metadata)
        logger.info(
            "model loaded", extra={"version": _bundle.version, "features": len(_bundle.features)}
        )
    except Exception as exc:  # noqa: BLE001 — startup must never crash on bad artifacts
        _bundle = ModelBundle(error=str(exc))
        logger.error("failed to load model artifacts: %s", exc)

    return _bundle


def get_bundle() -> ModelBundle:
    """Return the lazily loaded process-wide model bundle."""
    return load_bundle()
