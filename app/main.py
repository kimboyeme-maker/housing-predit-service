"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.ml.loader import load_bundle
from app.routes import health, model_info, predict

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("app.main")

DESCRIPTION = """
Regression API that predicts **housing prices** from property features using a
scikit-learn `LinearRegression` model.

### Endpoints
* `POST /predict` — single or batch price predictions (array in, array out)
* `GET /model-info` — coefficients, intercept and performance metrics
* `GET /health` — liveness/readiness

Every response carries an `X-Request-ID` header; errors use a uniform envelope with a
stable `HPP-xxxx` code.
"""

TAGS_METADATA = [
    {"name": "predict", "description": "Run price predictions."},
    {"name": "model", "description": "Inspect model coefficients and metrics."},
    {"name": "health", "description": "Service health."},
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model artifacts before serving while allowing degraded health mode."""
    bundle = load_bundle()
    if bundle.is_loaded:
        logger.info("startup complete", extra={"model_version": bundle.version})
    else:
        logger.error("startup: model NOT loaded — serving degraded", extra={"reason": bundle.error})
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose the gateway headers so browser fetch() can read them cross-origin.
    expose_headers=["X-Request-ID", "X-Error-Code", "X-Error-Message"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
def root():
    """Expose lightweight service discovery without duplicating Swagger metadata."""
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}
