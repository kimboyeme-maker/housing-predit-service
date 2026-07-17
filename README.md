# Housing Price Prediction API

FastAPI + scikit-learn (`LinearRegression`) service that predicts housing prices from property
features. Ships with a committed, pre-trained model and rich Swagger docs.

## Endpoints

| Method | Path          | Description                                             |
| ------ | ------------- | ------------------------------------------------------ |
| POST   | `/predict`    | Predict prices — **array in, array out** (single/batch) |
| GET    | `/model-info` | Coefficients, intercept and performance metrics        |
| GET    | `/health`     | Liveness / readiness + loaded model version            |

Interactive docs: **`/docs`** (Swagger) · **`/redoc`**.

### Contract highlights

- `POST /predict` always takes a JSON **array** of feature objects and returns an array of
  predictions (a single prediction is a one-element array).
- Every response carries an **`X-Request-ID`** header (echoed from the request if supplied).
- Errors use a uniform envelope **and** gateway headers the frontend consumes:
  ```json
  { "error": { "code": "HPP-1001", "message": "...", "details": [...] }, "requestId": "..." }
  ```
  Headers: `X-Error-Code`, `X-Error-Message`, `X-Request-ID`.

| Code       | Meaning                    | HTTP |
| ---------- | -------------------------- | ---- |
| `HPP-1001` | Validation error           | 422  |
| `HPP-1002` | Model not loaded           | 503  |
| `HPP-1003` | Inference failed           | 500  |
| `HPP-1004` | Not found                  | 404  |
| `HPP-5000` | Internal error             | 500  |

Logs are structured JSON with a per-request `requestId` (access + audit lines).

## Features

`square_footage, bedrooms, bathrooms, year_built, lot_size, distance_to_city_center, school_rating`
→ target `price`.

## Local development

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 fetched automatically).

```bash
uv sync --extra dev
uv run python train.py          # produces app/ml/artifacts/{model.pkl,metadata.json}
uv run uvicorn app.main:app --reload
# open http://localhost:8000/docs
```

Run tests / lint:

```bash
uv run pytest -q
uv run ruff check .
```

## The model seam

`train.py` is the one file to edit to change the model. Its only contract is to emit
`app/ml/artifacts/model.pkl` (a fitted estimator with `.predict`) and `metadata.json` in the
documented shape. The API loads those at startup — **no code changes needed** to swap models.
Artifacts are committed to git; regenerate them with the **Retrain Model** GitHub Action
(`workflow_dispatch`), which commits the refreshed files back.

## Docker

Multi-stage build (deps resolved with uv in a builder stage, slim non-root runtime). The
pre-trained artifacts are copied in — no training at build time.

```bash
docker compose up --build      # http://localhost:8000
```

## Deployment (VPS)

`git push` to `main` runs CI (lint + retrain + tests) and, on success, builds and pushes the
image to `ghcr.io/kimboyeme-maker/housing-price-predit`. Pull and run it on the VPS via
`docker compose up -d` (or `docker run -p 8000:8000 <image>`).
