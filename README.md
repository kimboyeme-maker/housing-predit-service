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
- Every response carries an **`X-Request-ID`** header. Safe caller IDs (`[A-Za-z0-9._:-]`, max
  128 characters) are echoed; missing/invalid values become a canonical UUIDv4. Request IDs are
  transport metadata and never appear in business response bodies.
- Errors use a uniform envelope **and** gateway headers the frontend consumes:
  ```json
  { "error": { "code": "HPP-1001", "message": "...", "details": [...] } }
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

`git push` to `main` runs CI (lint, retraining, tests, and the container build). A successful
workflow publishes these tags to GitHub Container Registry (GHCR):

```text
ghcr.io/kimboyeme-maker/housing-predit-service:latest
ghcr.io/kimboyeme-maker/housing-predit-service:<commit-sha>
```

The `latest` tag tracks the newest successful `main` build. A commit-SHA tag is immutable and is
the safer rollback target.

### Pull the private image

Public packages need no login. For a private package, create a GitHub classic PAT with
`read:packages`, then authenticate without putting the token in shell history:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u kimboyeme-maker --password-stdin
docker pull ghcr.io/kimboyeme-maker/housing-predit-service:latest
```

### Run FastAPI on the VPS

The container listens on `8000`. Bind it to VPS loopback because Caddy is the only public entry
point:

```bash
docker rm -f housing-price-api 2>/dev/null || true

docker run -d \
  --name housing-price-api \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e HPP_LOG_LEVEL=INFO \
  -e HPP_CORS_ORIGINS='https://your-site.netlify.app' \
  ghcr.io/kimboyeme-maker/housing-predit-service:latest
```

Verify the container before configuring public routing:

```bash
docker ps
docker logs --tail 100 housing-price-api
curl --fail http://127.0.0.1:8000/health
```

Expected port output is `127.0.0.1:8000->8000/tcp`. If Docker shows
`0.0.0.0:8000->8000/tcp`, the API is exposed on every interface instead of only through Caddy.
The left side of `-p HOST:CONTAINER` is the VPS port: for example, `-p 8080:8000` exposes VPS
port 8080, while `-p 8000:8000` exposes VPS port 8000.

### HTTPS for `.app` domains

Browsers preload `.app` domains with HSTS and force HTTPS. Uvicorn serves plain HTTP, so a URL
such as `https://hk.vps.kaeo.app:8080` cannot complete a TLS handshake. Caddy must terminate TLS
on port 443 and proxy to Uvicorn.

Before starting Caddy:

1. Point the domain A/AAAA records to the VPS.
2. Ensure ports 80 and 443 reach the VPS (including cloud-provider security groups).
3. Keep the Docker API bound to `127.0.0.1:8000`.

Use this `/etc/caddy/Caddyfile` to retain both direct backend routes and `/api`-prefixed routes:

```caddyfile
hk.vps.kaeo.app {
    # Netlify uses /api/health; strip /api before forwarding to FastAPI /health.
    handle_path /api/* {
        reverse_proxy 127.0.0.1:8000
    }

    # Preserve /health, /docs, /redoc, and /openapi.json for direct diagnostics.
    handle {
        reverse_proxy 127.0.0.1:8000
    }
}
```

Validate and reload Caddy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl enable caddy
sudo systemctl status caddy
sudo journalctl -u caddy -n 100 --no-pager
```

The fallback `handle` is required for Swagger. The HTML returned from `/api/docs` loads the schema
from `/openapi.json`; without the fallback route, Swagger opens but remains unable to load its API
definition. Prefer the canonical direct documentation URL `/docs`.

### Firewall

Allow SSH before modifying firewall rules. Only Caddy needs public ports; Docker's loopback port
does not need a UFW rule:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw delete allow 8000/tcp 2>/dev/null || true
sudo ufw delete allow 8080/tcp 2>/dev/null || true
sudo ufw status
```

Docker-published ports can interact with firewall rules differently across distributions. Binding
the application to `127.0.0.1` avoids relying on UFW to protect the backend port.

### Public verification

Both route forms intentionally work:

```bash
curl --fail https://hk.vps.kaeo.app/health
curl --fail https://hk.vps.kaeo.app/api/health
curl --fail https://hk.vps.kaeo.app/openapi.json
```

Public endpoints:

| Purpose | Direct path | Netlify-prefixed path |
| ------- | ----------- | --------------------- |
| Health | `/health` | `/api/health` |
| Model metadata | `/model-info` | `/api/model-info` |
| Prediction | `/predict` | `/api/predict` |
| Swagger | `/docs` | `/api/docs` |
| ReDoc | `/redoc` | `/api/redoc` |

### Update and rollback

Pulling a tag does not replace a running container. Recreate it after pulling:

```bash
docker pull ghcr.io/kimboyeme-maker/housing-predit-service:latest
docker rm -f housing-price-api
# Repeat the docker run command from "Run FastAPI on the VPS".
```

For rollback, replace `latest` in that command with a previously published commit SHA tag.

### Troubleshooting

```bash
# Container health, port mapping, and recent application logs
docker ps
docker inspect --format '{{json .State.Health}}' housing-price-api
docker logs --tail 200 housing-price-api

# Local listener and direct backend checks
sudo ss -lntp | grep -E ':80|:443|:8000|:8080'
curl -v http://127.0.0.1:8000/health

# Caddy routing and certificate diagnostics
sudo caddy validate --config /etc/caddy/Caddyfile
sudo journalctl -u caddy -n 200 --no-pager
curl -v https://hk.vps.kaeo.app/api/health
```

- `connection refused` on `127.0.0.1:8080` means Docker did not publish host port 8080. Check the
  `PORTS` column in `docker ps`.
- TLS handshake failure on `https://domain:8080` means plain Uvicorn HTTP was contacted as HTTPS.
- A pending browser request usually indicates unreachable TLS/port/proxy routing; a wrong API path
  normally returns a fast 404 instead.
- A Swagger page that opens but cannot load endpoints usually means `/openapi.json` is not routed.
