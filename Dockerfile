# syntax=docker/dockerfile:1

########## Stage 1: builder — resolve deps into a venv (no training) ##########
FROM python:3.12-slim AS builder

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml ./
# Create the venv and install runtime deps only (no dev extras).
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

########## Stage 2: runtime — slim, non-root ##########
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# App code + the committed, pre-trained artifacts.
COPY app ./app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
