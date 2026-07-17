"""End-to-end API tests (assumes artifacts exist; run `python train.py` first)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID = {
    "square_footage": 1550,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1997,
    "lot_size": 6800,
    "distance_to_city_center": 4.1,
    "school_rating": 7.6,
}


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "X-Request-ID" in r.headers


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "LinearRegression"
    assert len(body["features"]) == 7
    assert set(body["coefficients"]) == set(body["features"])
    assert "r2" in body["metrics"]


def test_predict_single():
    r = client.post("/predict", json=[VALID])
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["price"] > 0
    assert body["requestId"]


def test_predict_batch():
    r = client.post("/predict", json=[VALID, VALID])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_request_id_echoed():
    r = client.post("/predict", json=[VALID], headers={"X-Request-ID": "test-123"})
    assert r.headers["X-Request-ID"] == "test-123"
    assert r.json()["requestId"] == "test-123"


@pytest.mark.parametrize(
    "field,value",
    [("school_rating", 99), ("bedrooms", -1), ("year_built", 1000)],
)
def test_predict_validation_error(field, value):
    bad = {**VALID, field: value}
    r = client.post("/predict", json=[bad])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "HPP-1001"


def test_empty_batch_rejected():
    r = client.post("/predict", json=[])
    assert r.status_code == 422


def test_error_gateway_headers():
    """Errors must carry X-Error-Code / X-Error-Message / X-Request-ID for the frontend."""
    bad = {**VALID, "school_rating": 99}
    r = client.post("/predict", json=[bad])
    assert r.status_code == 422
    assert r.headers["X-Error-Code"] == "HPP-1001"
    assert r.headers["X-Error-Message"]
    assert r.headers["X-Request-ID"]
