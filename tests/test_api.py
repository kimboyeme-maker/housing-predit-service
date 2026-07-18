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
    """Healthy artifacts must produce a traceable readiness response."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "X-Request-ID" in r.headers


def test_model_info():
    """Model metadata must expose all trained feature coefficients."""
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "LinearRegression"
    assert len(body["features"]) == 7
    assert set(body["coefficients"]) == set(body["features"])
    assert "r2" in body["metrics"]


def test_predict_single():
    """A one-row batch must produce one positive prediction."""
    r = client.post("/predict", json=[VALID])
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["price"] > 0
    assert "requestId" not in body
    assert r.headers["X-Request-ID"]


def test_predict_batch():
    """Batch cardinality must be preserved by the API."""
    r = client.post("/predict", json=[VALID, VALID])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_single_and_batch_paths_produce_equivalent_predictions():
    """A/B contract check: batching must not change a row's prediction."""
    single = client.post("/predict", json=[VALID]).json()["predictions"][0]["price"]
    batch = client.post("/predict", json=[VALID, VALID]).json()["predictions"]
    assert [item["price"] for item in batch] == [single, single]


def test_predict_request_id_echoed():
    """Safe caller correlation IDs must survive header and body propagation."""
    r = client.post("/predict", json=[VALID], headers={"X-Request-ID": "test-123"})
    assert r.headers["X-Request-ID"] == "test-123"
    assert "requestId" not in r.json()


@pytest.mark.parametrize("request_id", ["", "contains spaces", "x" * 129, "line\nbreak"])
def test_invalid_request_id_is_replaced(request_id):
    """Unsafe or oversized correlation IDs must be replaced with UUIDv4."""
    r = client.get("/health", headers={"X-Request-ID": request_id})
    generated = r.headers["X-Request-ID"]
    assert generated != request_id
    assert len(generated) == 36


@pytest.mark.parametrize(
    "field,value",
    [("school_rating", 99), ("bedrooms", -1), ("year_built", 1000)],
)
def test_predict_validation_error(field, value):
    """Domain violations must use the stable validation error code."""
    bad = {**VALID, field: value}
    r = client.post("/predict", json=[bad])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "HPP-1001"


def test_empty_batch_rejected():
    """Prediction entry guard must reject empty work units."""
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
    assert "requestId" not in r.json()


def test_not_found_uses_uniform_error_contract():
    """Framework 404 responses must use the same traceable error envelope."""
    r = client.get("/missing")
    assert r.status_code == 404
    assert r.headers["X-Error-Code"] == "HPP-1004"
    assert r.headers["X-Request-ID"]
    assert "requestId" not in r.json()
