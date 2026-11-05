import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_compile_valid_report(client):
    response = client.post(
        "/compile",
        json={
            "report_text": "IDH1 p.R132H cannot be excluded.",
            "report_id": "TEST_001",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["packet_type"] == "compact_ai"
    assert data["report_id"] == "TEST_001"
    assert len(data["facts"]) == 1

    fact = data["facts"][0]

    assert fact["entity"] == "IDH1 p.R132H"
    assert fact["assertion"] == "POSSIBLE"
    assert fact["experiencer"] == "PATIENT"
    assert fact["temporality"] == "CURRENT"
    assert fact["certainty"] == "SUSPECTED"


def test_compile_empty_report(client):
    response = client.post(
        "/compile",
        json={
            "report_text": "",
            "report_id": "TEST_002",
        },
    )

    assert response.status_code == 422