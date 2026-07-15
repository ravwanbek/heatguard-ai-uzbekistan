from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cities() -> None:
    response = client.get("/cities")
    assert response.status_code == 200
    assert len(response.json()) == 12


def test_unknown_city() -> None:
    assert client.get("/weather/unknown").status_code == 404


def test_methodology_has_thresholds() -> None:
    response = client.get("/methodology")
    assert response.status_code == 200
    assert response.json()["thresholds"]["Extreme"] == "80–100"
