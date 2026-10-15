from fastapi.testclient import TestClient
from backend.app.main import app

def test_health():
    with TestClient(app) as client:
        r=client.get("/api/health")
        assert r.status_code==200
        assert r.json()["status"]=="ok"

def test_results():
    with TestClient(app) as client:
        r=client.get("/api/results/comparison")
        assert r.status_code==200
        assert any(x["algorithm"]=="FedProx + DP" for x in r.json())
