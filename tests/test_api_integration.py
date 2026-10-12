"""
Integration tests for RXT-J+ v4 API endpoints, categorical handling, and module wiring.
"""
import pytest
from fastapi.testclient import TestClient
import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app.app)


def test_categorical_and_missing_device_encodings(client):
    """Verify that scoring endpoints never crash on diverse categorical and device inputs."""
    test_cases = [
        {"device_info": None, "product_cd": "W"},
        {"device_info": "", "product_cd": "C"},
        {"device_info": "MISSING", "product_cd": "H"},
        {"device_info": "Windows", "product_cd": "R"},
        {"device_info": "MacOS", "product_cd": "S"},
        {"device_info": "UnrecognizedCustomPhone999", "product_cd": "W"},
        {"device_info": "nan", "product_cd": "w"},
    ]
    for tc in test_cases:
        payload = {
            "transaction_id": "TX_CAT_TEST",
            "amount": 250.0,
            "hour": 14,
            **tc
        }
        res = client.post("/score/form", json=payload)
        assert res.status_code == 200, f"Failed on {tc}: {res.text}"
        data = res.json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0


def test_score_form_full_cost_and_ring_integration(client):
    """Verify that /score/form-full updates fraud ring graph and runs cost router."""
    # First transaction from account A with a shared device
    p1 = {
        "account_id": "ACC_INTEG_A",
        "transaction_id": "TX_INT_1",
        "amount": 80000.0,
        "hour": 3,
        "product_cd": "W",
        "device_info": "SharedTabletPro",
        "addr1": 412.0
    }
    r1 = client.post("/score/form-full", json=p1)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "cost_routing" in d1
    assert "ring_association" in d1
    assert d1["ring_association"]["ring_size"] >= 1
    # Large amount should trigger freeze/block
    assert d1["recommended_action"] in ("FREEZE_AND_NOTIFY", "STEP_UP_AUTHENTICATION")

    # Second transaction from different account B sharing the same device
    p2 = {
        "account_id": "ACC_INTEG_B",
        "transaction_id": "TX_INT_2",
        "amount": 200.0,
        "hour": 3,
        "product_cd": "W",
        "device_info": "SharedTabletPro",
        "addr1": 412.0
    }
    r2 = client.post("/score/form-full", json=p2)
    assert r2.status_code == 200
    d2 = r2.json()
    # Ring size should now reflect the linked syndicate
    assert d2["ring_association"]["ring_size"] >= 2


def test_dedicated_cost_route_endpoint(client):
    """Verify POST /cost/route matches blueprint expected cost arithmetic."""
    # ₹200 at p=0.30 -> STEP_UP
    r1 = client.post("/cost/route", json={"probability": 0.30, "amount": 200.0})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["action"] == "STEP_UP"
    assert d1["cost_step_up"] == 29.0

    # ₹80,000 at p=0.30 -> BLOCK
    r2 = client.post("/cost/route", json={"probability": 0.30, "amount": 80000.0})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["action"] == "BLOCK"


def test_dedicated_ring_endpoints(client):
    """Verify GET /ring/graph/{account_id} and /ring/metrics/{account_id}."""
    rg = client.get("/ring/graph/ACC_INTEG_A")
    assert rg.status_code == 200
    graph_data = rg.json()
    assert "nodes" in graph_data
    assert "links" in graph_data

    rm = client.get("/ring/metrics/ACC_INTEG_A")
    assert rm.status_code == 200
    metrics_data = rm.json()
    assert "ring_size" in metrics_data
    assert metrics_data["ring_size"] >= 2


def test_guardian_workspace_and_legacy_dashboard_routes(client):
    """Guardian is the default experience and the legacy dashboard remains available."""
    root = client.get("/")
    assert root.status_code == 200
    assert "RXT-J+" in root.text and "Guardian" in root.text
    assert 'id="page-simulator"' in root.text
    legacy = client.get("/panel")
    assert legacy.status_code == 200
    assert "FraudShield" in legacy.text or "RXT-J+" in legacy.text
