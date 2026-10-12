"""
Unit test for Two-Sided Closed Loop customer response endpoint (/api/customer/respond)
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_customer_response_approve():
    payload = {
        "transaction_id": "GDN-TX-TEST-01",
        "response": "Confirmed by cardholder (Legitimate)",
        "review_status": "Customer Verified",
        "account_id": "DEMO_ACC_104"
    }
    r = client.post("/api/customer/respond", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["transaction_id"] == "GDN-TX-TEST-01"
    assert data["response"] == "Confirmed by cardholder (Legitimate)"
    assert data["review_status"] == "Customer Verified"
    assert data["loop_closed"] is True

def test_customer_response_freeze():
    payload = {
        "transaction_id": "GDN-TX-TEST-02",
        "response": "Rejected by cardholder (Fraud Confirmed)",
        "review_status": "Card Frozen",
        "account_id": "DEMO_ACC_104"
    }
    r = client.post("/api/customer/respond", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["review_status"] == "Card Frozen"
    assert data["loop_closed"] is True
