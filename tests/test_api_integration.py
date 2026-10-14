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


def test_cors_origin_hardening(client):
    """Verify CORS blocks untrusted external origins and accepts configured local origin."""
    res_bad = client.options("/health", headers={
        "Origin": "https://malicious-attacker.com",
        "Access-Control-Request-Method": "GET"
    })
    assert res_bad.headers.get("access-control-allow-origin") != "https://malicious-attacker.com"
    assert res_bad.headers.get("access-control-allow-origin") != "*"

    res_good = client.options("/health", headers={
        "Origin": "http://localhost:8000",
        "Access-Control-Request-Method": "GET"
    })
    assert res_good.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_ring_reset_authentication_protection(client, monkeypatch):
    """Verify /ring/reset rejects unauthenticated requests and requires credentials."""
    # 1. Unauthenticated request must be rejected with 401
    r_unauth = client.post("/ring/reset")
    assert r_unauth.status_code == 401
    assert "sign in" in r_unauth.json()["detail"].lower()

    # 2. Authenticated user can reset
    reg = client.post("/auth/register", json={
        "email": "analyst_reset@bank.com",
        "display_name": "Reset Analyst",
        "password": "Password12345!"
    })
    if reg.status_code == 201:
        cookie = reg.cookies.get("guardian_session")
        client.cookies.set("guardian_session", cookie)
    else:
        # If user already exists from previous run, login
        login_res = client.post("/auth/login", json={
            "email": "analyst_reset@bank.com",
            "password": "Password12345!"
        })
        cookie = login_res.cookies.get("guardian_session")
        client.cookies.set("guardian_session", cookie)

    r_auth = client.post("/ring/reset")
    assert r_auth.status_code == 200
    assert r_auth.json()["status"] == "ok"

    # 3. If DEMO_MODE is disabled, returns 403
    monkeypatch.setattr(app, "DEMO_MODE", False)
    r_disabled = client.post("/ring/reset")
    assert r_disabled.status_code == 403
    client.cookies.clear()


def test_unauthenticated_profile_and_history_endpoints_blocked(client, monkeypatch):
    """Verify Issue 3: /history, /account/profile, /account/history, /account/alerts require auth."""
    client.cookies.clear()

    # 1. Anonymous requests must be blocked with 401
    assert client.get("/history").status_code == 401
    assert client.get("/account/profile/ACC_TEST_42").status_code == 401
    assert client.get("/account/history/ACC_TEST_42").status_code == 401
    assert client.get("/account/alerts").status_code == 401
    assert client.post("/account/explain", json={
        "account_id": "ACC_TEST_42",
        "transaction_id": "TX_TEST_001"
    }).status_code == 401

    # 2. Authenticated requests with valid session or X-Admin-Key succeed
    monkeypatch.setenv("GUARDIAN_ADMIN_KEY", "secret-ci-admin-key")
    admin_headers = {"X-Admin-Key": "secret-ci-admin-key"}

    r_hist = client.get("/history?limit=5", headers=admin_headers)
    assert r_hist.status_code == 200
    assert isinstance(r_hist.json(), list)

    r_alerts = client.get("/account/alerts", headers=admin_headers)
    assert r_alerts.status_code == 200
    assert "alert_count" in r_alerts.json()

    r_acct_hist = client.get("/account/history/ACC_INTEG_A", headers=admin_headers)
    assert r_acct_hist.status_code == 200
    assert r_acct_hist.json()["account_id"] == "ACC_INTEG_A"


def test_batch_scoring_bounds_and_dos_protection(client, monkeypatch):
    """Verify Issue 4: BatchRequest enforces size bounds (1-500) and requires authentication."""
    client.cookies.clear()

    # 1. Unauthenticated batch scoring is rejected
    r_unauth = client.post("/score/batch", json={
        "transactions": [{"transaction_id": "TX_1", "features": [0.0] * 14}]
    })
    assert r_unauth.status_code == 401

    # Configure admin auth header
    monkeypatch.setenv("GUARDIAN_ADMIN_KEY", "secret-ci-admin-key")
    admin_headers = {"X-Admin-Key": "secret-ci-admin-key"}

    # 2. Empty batch list rejected by Pydantic validation (min_length=1)
    r_empty = client.post("/score/batch", json={"transactions": []}, headers=admin_headers)
    assert r_empty.status_code == 422

    # 3. Oversized batch (>500 items) rejected by Pydantic validation (max_length=500)
    oversized_payload = {
        "transactions": [
            {"transaction_id": f"TX_{i}", "features": [0.0] * 14}
            for i in range(501)
        ]
    }
    r_oversized = client.post("/score/batch", json=oversized_payload, headers=admin_headers)
    assert r_oversized.status_code == 422

    # 4. Valid batch within limits (1-500) successfully scores
    valid_payload = {
        "transactions": [
            {"transaction_id": "TX_BATCH_1", "features": [0.0] * 14},
            {"transaction_id": "TX_BATCH_2", "features": [1.5] * 14}
        ]
    }
    r_valid = client.post("/score/batch", json=valid_payload, headers=admin_headers)
    assert r_valid.status_code == 200
    data = r_valid.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "fraud_count" in data
    assert "total_latency_ms" in data


def test_auth_rate_limiting_protects_brute_force(client, monkeypatch):
    """Verify Issue 5: Throttles rapid login and registration attempts with HTTP 429 Retry-After."""
    app.auth_limiter.reset()
    monkeypatch.setattr(app, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(app, "RATE_LIMIT_LOGIN_MAX", 3)
    monkeypatch.setattr(app, "RATE_LIMIT_REGISTER_MAX", 3)

    # 1. First 3 bad logins return 401
    for _ in range(3):
        r = client.post("/auth/login", json={"email": "victim@bank.com", "password": "wrong"})
        assert r.status_code == 401

    # 4th login attempt from same client must be rate-limited with 429
    r_limited = client.post("/auth/login", json={"email": "victim@bank.com", "password": "wrong"})
    assert r_limited.status_code == 429
    assert "too many login attempts" in r_limited.json()["detail"].lower()
    assert "retry-after" in r_limited.headers

    # 2. Registrations from same IP are also throttled after 3 attempts
    for i in range(3):
        client.post("/auth/register", json={
            "email": f"spam_user_{i}@bank.com",
            "display_name": f"Spam User {i}",
            "password": "Password12345!"
        })

    # 4th registration attempt must be rate-limited with 429
    r_reg_limited = client.post("/auth/register", json={
        "email": "spam_user_overflow@bank.com",
        "display_name": "Spam User Overflow",
        "password": "Password12345!"
    })
    assert r_reg_limited.status_code == 429
    assert "too many registration requests" in r_reg_limited.json()["detail"].lower()
    assert "retry-after" in r_reg_limited.headers

    # Clean up
    app.auth_limiter.reset()


def test_sanitized_internal_error_responses(client, monkeypatch):
    """Verify Issue 6: Internal runtime exceptions are logged and never leak stack/file paths to clients."""
    # Simulate an internal crash with sensitive internal file paths and details
    def mock_broken_scorer(*args, **kwargs):
        raise RuntimeError("Internal SQLite corruption at /srv/secrets/master.key: memory dump 0xDEADBEEF")

    monkeypatch.setattr(app, "score_transaction_features", mock_broken_scorer)

    # Test /score/form
    res = client.post("/score/form", json={
        "transaction_id": "TX_CRASH_TEST",
        "amount": 500.0
    })
    assert res.status_code == 500
    detail = res.json()["detail"]
    assert detail == "Internal transaction scoring error."
    # Ensure sensitive runtime info is NOT leaked to caller
    assert "/srv/secrets" not in detail
    assert "master.key" not in detail
    assert "0xDEADBEEF" not in detail


def test_cookie_secure_by_default(monkeypatch):
    """Verify Issue 7: Session cookie is Secure by default, and configurable via env."""
    # 1. Without environment variable, helper defaults to True (secure)
    monkeypatch.delenv("GUARDIAN_COOKIE_SECURE", raising=False)
    assert app._is_cookie_secure() is True

    # 2. When explicitly disabled for local unencrypted dev, returns False
    monkeypatch.setenv("GUARDIAN_COOKIE_SECURE", "0")
    assert app._is_cookie_secure() is False

    # 3. When explicitly enabled, returns True
    monkeypatch.setenv("GUARDIAN_COOKIE_SECURE", "1")
    assert app._is_cookie_secure() is True




