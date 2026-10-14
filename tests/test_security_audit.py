"""
Security and Integrity Remediations Unit Tests
Verifies fixes for Issues 11, 12, and 13 from SECURITY_REVIEW.md:
  - Issue 11: Demo Ring Re-seeded at Module Import & SQLite graph persistence
  - Issue 12: Model artifact integrity verification (SHA-256) & safe loading
  - Issue 13: Pinned requirements and dev dependencies separation
"""
import os
import tempfile
import pytest
from ring_detector import FraudRingDetector
from app import verify_model_integrity, MODEL_HASHES


def test_ring_sqlite_persistence_and_multiworker_recovery():
    """Verify that transactions are persisted to SQLite and a new worker instance recovers the graph (Issue 11)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        # Worker 1 writes syndicate transactions
        worker1 = FraudRingDetector(db_path=db_path)
        worker1.add_transaction("WORKER_ACC_A", device_info="SharedPOS_1", addr1="550", timestamp=1000.0)
        worker1.add_transaction("WORKER_ACC_B", device_info="SharedPOS_1", addr1="550", timestamp=1050.0)
        m1 = worker1.get_account_ring("WORKER_ACC_A")
        assert m1.ring_size == 2

        # Worker 2 starts up fresh with the same SQLite db and rebuilds graph
        worker2 = FraudRingDetector(db_path=db_path)
        assert worker2.get_account_ring("WORKER_ACC_A").ring_size == 1  # Before loading, in-memory is empty

        loaded = worker2.load_from_db()
        assert loaded == 2

        m2 = worker2.get_account_ring("WORKER_ACC_A")
        assert m2.ring_size == 2
        assert set(m2.connected_accounts) == {"WORKER_ACC_A", "WORKER_ACC_B"}
        assert "SharedPOS_1" in m2.shared_devices

        # Reset with clear_db=True purges SQLite storage
        worker2.reset(clear_db=True)
        worker3 = FraudRingDetector(db_path=db_path)
        assert worker3.load_from_db() == 0
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_model_artifact_integrity_verification(tmp_path, monkeypatch):
    """Verify SHA-256 integrity check catches corrupted or tampered model artifacts (Issue 12)."""
    # 1. Existing valid model passes
    lgbm_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lgbm.pkl")
    if os.path.exists(lgbm_file):
        # Should not raise
        verify_model_integrity(lgbm_file)

    # 2. Tampered file with same name triggers RuntimeError
    fake_model = tmp_path / "lgbm.pkl"
    fake_model.write_text("malicious_tampered_payload")

    monkeypatch.setenv("GUARDIAN_VERIFY_MODEL_HASH", "1")
    with pytest.raises(RuntimeError, match="Model integrity check failed"):
        verify_model_integrity(str(fake_model))

    # 3. Bypass via GUARDIAN_VERIFY_MODEL_HASH=0 allowed for retraining environments
    monkeypatch.setenv("GUARDIAN_VERIFY_MODEL_HASH", "0")
    # Should not raise when bypassed
    verify_model_integrity(str(fake_model))


def test_requirements_pinning_and_dev_separation():
    """Verify that requirements.txt contains only pinned production dependencies (Issue 13)."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    prod_req = os.path.join(base_dir, "requirements.txt")
    dev_req = os.path.join(base_dir, "requirements-dev.txt")

    assert os.path.exists(prod_req), "requirements.txt must exist"
    assert os.path.exists(dev_req), "requirements-dev.txt must exist"

    with open(prod_req, "r") as f:
        prod_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Production requirements must not contain testing or plotting tools
    prod_pkg_names = [line.split("==")[0].split(">=")[0].lower() for line in prod_lines]
    assert "pytest" not in prod_pkg_names
    assert "httpx" not in prod_pkg_names
    assert "matplotlib" not in prod_pkg_names

    # All production dependencies must be pinned with ==
    for line in prod_lines:
        assert "==" in line, f"Production dependency not pinned: {line}"

    # Dev requirements must contain pytest and httpx
    with open(dev_req, "r") as f:
        dev_content = f.read()
    assert "pytest" in dev_content
    assert "httpx" in dev_content
