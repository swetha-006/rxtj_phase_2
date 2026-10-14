"""
Pytest configuration and test isolation fixtures.
Ensures pytest runs against an isolated temporary directory so that
test runs never modify the runtime databases on disk (Issue 9 in SECURITY_REVIEW.md).
"""
import os
import shutil
import tempfile
import pytest

_TEST_DIR = tempfile.mkdtemp(prefix="guardian_pytest_")
_SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Isolated test database paths
_TEST_PROFILE_DB = os.path.join(_TEST_DIR, "behavioral_profiles.db")
_TEST_ACCOUNTS_DB = os.path.join(_TEST_DIR, "guardian_accounts.db")
_TEST_HISTORY_DB = os.path.join(_TEST_DIR, "history.db")

# Seed the test profiles database if a seed file exists
_SEED_PROFILE_DB = os.path.join(_SEED_DIR, "behavioral_profiles.db")
if os.path.exists(_SEED_PROFILE_DB):
    shutil.copy2(_SEED_PROFILE_DB, _TEST_PROFILE_DB)

os.environ["RXTJ_PROFILE_DB"] = _TEST_PROFILE_DB
os.environ["GUARDIAN_ACCOUNTS_DB"] = _TEST_ACCOUNTS_DB
os.environ["RXTJ_HISTORY_DB"] = _TEST_HISTORY_DB
os.environ.setdefault("GUARDIAN_DEMO_MODE", "1")
os.environ.setdefault("GUARDIAN_COOKIE_SECURE", "0")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_sandbox():
    yield
    try:
        shutil.rmtree(_TEST_DIR, ignore_errors=True)
    except Exception:
        pass
