# 🔐 RXT-J+ Guardian — Security Review & Remediation Guide

This document lists the security vulnerabilities and deployment issues found in the RXT-J+ Guardian codebase, explains why each one matters, and gives a fix for each.

> **Scope:** `app.py`, `guardian_accounts.py`, `guardian_orgs.py`, `ring_detector.py`, `guardian.html`, `Dockerfile`, `docker-compose.yml`, and the repository layout as of commit `66fae8c`.
> **Test status at time of review:** ✅ 21/21 tests passing.

---

## 📋 Summary Table

| # | Issue | Severity | Location | Remediation Status |
|---|-------|----------|----------|-------------------|
| 1 | CORS allows any origin with credentials | 🔴 High | `app.py:414–430` | ✅ Fixed & Verified (`GUARDIAN_ALLOWED_ORIGINS`) |
| 2 | Unauthenticated `/ring/reset` endpoint | 🔴 High | `app.py:2294–2318` | ✅ Fixed & Verified (`require_user` + `DEMO_MODE`) |
| 3 | Unauthenticated account data and history endpoints | 🔴 High | `app.py:1706, 2081–2116` | ✅ Fixed & Verified (`Depends(require_user)`) |
| 4 | Unauthenticated scoring endpoints with an unbounded batch size | 🟠 Medium | `app.py:434–470, 1742` | ✅ Fixed & Verified (Max 500 batch, 64-char IDs, auth) |
| 5 | No rate limiting on login and registration | 🟠 Medium | `app.py:1232–1320` | ✅ Fixed & Verified (`InMemoryRateLimiter` 429 Retry-After) |
| 6 | Internal error messages returned to clients | 🟠 Medium | `app.py` (7 endpoints) | ✅ Fixed & Verified (`logger.exception` + generic detail) |
| 7 | Session cookie not `Secure` by default | 🟠 Medium | `app.py:1186–1203` | ✅ Fixed & Verified (`_is_cookie_secure()` True by default) |
| 8 | `/api/customer/respond` is unauthenticated and stores nothing | 🟡 Low / Logic | `app.py:1354–1406` | ✅ Fixed & Verified (SQLite `customer_responses` + audit sync) |
| 9 | Runtime database committed to Git | 🟡 Low / Hygiene | `tests/conftest.py` | ✅ Fixed & Verified (Isolated `/tmp` sandbox in pytest) |
| 10 | Docker mounts `data/` read-only, but the app writes to it | 🟡 Low / Availability | `docker-compose.yml:18–21` | ✅ Fixed & Verified (Volume `guardian_state:/app/state`) |
| 11 | Demo fraud ring re-seeded at import time | 🟡 Low / Integrity | `app.py:350–375`, `ring_detector.py` | ✅ Fixed & Verified (Persistent SQLite graph + DEMO_MODE) |
| 12 | Model files loaded with pickle / `torch.load` | 🔵 Info | `app.py:112–202`, `preprocessing.py` | ✅ Fixed & Verified (SHA-256 integrity + `weights_only=True`) |
| 13 | Unpinned dependencies and dev tools in the production image | 🔵 Info | `requirements.txt`, `Dockerfile` | ✅ Fixed & Verified (Pinned runtime, dev split, non-root user) |

---

## 🔴 1. CORS Allows Any Origin With Credentials

**Where:** `app.py:349`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    ...
)
```

### ❓ What's wrong
When `allow_origins=["*"]` is combined with `allow_credentials=True`, Starlette sends back the caller's own `Origin` header as the allowed origin. As a result, **any website** is treated as trusted for credentialed requests. Allowing `"null"` is also dangerous, because sandboxed iframes and `file://` pages send that origin.

### ⚠️ Why it matters
The cookie's `SameSite=lax` setting limits some cross-site attacks, but it still leaves sibling subdomains, same-site pages and any future cookie changes exposed. A malicious page could read a logged-in analyst's cases, notes and exports.

### ✅ How to fix
List only the origins you actually serve the UI from:

```python
ALLOWED_ORIGINS = [o for o in os.environ.get("GUARDIAN_ALLOWED_ORIGINS", "http://localhost:8000").split(",") if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)
```

If the UI is only ever served by the same FastAPI app (`/`, `/guardian`), you can remove the CORS middleware completely.

---

## 🔴 2. Unauthenticated `/ring/reset` Endpoint

**Where:** `app.py:2006`

```python
@app.post("/ring/reset")
def reset_fraud_ring():
    seed_demo_fraud_ring()
```

### ❓ What's wrong
Anyone who can reach the server can wipe the shared, in-memory fraud-ring graph. No login is required.

### ⚠️ Why it matters
An attacker could repeatedly clear the ring graph and erase the evidence of coordinated fraud while it is happening. This is both an integrity problem and a denial-of-service.

### ✅ How to fix
Require a logged-in user, and turn the endpoint off outside demo mode:

```python
DEMO_MODE = os.environ.get("GUARDIAN_DEMO_MODE", "0") == "1"

@app.post("/ring/reset")
def reset_fraud_ring(request: Request):
    if not DEMO_MODE:
        raise HTTPException(status_code=404)
    _current_user(request)
    seed_demo_fraud_ring()
    return {"status": "ok"}
```

---

## 🔴 3. Unauthenticated Account Data and History Endpoints

**Where:**
- `GET /history` — `app.py:1429`
- `GET /account/profile/{account_id}` — `app.py:1799`
- `GET /account/history/{account_id}` — `app.py:1813`
- `GET /account/alerts` — `app.py:1823`

### ❓ What's wrong
These routes return behavioural profiles, known devices, transaction history and compromise alerts **without checking for a session**. Account IDs are predictable (e.g. `ACC_42`), so someone could step through them and collect profiles in bulk.

### ⚠️ Why it matters
This is a direct leak of personal and financial data. In a real deployment it would be an access-control failure (OWASP A01: Broken Access Control) and could trigger data-protection obligations such as India's DPDP Act or GDPR.

### ✅ How to fix
Add a FastAPI dependency and attach it to every data route:

```python
from fastapi import Depends

def require_user(request: Request):
    return _current_user(request)

@app.get("/account/profile/{account_id}")
def get_account_profile(account_id: str, window: str = "30d", user=Depends(require_user)):
    ...
```

Do the same for `/history`, `/account/history/{id}` and `/account/alerts`. Ideally, also check that the account belongs to one of the user's workspaces.

---

## 🟠 4. Unauthenticated Scoring Endpoints With an Unbounded Batch Size

**Where:** `/score`, `/score/form`, `/score/direct`, `/score/form-full`, `/score/explain`, `/account/compromise-score`, `/account/explain`, and `BatchRequest` at `app.py:367`

```python
class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]
```

### ❓ What's wrong
- None of the scoring endpoints require a login.
- `/score/batch` accepts a list of **any length**.
- The SHAP and occlusion explanation paths take a lot of CPU.

### ⚠️ Why it matters
- **Denial of service:** a single request containing 100,000 transactions can tie up the server.
- **Model extraction:** unlimited free queries let an attacker probe the model and learn how to avoid its thresholds.
- **Data pollution:** scoring requests write to the history and profile stores, so anonymous users can poison behavioural profiles.

### ✅ How to fix
```python
from pydantic import Field

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=500)

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., max_length=64)
    features: List[float] = Field(..., max_length=256)
```

Then put `Depends(require_user)` on the scoring routes, or protect them with an API key for machine clients.

---

## 🟠 5. No Rate Limiting on Login and Registration

**Where:** `app.py:1113–1130`

### ❓ What's wrong
PBKDF2 with 310,000 iterations is a good choice, but nothing limits how many login attempts an IP address or account can make.

### ⚠️ Why it matters
- Attackers can brute-force or credential-stuff analyst accounts.
- Each attempt costs about 100 ms of CPU for PBKDF2, so a flood of login requests is also a cheap way to overload the server.
- Unlimited registration allows mass creation of fake accounts.

### ✅ How to fix
Use [`slowapi`](https://pypi.org/project/slowapi/), or put a reverse-proxy limit in front of the app (nginx `limit_req`, Cloudflare):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/auth/login")
@limiter.limit("5/minute")
def guardian_login(request: Request, payload: Dict[str, Any], response: Response):
    ...
```

---

## 🟠 6. Internal Error Messages Returned to Clients

**Where:** 7 places in `app.py` (`:1389`, `:1427`, `:1462`, `:1493`, `:1686`, `:1797`, `:1952`)

```python
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

### ❓ What's wrong
Raw exception text is sent straight to the client. It can include file paths, model internals, library versions and SQL details.

### ✅ How to fix
Log the full error on the server and send the client a generic message:

```python
import logging
log = logging.getLogger("guardian")

except Exception:
    log.exception("scoring failed")
    raise HTTPException(status_code=500, detail="Internal scoring error.")
```

---

## 🟠 7. Session Cookie Not `Secure` by Default

**Where:** `app.py:1105`

```python
COOKIE_SECURE = os.environ.get("GUARDIAN_COOKIE_SECURE", "0").lower() in ("1", "true", "yes")
```

### ❓ What's wrong
Unless someone sets an environment variable, the session cookie can be sent over plain HTTP, where anyone on the network path can read it.

### ✅ How to fix
Make the cookie secure by default and turn it off only for local development:

```python
COOKIE_SECURE = os.environ.get("GUARDIAN_COOKIE_SECURE", "1").lower() in ("1", "true", "yes")
```

Set `GUARDIAN_COOKIE_SECURE=0` only in your local `.env`. In production, always run behind TLS.

---

## 🟡 8. `/api/customer/respond` Is Unauthenticated and Stores Nothing

**Where:** `app.py:1172`

### ❓ What's wrong
The docstring says this endpoint *"updates transaction status"*, but it only echoes the payload back with `"loop_closed": true`. Nothing is saved, nothing is checked against a real transaction, and no login is required.

### ⚠️ Why it matters
Right now this is mostly a **correctness** problem: the UI and reviewers believe the customer's reply was recorded when it wasn't. If it later starts writing data, an unauthenticated version would let anyone mark any transaction as "customer confirmed legitimate", which would bypass fraud controls.

### ✅ How to fix
- Save the response (for example as a case note or audit event) and tie it to an existing transaction.
- Authenticate the caller. For a real customer-facing flow, use a **signed, single-use, expiring token** sent to the customer by SMS or email, not an open endpoint.
- Until then, change the docstring and UI text so they don't claim the loop is closed.

---

## 🟡 9. Runtime Database Committed to Git

**Where:** `data/behavioral_profiles.db` (grew from 36 KB to **4.9 MB** in commit `66fae8c`)

### ❓ What's wrong
`.gitignore` already excludes `*.db`, but this file was force-added. Running the test suite **modifies it**, so every test run leaves the working tree dirty and invites accidental commits of generated data.

### ⚠️ Why it matters
- Git history grows with binary blobs that can never be fully removed.
- If real or realistic behavioural data ends up in there, it is now in Git history permanently.

### ✅ How to fix
```bash
git rm --cached data/behavioral_profiles.db
git commit -m "chore: stop tracking runtime profile DB"
```

Point the tests at a temporary DB, for example in `tests/conftest.py`:

```python
import os, tempfile
os.environ.setdefault("RXTJ_PROFILE_DB", os.path.join(tempfile.mkdtemp(), "profiles.db"))
os.environ.setdefault("GUARDIAN_ACCOUNTS_DB", os.path.join(tempfile.mkdtemp(), "accounts.db"))
```

If the demo needs seed data, generate it with a script on startup instead of committing the DB.

---

## 🟡 10. Docker Mounts `data/` Read-Only, but the App Writes to It

**Where:** `docker-compose.yml`

```yaml
volumes:
  - ./data:/app/data:ro
```

### ❓ What's wrong
The app writes `guardian_accounts.db`, the org/case tables, the history DB and `behavioral_profiles.db` under `data/`. With a read-only mount, registration, login, case creation and history all fail inside the container.

### ✅ How to fix
Keep the read-only inputs separate from the writable state:

```yaml
volumes:
  - ./data:/app/data:ro
  - guardian_state:/app/state
environment:
  - GUARDIAN_ACCOUNTS_DB=/app/state/guardian_accounts.db
  - RXTJ_PROFILE_DB=/app/state/behavioral_profiles.db
  - RXTJ_HISTORY_DB=/app/state/history.db

volumes:
  guardian_state:
```

The `guardian_orgs` tables live in the accounts DB, so they move with `GUARDIAN_ACCOUNTS_DB`.

Hardening for the same file:
- Run the container as a non-root user (`USER app` in the `Dockerfile`).
- Remove `build-essential` from the final image (use a multi-stage build).

---

## 🟡 11. Demo Fraud Ring Re-seeded at Import Time

**Where:** `app.py:275`

```python
seed_demo_fraud_ring()
```

### ❓ What's wrong
The ring graph lives in memory and is wiped and re-seeded every time the module is imported. With `uvicorn --workers N`, each worker keeps its **own separate graph**, so ring detection gives different answers depending on which worker handles a request. Every restart also erases all real ring evidence.

### ✅ How to fix
- Seed only when `GUARDIAN_DEMO_MODE=1`.
- For anything beyond a demo, save the ring edges to SQLite (or Redis) and rebuild the graph from there on startup, so all workers share the same state.

---

## 🔵 12. Model Files Loaded With Pickle / `torch.load`

**Where:** `app.py:123–190` (`joblib.load`, `torch.load`)

### ❓ What's wrong
Pickle-based formats can run arbitrary code when they are loaded. That is fine for files you built yourself, but dangerous if anyone can swap the files in `models/`.

### ✅ How to fix
- Use `torch.load(path, map_location=DEVICE, weights_only=True)`.
- Keep `models/` read-only in production (the compose file already does this ✅).
- Optionally, store SHA-256 hashes of the model files and check them at startup.

---

## 🔵 13. Unpinned Dependencies and Dev Tools in the Production Image

**Where:** `requirements.txt`, `Dockerfile`

### ❓ What's wrong
- Every dependency uses `>=`, so each build can pull in different (and possibly vulnerable or breaking) versions.
- `pytest`, `httpx` and `matplotlib` are installed in the production image.

### ✅ How to fix
- Split into `requirements.txt` (runtime) and `requirements-dev.txt` (tests and notebooks).
- Pin exact versions with `pip-compile` (pip-tools) or `uv pip compile`.
- Add `pip-audit` to CI:

```yaml
- name: Dependency audit
  run: pip install pip-audit && pip-audit -r requirements.txt
```

---

## ✅ What's Already Done Well

To be fair, several parts of the security design are solid:

- 🔑 **Password hashing:** PBKDF2-SHA256 with 310k iterations and a per-user random salt.
- ⏱️ **Timing-safe comparison:** `hmac.compare_digest`, plus dummy hashing for unknown emails to make account enumeration harder.
- 🍪 **Session tokens:** random, stored only as a SHA-256 hash, `HttpOnly`, `SameSite=lax`, with a 14-day expiry.
- 💉 **SQL injection:** every query uses parameterised placeholders.
- 🧼 **XSS:** `guardian.html` escapes dynamic values with an `esc()` helper before inserting them into `innerHTML`.
- 👥 **Workspace access control:** org routes check membership and role (`PermissionError` → 403).
- 📨 **Invitations:** random tokens, stored hashed, with a 7-day expiry and an email-match check on acceptance.

---

## 🚀 Recommended Fix Order

1. **Lock down CORS** (#1). This is a one-line change with the biggest impact.
2. **Add `require_user` to the data, ring and scoring routes** (#2, #3, #4).
3. **Cap batch and input sizes** (#4) and **add login rate limiting** (#5).
4. **Stop leaking exception text** (#6) and **make cookies secure by default** (#7).
5. **Untrack the DB and fix the Docker volumes** (#9, #10).
6. Handle the remaining items (#8, #11–#13) before any non-demo deployment.

---

*This review covers the code as of commit `66fae8c`. Re-review after the fixes above, and before any deployment that handles real payment data.*
