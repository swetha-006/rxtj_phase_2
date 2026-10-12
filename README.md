<div align="center">

<h1>🛡️ RXT-J+ v4 "Glass Box"</h1>
<h3>Real-Time Digital Fraud Detection & Cost-Aware Decision Engine</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.14-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-v4%20(Calibrated)-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/PR--AUC-0.2072%20(2x%20Baseline)-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Inference-0.05ms-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Audit%20Gate-PASSED%20(6/6)-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-purple?style=flat-square"/>
</p>

<p>
  <strong><a href="#quickstart">Quickstart</a></strong> •
  <strong><a href="#core-differentiators">Differentiators</a></strong> •
  <strong><a href="#architecture">Architecture</a></strong> •
  <strong><a href="#api-reference">API Endpoints</a></strong> •
  <strong><a href="#audit-gate">Audit Gate</a></strong> •
  <strong><a href="#docker-deployment">Docker</a></strong>
</p>

</div>

---

## 0. The Pitch

> Most fraud detection systems are evaluated on AUC over balanced synthetic benchmarks. Banks do not pay in AUC — they pay in **fraud losses**, **analyst review hours**, and **churn from annoyed, falsely blocked customers**. RXT-J+ v4 optimizes those three costs directly.

When evaluated on the realistic, severely imbalanced production distribution (3.499% fraud rate), legacy models that claim 0.96 AUC on resampled SMOTE sets collapse, often flagging over 99% of legitimate transactions. 

**RXT-J+ v4** fixes the entire pipeline:
1. **Calibrated Gradient-Boosted Scorer:** Swaps opaque compression chains for a clean, leak-free LightGBM model calibrated with Isotonic Regression (**PR-AUC 0.2072**, double the baseline).
2. **Cost-Sensitive Decision Router:** Evaluates the expected rupee loss of **ALLOW**, **STEP_UP**, and **BLOCK** rather than enforcing an arbitrary static cutoff. A ₹200 coffee and an ₹80,000 wire transfer never receive the same threshold.
3. **Graph-Based Fraud Ring Detection:** Identifies syndicated fraud by maintaining a real-time bipartite entity graph (`card1 ↔ DeviceInfo ↔ addr1`), surfacing coordinated attacks that individual transaction scorers structurally cannot detect.
4. **Regulator-Ready Explainability:** Uses exact TreeSHAP contributions to trigger Adverse-Action reason codes (`R07`, `R12`, `R03`, `R21`) and delivers bounded counterfactual guidance (*"Approved if amount ≤ ₹X"*).
5. **Glass-Box Audit Harness:** Ships with an automated 6-point verification gate (`validate_model.py`) that fails CI builds on label leakage, future-to-past temporal bleed, class imbalance distortions, or degenerate features.

---

## 1. Core Differentiators

### 1. Cost-Sensitive Decision Router (§5.2)
A flat threshold creates massive false-positive friction on micro-transactions while failing to protect against catastrophic high-value breaches. For every transaction, RXT-J+ computes:

$$\text{Cost}(\text{ALLOW}) = p \cdot \text{amount}$$
$$\text{Cost}(\text{STEP\_UP}) = C_{\text{step}} + p \cdot (1 - \gamma) \cdot \text{amount}$$
$$\text{Cost}(\text{BLOCK}) = C_{\text{block}} \cdot (1 - p)$$

*Where $p$ = calibrated fraud probability, $\gamma$ = fraud defeat rate via step-up authentication (85%), $C_{\text{step}}$ = user friction cost (₹20), and $C_{\text{block}}$ = customer insult cost (₹400).*

The engine routes to the action with minimum expected cost:

| Amount | Fraud Prob ($p$) | ALLOW Cost | STEP_UP Cost | BLOCK Cost | Decision | Business Rationale |
|---|---|---|---|---|---|---|
| **₹200** | 0.30 | ₹60.00 | **₹29.00** | ₹280.00 | **STEP_UP** | Micro-transaction: challenge saves ₹31 vs allow |
| **₹200** | 0.70 | ₹140.00 | **₹41.00** | ₹120.00 | **STEP_UP** | High risk, but amount is too small to insult customer |
| **₹80,000** | 0.30 | ₹24,000.00 | ₹3,620.00 | **₹280.00** | **BLOCK** | Moderate probability on large transfer demands freeze |
| **₹80,000** | 0.08 | ₹6,400.00 | ₹980.00 | **₹368.00** | **BLOCK** | Even at 8% risk, potential loss justifies immediate block |

### 2. Fraud Ring Network Detection (§5.5)
Coordinated syndicates execute small, unremarkable transactions across multiple cards using shared device fingerprints and billing addresses.
- **Bipartite Graph Architecture:** Accounts link through shared `DeviceInfo` and `addr1` nodes.
- **Component Clustering:** Real-time connected component analysis extracts cluster size, new account ratio, and burst velocity.
- **Null Safety:** Missing or sentinel values (`None`, `"nan"`, `"MISSING"`) are strictly isolated to prevent false mega-clusters.

### 3. Glass-Box Explainability (§5.4)
- **Layer 1 — TreeSHAP:** Instant mathematical decomposition of the raw booster margin into positive and negative feature contributions.
- **Layer 2 — Adverse Action Reason Codes:**
  - `R07 AMOUNT_ANOMALY` — Transaction deviates by multiple standard deviations from account history.
  - `R12 DEVICE_UNRECOGNIZED` — Unrecognized device fingerprint never before associated with the account.
  - `R03 VELOCITY_SPIKE` — Surge in transaction frequency within a 1h/24h rolling window.
  - `R21 RING_ASSOCIATION` — Entity linked to a multi-account network cluster.
- **Layer 3 — Bounded Counterfactual:** Binary search identifies the maximum permissible transaction amount that would allow approval holding anomalous context fixed.

### 4. Alert-Budget Capacity Optimizer (§5.3)
Operational reality dictates that fraud teams have a finite number of analysts who can review alerts each day. The built-in optimizer allows operators to select their capacity budget and inspect empirical trade-offs:
- **Top 0.1% alerts:** 81.8% Precision | 2.3% Recall
- **Top 0.5% alerts:** 52.5% Precision | 7.5% Recall
- **Top 1.0% alerts:** 40.9% Precision | 11.7% Recall

---

## 2. Architecture

```
                          ┌──────────────────────────────────────────┐
    Transaction  ────────▶│  1. FEATURE ASSEMBLY                     │
    (JSON / Kafka)        │     amount · frequency · location        │
                          │     device · timing · history            │
                          │     ProfileStore EMA lookup  (<1ms)      │
                          └────────────────┬─────────────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  2. RING GRAPH ENRICHMENT                │
                          │     card ↔ device ↔ address components   │
                          │     NetworkX bipartite clustering        │
                          └────────────────┬─────────────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  3. GRADIENT-BOOSTED SCORER              │
                          │     Calibrated LightGBM (14 features)    │
                          │     → Calibrated P(fraud) in <0.05ms     │
                          └────────────────┬─────────────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  4. COST-SENSITIVE ROUTER                │
                          │     argmin of expected rupee cost        │
                          │     ALLOW / STEP-UP / BLOCK              │
                          └────────────────┬─────────────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  5. EXPLANATION LAYER                    │
                          │     TreeSHAP contributions               │
                          │     Adverse Reason Codes (R07, R12, etc) │
                          │     Counterfactual ("Approved if ≤ ₹X")  │
                          └────────────────┬─────────────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  6. GLASS-BOX AUDIT GATE (CI / CD)       │
                          │     blocks deploy on leak / imbalance /  │
                          │     degenerate-feature / flag-rate fail  │
                          └──────────────────────────────────────────┘
```

---

## 3. Quickstart

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Linux, macOS, or Windows

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/swetha-006/rxtj_phase_2.git
cd rxtj_phase_2

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Glass-Box Audit Gate
python validate_model.py

# 5. Run the complete test suite
pytest tests/ -v

# 6. Start the API service
uvicorn app:app --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive UI Dashboard:** [http://localhost:8000/panel](http://localhost:8000/panel)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Specification:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 4. Docker Deployment

### Run via Docker Compose
```bash
docker-compose up --build -d
```

### Run via Single Docker Container
```bash
docker build -t rxtj-v4-fraud-engine .
docker run -d -p 8000:8000 --name rxtj-engine rxtj-v4-fraud-engine
```

Healthcheck:
```bash
curl http://localhost:8000/health
```

---

## 5. API Reference

All requests and responses use JSON and are strictly validated via Pydantic v2.

### `POST /score/form-full`
Primary scoring endpoint performing feature generation, LightGBM inference, cost routing, ring association, and reason code extraction.

**Request:**
```json
{
  "account_id": "ACC_42",
  "transaction_id": "TXN_9876",
  "amount": 80000.0,
  "hour": 3,
  "product_cd": "W",
  "device_info": "MacOS",
  "addr1": 299.0
}
```

**Response (Sample):**
```json
{
  "account_id": "ACC_42",
  "transaction_id": "TXN_9876",
  "recommended_action": "STEP_UP_AUTHENTICATION",
  "cost_routing": {
    "action": "STEP_UP",
    "recommended_action": "STEP_UP_AUTHENTICATION",
    "p": 0.0071,
    "amount": 80000.0,
    "cost_allow": 564.42,
    "cost_step_up": 104.66,
    "cost_block": 397.18,
    "optimal_cost": 104.66,
    "rupee_savings": 459.75,
    "explanation": "Moderate risk on ₹80,000. Step-up challenge costs only ₹104.66, saving ₹459.75 vs unchecked approval."
  },
  "ring_association": {
    "ring_id": "RING_ACC_42",
    "ring_size": 2,
    "connected_accounts": ["ACC_42", "ACC_88"],
    "shared_devices": ["MacOS"],
    "is_ring_alert": false,
    "risk_level": "LOW"
  },
  "p1_risk_score": 0.0071,
  "latency_ms": 32.4
}
```

### `POST /cost/route`
Standalone cost calculator for any calibrated probability and amount.

**Request:**
```json
{
  "p": 0.30,
  "amount": 200.0
}
```

**Response:**
```json
{
  "action": "STEP_UP",
  "recommended_action": "STEP_UP_AUTHENTICATION",
  "p": 0.3,
  "amount": 200.0,
  "cost_allow": 60.0,
  "cost_step_up": 29.0,
  "cost_block": 280.0,
  "optimal_cost": 29.0
}
```

### `GET /ring/graph/{account_id}`
Returns ego-network nodes and links formatted for Canvas/D3 force-directed visualizers.

### `GET /ring/metrics/{account_id}`
Returns cluster degree, connected entity lists, and syndicate risk classification.

---

## 6. Glass-Box Audit Gate

Run `validate_model.py` to trigger the pre-deployment integrity suite:

```bash
python validate_model.py
```

```
============================================================================
  RXT-J+ GLASS-BOX AUDIT GATE REPORT
  Target: RXT-J+ v4 LightGBM (Calibrated)
  Status: [ PASS ]
============================================================================

1. Class-Balance Integrity: [ PASS ]
   Eval fraud rate (3.480%) matches production prevalence (3.499%) within 0.53% (<20% limit).

2. Degenerate Features:     [ PASS ]
   All 14 model features exhibit non-zero variance (std >= 0.05).

3. Label Leakage:           [ PASS ]
   No feature exceeds |corr| > 0.95 with target (max correlation: 0.0936).

4. Temporal Integrity:      [ PASS ]
   Strict chronological splitting verified: max(train) <= min(val) <= min(test).

5. Flag-Rate Sanity:        [ PASS ]
   Flag rate of 5.21% at operating threshold 0.15 is well within <20.0% operational limit.

6. Causality Audit:         [ PASS ]
   Verified derived features use strictly prior rows (no lookahead bias).

============================================================================
  OVERALL AUDIT GATE RESULT: ALL CHECKS PASSED (EXIT CODE 0)
============================================================================
```

---

## 7. Project Structure

```
rxtj_phase_2/
├── .github/workflows/
│   └── ci.yml               # Automated CI test & audit gate
├── data/
│   ├── tx_snapshot.parquet  # Causal transaction dataset (590k rows)
│   └── profile_stats.json   # Real-world class distribution stats
├── models/
│   └── lgbm.pkl             # Calibrated LightGBM model artifact
├── tests/
│   ├── test_api_integration.py # 12 endpoint tests & categorical edge cases
│   ├── test_cost_router.py     # Expected rupee cost verification
│   └── test_ring_detector.py   # NetworkX syndicate cluster tests
├── app.py                   # FastAPI application & ML serving core
├── cost_router.py           # Cost-Sensitive Decision Router
├── ring_detector.py         # NetworkX Bipartite Ring Graph Detector
├── panel.html               # Glass-Box interactive dashboard
├── validate_model.py        # Automated 6-point audit harness
├── requirements.txt         # Production dependencies
├── Dockerfile               # Multi-stage production container
└── docker-compose.yml       # Container orchestration
```

---

## 8. License & Acknowledgments

This project is open-source under the **MIT License**.
Based on research on the IEEE-CIS Fraud Detection dataset and developed for real-time payment fraud prevention.
