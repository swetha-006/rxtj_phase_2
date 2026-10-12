<div align="center">

<!-- ══════════════════════════════════════════════
     GRAPHICS NOTE FOR MAINTAINERS
     ══════════════════════════════════════════════
     Replace the placeholder blocks below with real assets:

     1. LOGO        → Save your project logo as docs/assets/logo.png (500×120px, transparent bg)
     2. DEMO GIF    → Screen-record the live dashboard hitting /account/compromise-score
                      and convert to docs/assets/demo.gif (max 5 MB)
     3. ARCH DIAGRAM→ Export the architecture SVG from the HTML build guide as
                      docs/assets/architecture.png (1200×700px)
     4. BADGES      → All badge URLs below auto-generate from shields.io —
                      swap {username}/{repo} with your actual GitHub path
     5. ROC CHART   → Copy results/fusion_roc.png → docs/assets/roc.png after training

     Recommended banner tool:  https://www.canva.com  (free, 1280×320px)
     Badge generator:          https://shields.io
     GIF recorder (Windows):   ShareX  https://getsharex.com
     ══════════════════════════════════════════════ -->

<!-- PROJECT LOGO — replace src with docs/assets/logo.png -->
<img src="docs/assets/logo.png" alt="RXT-J+ Logo" width="480"/>

<h1>RXT-J+ · Real-Time Transaction Risk Scoring</h1>
<h3>AI-powered payment fraud detection and account compromise scoring engine</h3>

<!-- BADGES — swap {username}/{repo} -->
<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/AUC-96.24%25-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/MCC-0.8426-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/Latency-0.019ms-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
</p>

<!-- DEMO GIF — replace src with docs/assets/demo.gif -->
<img src="docs/assets/demo.gif" alt="Live scoring demo" width="700"/>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Data Models](#data-models)
- [API Reference](#api-reference)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Training the Models](#training-the-models)
- [Testing](#testing)
- [Deployment](#deployment)
- [Usage Examples](#usage-examples)
- [User Flows](#user-flows)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Authors & Credits](#authors--credits)
- [License](#license)

---

## Overview

**RXT-J+** is a production-grade, real-time ML risk scoring engine for payment fraud detection and account compromise identification. It processes individual transactions in **0.019 ms** at over **51,000 transactions per second** and simultaneously evaluates whether the account behind a transaction has been taken over.

The system is built on the IEEE-CIS Fraud Detection dataset (590,540 transactions, 433 features) and surpasses the baseline RXT-J model (Almazroi & Ayub, IEEE Access 2023) by replacing static PCA with non-linear EARN+ feature extraction, augmenting the GRU with a self-attention mechanism, and extending beyond per-transaction fraud detection to full account-level behavioural scoring.

> **Base paper:** Almazroi & Ayub, *IEEE Access* 2023 · DOI: [10.1109/ACCESS.2023.3339226](https://doi.org/10.1109/ACCESS.2023.3339226)

---

## Key Features

| Feature | Detail |
|---|---|
| **Transaction fraud scoring** | ResNeXt + Self-Attention GRU + Isolation Forest ensemble, Jaya-optimised thresholds |
| **Account compromise detection** | Per-account behavioural profiling, 8 contextual deviation features, FusionNet scoring |
| **EARN+ feature engineering** | Autoencoder (latent 64) + ResNet (features 64) → concat 128 → Nystroem → IncrementalPCA |
| **Real-time API** | FastAPI, 12 endpoints, sub-millisecond latency, batch scoring |
| **Explainability** | Learnable attention weights per contextual feature, `/account/explain` endpoint |
| **Behavioural profiling** | SQLite WAL profile store, exponential moving average updates, 13,553 account profiles |
| **Streaming-ready** | Kafka producer stub on every scored transaction; consumer for async profile updates |
| **Multi-objective optimisation** | Jaya algorithm minimises `2×FPR + FNR` for fiscally-aware threshold selection |

---

## Tech Stack

### ML / Data Science
| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.2.2 | ResNeXt, GRU, Autoencoder, ResNet, FusionNet |
| scikit-learn | 1.4.2 | Nystroem, IncrementalPCA, IsolationForest, metrics |
| NumPy | 1.26.4 | Array operations throughout |
| pandas | 2.x | Data loading, profiling, feature engineering |
| pyarrow | 16.x | Parquet snapshot storage |
| matplotlib | 3.7+ | ROC curves, attention weight charts |
| joblib | 1.3.2 | Model serialisation |

### Backend / API
| Component | Version | Purpose |
|---|---|---|
| FastAPI | 0.110.0 | REST API framework |
| Uvicorn | 0.29.0 | ASGI server |
| Pydantic | 2.6.4 | Request/response validation |
| SQLite (WAL) | built-in | Behavioural profile store |
| kafka-python | 2.0.2 | Transaction event streaming |

### Infrastructure
| Tool | Purpose |
|---|---|
| Anaconda / conda | Environment management |
| Python 3.11 | Runtime (3.12+ lacks pre-built wheels for ML stack) |
| Apache Kafka | Transaction stream ingestion (optional for local dev) |

---

## Architecture

<!-- ARCHITECTURE DIAGRAM — replace src with docs/assets/architecture.png -->
<div align="center">
<img src="docs/assets/architecture.png" alt="System Architecture" width="800"/>
</div>

```
Incoming Transaction (224 raw features)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  EARN+ Feature Extraction                                │
│  Imputer(224) → Scaler(224)                             │
│  Autoencoder encoder → 64-dim latent                    │
│  ResNet extractor   → 64-dim features                   │
│  concat(128) → Nystroem(300) → IncrementalPCA(50)       │
└─────────────────────────┬───────────────────────────────┘
                          │ 50-dim feature vector
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Transaction Fraud Classifier                            │
│  ResNeXt (cardinality-4 paths) → 64-dim                 │
│  Self-Attention GRU (2-layer, seq_len=8) → context 64   │
│  Isolation Forest Model → anomaly score                 │
│  Jaya ensemble: W_MODEL×prob + W_IFM×ifm_score         │
│  Output: transaction_risk_score [0,1]                   │
└─────────────┬───────────────────────────────────────────┘
              │
   ┌──────────┴───────────┐
   │  Per-account profile  │
   │  (SQLite, EMA α=0.1) │
   └──────────┬───────────┘
              │  8 contextual deviation features
              ▼
┌─────────────────────────────────────────────────────────┐
│  FusionNet v2  (8→64→32→16→1 + feature_attn[8])        │
│  Features: amount_z · merchant_novelty · geo_disp       │
│            hour_dev · device_novel · vel_ratio          │
│            behavioral_drift · p1_risk_score             │
│  Jaya-optimised threshold                               │
│  Output: compromise_probability + explainability        │
└─────────────────────────┬───────────────────────────────┘
                          │
              ┌───────────┼──────────────┐
              ▼           ▼              ▼
         API response  Audit log    Kafka topic
         (JSON)        (SQLite)  scored_transactions
```

---

## Directory Structure

```
rxtj_plus/
│
├── app.py                          ← FastAPI application (all 12 endpoints)
├── preprocessing.py                ← Shared imputer + scaler transform functions
├── profile_store.py                ← SQLite profile CRUD (ProfileStore class)
├── profile_consumer.py             ← Kafka consumer for async profile updates
├── requirements.txt
├── .env.example                    ← Environment variable template
│
├── notebooks/                      ← Training pipeline (run in order)
│   ├── 01_preprocessing.ipynb      ← Mean imputation + StandardScaler
│   ├── 02_balancing.ipynb          ← SMOTE-Tomek class balancing
│   ├── 03_earn_features.ipynb      ← EARN+ (Autoencoder + ResNet + Nystroem + PCA)
│   ├── 04_attention_rxtj.ipynb     ← ResNeXt + SelfAttentionGRU training
│   ├── 05_jaya_optimize.ipynb      ← Multi-objective Jaya threshold optimisation
│   ├── 06_behavioral_profiling.py  ← Per-account behavioral profile builder
│   ├── 07_contextual_features.py   ← 8 contextual deviation feature computation
│   └── 08_fusion_model_training.py ← FusionNet v2 training + Jaya threshold
│
├── models/                         ← All trained model artifacts
│   ├── imputer.pkl                 ← SimpleImputer (224 features)
│   ├── scaler.pkl                  ← StandardScaler (224 features)
│   ├── nystroem.pkl                ← Nystroem approximation (128→300)
│   ├── incremental_pca.pkl         ← IncrementalPCA (300→50)
│   ├── autoencoder.pt              ← EARN+ autoencoder (224→64 latent)
│   ├── resnet_extractor.pt         ← EARN+ ResNet (224→64 features)
│   ├── attention_rxtj.pt           ← ResNeXt + AttentionGRU model
│   ├── isolation_forest.pkl        ← IFM anomaly scorer
│   ├── jaya_optimal_weights.pkl    ← Jaya ensemble weights
│   ├── fusion_net.pt               ← FusionNet v2 weights
│   └── fusion_scaler.pkl           ← StandardScaler for 8 fusion features
│
├── data/
│   ├── IEEE CIS/
│   │   ├── train_transaction.csv   ← 590,540 rows · 433 features
│   │   └── train_identity.csv      ← 144,233 rows · device/email features
│   ├── demo_samples.json           ← 8 hand-crafted demo transactions
│   ├── behavioral_profiles.db      ← SQLite: 13,553 account profiles
│   ├── tx_snapshot.parquet         ← Per-row historical snapshot (590,540 rows)
│   ├── contextual_features.npy     ← Fusion training features (590,540 × 8)
│   ├── fusion_labels.npy           ← Training labels
│   ├── fusion_feature_names.json   ← Ordered feature name list
│   └── drift_norm_params.json      ← Autoencoder normalisation params
│
├── results/
│   ├── deployment_config.json      ← Jaya weights + transaction model metrics
│   ├── fusion_config.json          ← FusionNet thresholds + metrics + attn weights
│   ├── fusion_roc.png              ← ROC curve + attention weight chart
│   └── fusion_training_curves.png  ← Training loss + AUC curves
│
├── docs/
│   └── assets/
│       ├── logo.png                ← Project logo (replace placeholder)
│       ├── demo.gif                ← Live demo GIF (replace placeholder)
│       ├── architecture.png        ← Architecture diagram
│       └── roc.png                 ← ROC chart (copy from results/)
│
└── tests/
    ├── test_preprocessing.py
    ├── test_profile_store.py
    ├── test_api_endpoints.py
    └── test_fusion_model.py
```

---

## Data Models

### Transaction Score Request

```python
class ScoreRequest(BaseModel):
    features: List[float]           # 224-dim raw feature vector
    threshold: Optional[float]      # override decision threshold (default 0.50)
```

### Compromise Score Request

```python
class CompromiseRequest(BaseModel):
    account_id:     str             # card1 value as string
    transaction_id: str             # unique transaction identifier
    features:       List[float]     # 224-dim raw feature vector
    amount:         Optional[float] # transaction amount
    hour:           Optional[int]   # hour of day (0–23)
    product_cd:     Optional[str]   # merchant category (W/H/C/S/R)
    device_info:    Optional[str]   # device fingerprint string
    addr1:          Optional[str]   # billing address region code
    timestamp:      Optional[float] # unix timestamp
```

### Compromise Score Response

```json
{
  "account_id":             "12345",
  "transaction_id":         "TXN_XYZ",
  "compromise_probability": 0.847,
  "decision":               "HIGH",
  "p1_risk_score":          0.612,
  "behavioral_drift_score": 0.731,
  "contextual_features": {
    "amount_z_score":    3.21,
    "merchant_novelty":  0.94,
    "geo_displacement":  0.88,
    "hour_deviation":    0.72,
    "device_novelty":    1.0,
    "velocity_ratio":    6.4,
    "behavioral_drift":  0.731,
    "p1_risk_score":     0.612
  },
  "explainability": {
    "device_novelty":    0.28,
    "merchant_novelty":  0.22,
    "geo_displacement":  0.19,
    "behavioral_drift":  0.16
  },
  "top_trigger_feature":  "device_novelty",
  "recommended_action":   "FREEZE_AND_NOTIFY",
  "latency_ms":           2.41,
  "model_version":        "rxtj_plus_v2"
}
```

### Account Behavioral Profile

```json
{
  "account_id":         "12345",
  "txn_count":          47,
  "amt_mean":           62.40,
  "amt_std":            18.70,
  "amt_mean_7d":        58.20,
  "hour_hist":          [0.01, 0.00, ..., 0.12, 0.15, ...],
  "merchant_counts":    {"W": 32, "H": 8, "C": 7},
  "known_device_count": 2,
  "geo_cluster":        "315",
  "velocity_1h":        0,
  "velocity_24h":       1,
  "profile_age_hours":  3.2
}
```

---

## API Reference

### Transaction Scoring (7 endpoints)

| Method | Path | Description |
|---|---|---|
| `POST` | `/score` | Score a 224-dim raw feature vector |
| `POST` | `/score/direct` | Score with key-value feature pairs |
| `POST` | `/score/form` | Score from a partial form input (pads missing with NaN) |
| `POST` | `/score/batch` | Score up to 1,000 transactions in one request |
| `GET` | `/history` | Last 50 scored transactions (all accounts) |
| `GET` | `/health` | Server health check and model load status |
| `GET` | `/model/info` | Loaded model metadata, weights, thresholds |

### Account Compromise Scoring (5 endpoints)

| Method | Path | Description |
|---|---|---|
| `POST` | `/account/compromise-score` | Full behavioural + contextual fusion scoring |
| `GET` | `/account/profile/{account_id}` | Retrieve stored behavioural profile |
| `GET` | `/account/history/{account_id}` | Last N scored events for an account |
| `GET` | `/account/alerts` | Accounts above compromise threshold in last 24 h |
| `POST` | `/account/explain` | Natural language explanation for a stored score |

> Full interactive docs available at `http://localhost:8000/docs` when the server is running.

---

## Setup & Installation

### Prerequisites

- Anaconda or Miniconda — [download](https://docs.conda.io/en/latest/miniconda.html)
- Git
- 16 GB RAM recommended for training (8 GB minimum for inference only)
- IEEE-CIS Fraud Detection dataset — [download from Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

> **Python version:** Must be **3.11**. Python 3.12+ has no pre-built wheels for PyTorch 2.2.2, scikit-learn 1.4.2, or numpy 1.26.4 on Windows.

### Step 1 — Clone the repository

```bash
git clone https://github.com/{username}/rxtj-plus.git
cd rxtj-plus
```

### Step 2 — Create the conda environment

```bash
conda create -n rxtj_env python=3.11 -y
conda activate rxtj_env
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```text
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.6.4
numpy==1.26.4
torch==2.2.2
scikit-learn==1.4.2
joblib==1.3.2
kafka-python==2.0.2
pandas>=2.0.0
pyarrow>=14.0.0
matplotlib>=3.7.0
```

### Step 4 — Place the dataset

```
data/
└── IEEE CIS/
    ├── train_transaction.csv   ← 590,540 rows
    └── train_identity.csv      ← 144,233 rows
```

### Step 5 — Verify installation

```bash
python -c "import torch, sklearn, fastapi, pandas; print('All dependencies OK')"
```

---

## Configuration

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

**`.env.example`**

```dotenv
# ── Server ──────────────────────────────────────────────
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true                    # set false in production

# ── Model paths ─────────────────────────────────────────
MODEL_DIR=models
DATA_DIR=data
RESULTS_DIR=results

# ── Decision thresholds (loaded from fusion_config.json) ─
# These are auto-loaded at startup — only override for testing
FUSION_HIGH_THRESHOLD=0.66
FUSION_ELEVATED_THRESHOLD=0.41

# ── Kafka (optional — set KAFKA_ENABLED=false for local dev) ─
KAFKA_ENABLED=false
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=scored_transactions

# ── Profile store ────────────────────────────────────────
PROFILE_DB_PATH=data/behavioral_profiles.db
PROFILE_EMA_ALPHA=0.1              # slow adaptation to prevent ATO retraining

# ── Logging ─────────────────────────────────────────────
LOG_LEVEL=INFO
```

---

## Running Locally

### Inference only (pre-trained models exist)

```bash
conda activate rxtj_env
cd rxtj-plus
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Expected startup output:

```
Loading RXT-J+ models...
  imputer / scaler loaded (224 features)
  EARN+ autoencoder loaded (input=224 → latent=64)
  EARN+ resnet extractor loaded (input=224 → features=64)
  attention_rxtj.pt loaded (IPCA_DIM=50)
  IFM loaded  |  W_MODEL=0.5343  W_IFM=0.4657  threshold=0.50
  FusionNet v2 loaded (8→64→32→16→1)
  [Phase2] ProfileStore ready. Accounts: 13,553
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### With Kafka (optional)

```bash
# Terminal 1 — Kafka (requires Docker)
docker-compose up kafka

# Terminal 2 — API server
python -m uvicorn app:app --reload --port 8000

# Terminal 3 — Profile update consumer
python profile_consumer.py
```

---

## Training the Models

Run notebooks **in strict order**. Each notebook saves artifacts consumed by the next.

| # | Notebook | Command | Runtime | Saves |
|---|---|---|---|---|
| 01 | Preprocessing | `jupyter nbconvert --to notebook --execute notebooks/01_preprocessing.ipynb` | ~2 min | `imputer.pkl` · `scaler.pkl` |
| 02 | Balancing | same pattern | ~5 min | balanced numpy arrays |
| 03 | EARN+ features | same pattern | ~15 min | `autoencoder.pt` · `resnet_extractor.pt` · `nystroem.pkl` · `incremental_pca.pkl` |
| 04 | Attention RXT-J | same pattern | ~30 min | `attention_rxtj.pt` · `isolation_forest.pkl` |
| 05 | Jaya optimisation | same pattern | ~10 min | `jaya_optimal_weights.pkl` · `deployment_config.json` |
| 06 | Behavioral profiling | `python notebooks/06_behavioral_profiling.py` | ~2.5 h (first run) | `behavioral_profiles.db` · `tx_snapshot.parquet` |
| 07 | Contextual features | `python notebooks/07_contextual_features.py` | ~15 min | `contextual_features.npy` · `drift_norm_params.json` |
| 08 | FusionNet training | `python notebooks/08_fusion_model_training.py` | ~40 min | `fusion_net.pt` · `fusion_scaler.pkl` · `fusion_config.json` |

> **Smart resume:** Notebooks 06 and 08 auto-detect existing artifacts and skip completed stages. Notebook 06 skips the 2.5-hour profile loop if `behavioral_profiles.db` already contains ≥90% of accounts.

---

## Testing

### Run all tests

```bash
conda activate rxtj_env
pytest tests/ -v --tb=short
```

### Run specific test modules

```bash
# Preprocessing pipeline
pytest tests/test_preprocessing.py -v

# Profile store CRUD operations
pytest tests/test_profile_store.py -v

# All API endpoints
pytest tests/test_api_endpoints.py -v

# FusionNet model loading and inference
pytest tests/test_fusion_model.py -v
```

### Coverage report

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Key tests included

```python
# tests/test_api_endpoints.py (excerpt)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_score_returns_risk():
    payload = {"features": [0.0] * 224}
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0

def test_compromise_score_all_fields():
    payload = {
        "account_id": "test_acct",
        "transaction_id": "TXN_001",
        "features": [0.0] * 224,
        "amount": 500.0, "hour": 3
    }
    response = client.post("/account/compromise-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert all(k in data for k in [
        "compromise_probability", "decision",
        "contextual_features", "explainability",
        "recommended_action", "latency_ms"
    ])
    assert len(data["contextual_features"]) == 8

def test_profile_created_after_scoring():
    client.post("/account/compromise-score", json={...})
    response = client.get("/account/profile/test_acct")
    assert response.status_code == 200
    assert response.json()["txn_count"] >= 1
```

---

## Deployment

### Production with Uvicorn + Gunicorn

```bash
pip install gunicorn

# Multi-worker production server (4 workers)
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 30 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
docker build -t rxtj-plus:latest .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  rxtj-plus:latest
```

### Docker Compose (with Kafka)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    env_file: .env
    depends_on: [kafka]

  profile-consumer:
    build: .
    command: python profile_consumer.py
    env_file: .env
    depends_on: [kafka, api]

  kafka:
    image: bitnami/kafka:3.6
    environment:
      KAFKA_CFG_PROCESS_ROLES: broker,controller
      KAFKA_CFG_NODE_ID: 1
      KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
    ports: ["9092:9092"]
```

```bash
docker-compose up -d
```

### Environment Variables for Production

```dotenv
APP_RELOAD=false
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
LOG_LEVEL=WARNING
```

### CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Test and Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -t rxtj-plus:${{ github.sha }} .
          docker push your-registry/rxtj-plus:${{ github.sha }}
```

---

## Usage Examples

### Score a single transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [86400, 500.0, 0, 12345, 111, 150, 0, 220, 1, 330, 87, 0, 2, 60, 1, 0, 0, 1, 0, 0.5, 0, 1, 1.2, 0, 0]}'
```

### Check compromise score for an account

```bash
curl -X POST http://localhost:8000/account/compromise-score \
  -H "Content-Type: application/json" \
  -d '{
    "account_id":     "12345",
    "transaction_id": "TXN_20241201_001",
    "features":       [86400, 499.99, 0, 12345, 111, 150, 0, 220, 1, 330],
    "amount":         499.99,
    "hour":           3,
    "product_cd":     "W",
    "device_info":    "iPhone 14 Pro",
    "addr1":          "330"
  }'
```

### Get an account's behavioral profile

```bash
curl http://localhost:8000/account/profile/12345
```

### Get all high-risk accounts in the last 24 hours

```bash
curl "http://localhost:8000/account/alerts?threshold=0.70&limit=20"
```

### Get a human-readable explanation for a scored transaction

```bash
curl -X POST http://localhost:8000/account/explain \
  -H "Content-Type: application/json" \
  -d '{"account_id": "12345", "transaction_id": "TXN_20241201_001"}'
```

### Batch scoring (up to 1,000 transactions)

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"features": [...]}, {"features": [...]}]}'
```

### Python client example

```python
import requests

BASE = "http://localhost:8000"

# Transaction fraud score
resp = requests.post(f"{BASE}/score", json={"features": [0.0] * 224})
print(f"Risk score: {resp.json()['risk_score']:.4f}")

# Account compromise score
resp = requests.post(f"{BASE}/account/compromise-score", json={
    "account_id": "12345",
    "transaction_id": "TXN_001",
    "features": [0.0] * 224,
    "amount": 150.0,
    "hour": 14,
    "product_cd": "W",
})
data = resp.json()
print(f"Compromise: {data['compromise_probability']:.4f} → {data['decision']}")
print(f"Top trigger: {data['top_trigger_feature']}")
print(f"Action: {data['recommended_action']}")
```

---

## User Flows

### Flow 1 — Payment gateway integration

```
Customer initiates payment
    → Gateway extracts 224 transaction features
    → POST /account/compromise-score  (or /score for transaction-only)
    → decision = "LOW"  → Approve, update profile silently
    → decision = "ELEVATED"  → Trigger SMS OTP before approving
    → decision = "HIGH"  → Block, freeze account, notify fraud team
```

### Flow 2 — Fraud analyst review

```
Alert fires (decision = "HIGH")
    → Analyst opens dashboard
    → GET /account/alerts  → sees high-risk account list
    → GET /account/history/{account_id}  → sees timeline of scored events
    → POST /account/explain  → reads natural language explanation
    → Analyst decides: confirm fraud or clear account
```

### Flow 3 — Merchant risk monitoring

```
Merchant queries aggregate risk
    → GET /account/alerts?threshold=0.5  → accounts with elevated activity
    → Filter by merchant's card base
    → Flag accounts for proactive outreach before next transaction
```

---

## Known Limitations

**P1 feature sparsity at inference:** The `/account/compromise-score` endpoint constructs the raw feature vector from the partial metadata fields provided (`amount`, `hour`, `product_cd`, `addr1`). This fills only 4 of 224 imputer input features; the remaining 220 are imputed to training-set means. When all 433 original transaction fields are available, connecting the full feature vector directly produces significantly better P1 scores.

**Behavioral profile cold start:** Accounts with fewer than 5 historical transactions produce unreliable contextual deviation scores. All 8 contextual features are available from the first transaction, but accuracy improves as the profile accumulates history.

**Autoencoder reconstruction range:** The autoencoder was trained on SMOTE-Tomek balanced data. Its reconstruction error range on the raw imbalanced data is narrow (0.1995–0.2530), giving `behavioral_drift` limited spread. Retraining the autoencoder on raw data would widen this range.

**Single-node SQLite:** The profile store uses SQLite in WAL mode. This is suitable for a single-server deployment. For multi-node or high-throughput production, migrate `profile_store.py` to PostgreSQL or Redis.

**Kafka is non-blocking stub:** The Kafka producer in `/score` is wrapped in a `try/except` and silently ignores failures. In a production deployment, add dead-letter queue handling and producer error monitoring.

---

## Future Enhancements

- [ ] **Full feature inference** — accept complete 224-column feature vectors via API and re-score with real P1 spread
- [ ] **Fraud pattern visualisation** — merchant-level risk heatmaps, temporal fraud cluster charts, account compromise timeline dashboard
- [ ] **Real-time Kafka consumer** — fully wire `profile_consumer.py` into a production consumer group with offset management and DLQ
- [ ] **Multi-node profile store** — migrate from SQLite to Redis or PostgreSQL for horizontal scaling
- [ ] **Online model updating** — incremental model fine-tuning as new confirmed fraud labels arrive without full retraining
- [ ] **Graph-based account linking** — detect fraud rings by modelling shared device / email / address connections across accounts using GNN
- [ ] **REST streaming endpoint** — Server-Sent Events for real-time dashboard push without polling `/account/alerts`
- [ ] **Federated scoring** — privacy-preserving model updates across multiple bank data silos without raw data sharing

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'`

You are on Python 3.14 or outside the conda environment.

```bash
conda activate rxtj_env
python --version   # must show Python 3.11.x
pip install pandas pyarrow matplotlib
```

### `RuntimeError: Error(s) in loading state_dict — size mismatch`

The model class definition in `app.py` does not match the saved checkpoint architecture.

```bash
# Inspect saved keys to identify the mismatch
python -c "
import torch
state = torch.load('models/attention_rxtj.pt', map_location='cpu')
for k, v in state.items(): print(k, tuple(v.shape))
"
```

Compare the printed keys against the class definition in `app.py` and align attribute names.

### Server crashes on startup with `fusion_config.json not found`

Notebooks 06–08 have not been run yet. Run them in order before starting the server.

```bash
cd notebooks
python 06_behavioral_profiling.py
python 07_contextual_features.py
python 08_fusion_model_training.py
```

### `ValueError: X has 224 features, but Nystroem is expecting 128 features`

You are feeding raw 224-dim features directly to Nystroem. The correct pipeline is:

```
raw(224) → imputer → scaler → AE encoder(64) + ResNet extract(64) → concat(128) → nystroem
```

Check the `build_earn_features()` function in `app.py`.

### `kafka.errors.NoBrokersAvailable`

Kafka is not running or `KAFKA_BOOTSTRAP_SERVERS` is incorrect. For local development set `KAFKA_ENABLED=false` in `.env` — the scoring endpoints work without Kafka.

### `P1 scores: std=0.011, 100% predicted fraud`

Only 4 of 224 features are being filled in the raw feature matrix. Use `imputer.feature_names_in_` to load all 224 columns from the original transaction CSV. See `fix_p1_scores.py` in the project root.

### Port 8000 already in use

```bash
# Find and kill the process on port 8000
netstat -ano | findstr :8000       # Windows
lsof -i :8000                      # macOS / Linux

# Or run on a different port
python -m uvicorn app:app --port 8080
```

---

## Contributing

We welcome contributions. Please follow these steps:

### 1. Fork and branch

```bash
git fork https://github.com/{username}/rxtj-plus.git
git checkout -b feature/your-feature-name
```

### 2. Code standards

- Follow PEP 8. Use `black` for formatting: `black .`
- All new functions require a docstring with `Args` and `Returns`
- New model classes must have matching unit tests in `tests/`
- Do not commit model artifacts (`.pt`, `.pkl`) to git — add to `.gitignore`

### 3. Testing

```bash
pytest tests/ -v
```

All tests must pass before opening a pull request.

### 4. Pull request checklist

- [ ] Tests added for new functionality
- [ ] Docstrings added / updated
- [ ] `requirements.txt` updated if new dependencies added
- [ ] README updated if API or configuration changes

### Code of Conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be respectful, constructive, and inclusive. Harassment of any kind will not be tolerated.

---

## Authors & Credits

| Name | Role |
|---|---|
| Sm | ML engineering lead, architecture, API design |
| Deepak B | Feature engineering, model training, evaluation |
| Joselin Jennilia A | Data preprocessing, balancing pipeline, testing |

**Faculty guide:** Ms. Soundarya, Assistant Professor, AIML Department, Sri Krishna College of Technology (SKCT), Coimbatore.

**Base paper:** Almazroi, A. A. and Ayub, N. (2023). *RXT-J: ResNeXt and Gated Recurrent Unit Based Model for Credit Card Fraud Detection.* IEEE Access. DOI: [10.1109/ACCESS.2023.3339226](https://doi.org/10.1109/ACCESS.2023.3339226)

**Dataset:** IEEE-CIS Fraud Detection — Kaggle Competition (2019). [Link](https://www.kaggle.com/c/ieee-fraud-detection)

---

## License

```
MIT License

Copyright (c) 2024 SKCT Team 13

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

<!-- ROC CHART — replace src with docs/assets/roc.png -->
<img src="docs/assets/roc.png" alt="ROC Curve" width="600"/>

<sub>Transaction fraud ROC · AUC = 96.24% &nbsp;|&nbsp; Account compromise ROC · AUC = 75.67%</sub>

<br/><br/>

<sub>Built with PyTorch · FastAPI · scikit-learn · IEEE-CIS Dataset</sub>

<br/>

<img src="https://img.shields.io/badge/SKCT-Team%2013-0052CC?style=flat-square"/>
<img src="https://img.shields.io/badge/B.E.%20CSE-Cyber%20Security-7B2D8B?style=flat-square"/>
<img src="https://img.shields.io/badge/NEP-2020-FF6B35?style=flat-square"/>

</div>
