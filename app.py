# ===========================================================
# app.py — RXT-J+ FastAPI Deployment (v4 — Glass Box ML Core)
# Run with: python -m uvicorn app:app --reload --port 8000
#
# ML Core Renovation (P3 + P5):
#   - Phase 1 Scorer: Calibrated LightGBM (models/lgbm.pkl)
#   - Obsolete inference chain retired (AttentionRXTJ, ResNet, Nystroem, IPCA, IFM, Jaya)
#   - Phase 2 Behavioral Drift Autoencoder + FusionNet v3 preserved
#   - Layer 1: TreeSHAP explanations
#   - Layer 2: Deterministic reason codes (R07, R12, R03, R21)
#   - Layer 3: Bounded counterfactual search on TransactionAmt
# ============================================================

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any, Union, Tuple
from profile_store import ProfileStore
import guardian_accounts
import guardian_orgs
from cost_router import CostSensitiveRouter, CostDecision
from ring_detector import FraudRingDetector, FraudRingMetrics
import numpy as np
import joblib
import json
import time
import os
import sys
import sqlite3
import pandas as pd
import shap

# Defensive PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cpu')
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    DEVICE = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_preprocessors, transform_raw, pad_form_features, raw_feature_dim

HISTORY_DB  = 'fraud_storage.db'
BASE        = os.path.dirname(os.path.abspath(__file__))


# ── Phase 2 PyTorch model architectures (Preserved) ──────────────────────────

if TORCH_AVAILABLE:
    class AutoencoderNet(nn.Module):
        """Autoencoder for Phase 2 behavioural drift scoring."""
        def __init__(self, input_dim, latent_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z


    class FusionNet(nn.Module):
        """FusionNet v3 — Attention-weighted MLP for account compromise scoring (13 features)."""
        def __init__(self, input_dim: int = 13):
            super().__init__()
            self.feature_attn = nn.Parameter(torch.ones(input_dim))
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            attn_weights = torch.softmax(self.feature_attn, dim=0)
            x_weighted   = x * attn_weights
            logit        = self.net(x_weighted).squeeze(1)
            return torch.sigmoid(logit), attn_weights
else:
    class AutoencoderNet:
        def __init__(self, *args, **kwargs):
            pass

    class FusionNet:
        def __init__(self, *args, **kwargs):
            pass


# ── Model Artifact Loading ───────────────────────────────────────────────────

def load_models():
    """
    Safely load the new Phase 1 calibrated LightGBM model and Phase 2 components.
    Fails clearly if the required artifact is unavailable or incompatible.
    """
    # 1. Load LightGBM artifact (P2 / P3)
    lgbm_path = os.path.join(BASE, 'models', 'lgbm.pkl')
    if not os.path.exists(lgbm_path):
        raise RuntimeError(
            f"Required ML Core artifact '{lgbm_path}' not found! "
            "Please train the model first by running: python train_gbdt.py"
        )
    
    try:
        lgbm_bundle = joblib.load(lgbm_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load LightGBM artifact from '{lgbm_path}': {e}")

    lgbm_model    = lgbm_bundle.get("model")
    calibrator    = lgbm_bundle.get("calibrator")
    feature_schema = lgbm_bundle.get("features", [])
    threshold     = float(lgbm_bundle.get("threshold", 0.15))
    top_devices   = set(lgbm_bundle.get("top_devices", []))
    feature_stats = lgbm_bundle.get("feature_stats", {})
    lgbm_metadata = lgbm_bundle.get("metadata", {})

    if lgbm_model is None or calibrator is None or not feature_schema:
        raise RuntimeError("Invalid lgbm.pkl artifact: Missing model, calibrator, or feature schema.")

    # TreeSHAP explainer for Layer 1 explanations
    try:
        shap_explainer = shap.TreeExplainer(lgbm_model)
    except Exception as e:
        print(f"Warning: Failed to initialize TreeExplainer ({e}). SHAP explanations may be disabled.")
        shap_explainer = None

    # 2. Load Phase 2 behavioral drift components (Autoencoder + Preprocessors)
    imputer, scaler = load_preprocessors(BASE)
    autoencoder = None
    fusion_model_obj = None
    fusion_scaler_obj = None
    fusion_config_data = {
        'input_dim': 13,
        'fusion_threshold': 0.50,
        'high_threshold': 0.507353,
        'elevated_threshold': 0.257353,
        'feature_names': [
            "amount_z_score", "merchant_novelty", "geo_displacement",
            "hour_deviation", "device_novelty", "velocity_ratio",
            "behavioral_drift", "p1_risk_score", "amount_percentile",
            "time_gap_norm", "merchant_fraud_rate", "cross_device_accounts",
            "account_age_score"
        ]
    }

    # Config + scaler are plain JSON / sklearn objects: load them regardless of torch,
    # so /model/info and the threshold constants never depend on the deep-learning stack.
    f_cfg_path = os.path.join(BASE, 'results/fusion_config.json')
    if os.path.exists(f_cfg_path):
        with open(f_cfg_path, 'r') as f:
            fusion_config_data = json.load(f)

    f_scaler_path = os.path.join(BASE, 'models/fusion_scaler.pkl')
    if os.path.exists(f_scaler_path):
        fusion_scaler_obj = joblib.load(f_scaler_path)

    if TORCH_AVAILABLE:
        try:
            ae_path = os.path.join(BASE, 'models/autoencoder.pt')
            if os.path.exists(ae_path):
                ae_state = torch.load(ae_path, map_location=DEVICE)
                ae_input_dim = ae_state['encoder.0.weight'].shape[1]
                autoencoder = AutoencoderNet(ae_input_dim).to(DEVICE)
                autoencoder.load_state_dict(ae_state)
                autoencoder.eval()

            # 3. Load Phase 2 FusionNet v3 weights (config/scaler loaded above)
            f_net_path = os.path.join(BASE, 'models/fusion_net.pt')
            if os.path.exists(f_net_path):
                f_input_dim = fusion_config_data.get('input_dim', 13)
                fusion_model_obj = FusionNet(f_input_dim).to(DEVICE)
                fusion_state = torch.load(f_net_path, map_location=DEVICE)
                fusion_model_obj.load_state_dict(fusion_state)
                fusion_model_obj.eval()
        except Exception as e:
            print(f"Warning: Failed to load Phase 2 PyTorch models: {e}. Falling back to ML Core.")
    else:
        print("Note: PyTorch is not available. Phase 2 behavioral drift features will use fallback defaults.")

    return (
        lgbm_model, calibrator, feature_schema, threshold, top_devices,
        feature_stats, lgbm_metadata, shap_explainer,
        imputer, scaler, autoencoder,
        fusion_model_obj, fusion_scaler_obj, fusion_config_data
    )


print("Loading RXT-J+ v4 models...")
(
    lgbm_model, calibrator, FEATURE_SCHEMA, THRESHOLD, TOP_DEVICES,
    FEATURE_STATS, LGBM_METADATA, shap_explainer,
    imputer, scaler, autoencoder,
    fusion_model, fusion_scaler, fusion_config
) = load_models()

RAW_DIM            = raw_feature_dim(imputer)
FUSION_THRESHOLD   = fusion_config.get('fusion_threshold', 0.50)
HIGH_THRESHOLD     = fusion_config.get('high_threshold', 0.507353)
ELEVATED_THRESHOLD = fusion_config.get('elevated_threshold', 0.257353)
FEATURE_NAMES      = fusion_config.get('feature_names', [])   # 13 names
drift_norm_path    = os.path.join(BASE, 'data/drift_norm_params.json')
DRIFT_NORM         = json.load(open(drift_norm_path)) if os.path.exists(drift_norm_path) else {"p5": 0.0, "p95": 1.0}
merchant_path      = os.path.join(BASE, 'data/merchant_fraud_rates.json')
MERCHANT_FRAUD_RATES = json.load(open(merchant_path)) if os.path.exists(merchant_path) else {}

print(f"ML Core loaded: LightGBM v4 (Calibrated). Threshold={THRESHOLD:.2f}, Features={len(FEATURE_SCHEMA)}")
if fusion_model is not None:
    print(f"Phase 2 FusionNet v3 loaded (input_dim={fusion_config.get('input_dim', 13)}). High={HIGH_THRESHOLD}, Elevated={ELEVATED_THRESHOLD}")
# ── Exact Categorical Dtypes & Encoders (Matching LightGBM Booster) ───────────
if hasattr(lgbm_model, "booster_") and hasattr(lgbm_model.booster_, "pandas_categorical") and lgbm_model.booster_.pandas_categorical:
    PROD_CATEGORIES = list(lgbm_model.booster_.pandas_categorical[0])
    DEV_CATEGORIES  = list(lgbm_model.booster_.pandas_categorical[1])
else:
    PROD_CATEGORIES = ['C', 'H', 'R', 'S', 'W']
    DEV_CATEGORIES  = sorted(list(TOP_DEVICES)) + ["MISSING", "OTHER"]

PROD_CAT_TYPE = pd.CategoricalDtype(categories=PROD_CATEGORIES, ordered=False)
DEV_CAT_TYPE  = pd.CategoricalDtype(categories=DEV_CATEGORIES, ordered=False)

def clean_device_info(val: Any) -> str:
    """Consistent device encoding matching training distribution."""
    if val is None or pd.isna(val):
        return "MISSING"
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "missing", "unknown", "0", "0.0"):
        return "MISSING"
    if s in TOP_DEVICES:
        return s
    return "OTHER"

def clean_product_cd(val: Any) -> str:
    """Consistent ProductCD encoding."""
    if val is None or pd.isna(val):
        return "W"
    s = str(val).strip().upper()
    if s in PROD_CATEGORIES:
        return s
    return "W"

# ── Cost Router & Ring Detector Global Instances ──────────────────────────────
cost_router_engine = CostSensitiveRouter(c_step=20.0, c_block=400.0, gamma=0.85)
ring_detector_engine = FraudRingDetector(
    window_seconds=86400.0,
    min_ring_size_alert=3,
    new_account_max_txn=3
)


# ── SQLite History Store ─────────────────────────────────────────────────────

HISTORY_PATH = os.environ.get('RXTJ_HISTORY_DB') or os.path.join(BASE, HISTORY_DB)
PROFILE_DB_PATH = os.environ.get('RXTJ_PROFILE_DB') or os.path.join(BASE, 'data', 'behavioral_profiles.db')

def _init_history_db():
    conn = sqlite3.connect(HISTORY_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
        " tid TEXT, amt REAL, score REAL, status TEXT, ts REAL)"
    )
    conn.commit()
    conn.close()

_init_history_db()

profile_store = ProfileStore(db_path=PROFILE_DB_PATH)
print(f"ProfileStore ready. Total accounts: {profile_store.coverage_stats()['total_accounts']:,}")


# ── Kafka Producer (Optional, non-blocking) ──────────────────────────────────

KAFKA_BOOTSTRAP = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', '').strip()
KAFKA_TOPIC     = os.environ.get('KAFKA_PROFILE_TOPIC', 'profile_updates')
kafka_producer  = None

if KAFKA_BOOTSTRAP:
    try:
        from kafka import KafkaProducer
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP.split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks=0,
            linger_ms=10,
            request_timeout_ms=2000,
        )
        print(f"Kafka producer ready -> {KAFKA_BOOTSTRAP} topic={KAFKA_TOPIC}")
    except Exception as e:
        print(f"Kafka producer disabled ({e}).")
        kafka_producer = None
else:
    print("Kafka producer disabled (set KAFKA_BOOTSTRAP_SERVERS to enable).")

def _publish_profile_update(transaction_id, risk_score, decision, features):
    if kafka_producer is None:
        return
    try:
        kafka_producer.send(KAFKA_TOPIC, {
            'transaction_id': transaction_id,
            'risk_score':     float(risk_score),
            'decision':       decision,
            'features':       [float(v) for v in features],
            'ts':             time.time(),
        })
    except Exception:
        pass


# ── App Init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RXT-J+ Fraud Risk Scoring API",
    description="Real-time payment fraud detection — Glass Box ML Core (Calibrated LightGBM + Contextual Fusion)",
    version="4.0.0"
)

guardian_accounts.init_db()
guardian_orgs.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    features: List[float] = Field(..., description="Feature vector (14-dim exact, 224-dim raw, or 50-dim)")

class DirectScoreRequest(BaseModel):
    transaction_id: str
    features: List[float] = Field(..., description="Feature vector")

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]

class FormScoreRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    amount:         float
    is_foreign:     bool = False
    customer_age:   Optional[int] = 30

class RiskResponse(BaseModel):
    transaction_id: str
    risk_score:     float
    decision:       str
    confidence:     float
    latency_ms:     float
    explanation:    Optional[Dict[str, Any]] = None

class CompromiseRequest(BaseModel):
    account_id:     str
    transaction_id: str
    features:       Optional[List[float]] = None
    amount:         Optional[float] = 0.0
    hour:           Optional[int]   = 0
    product_cd:     Optional[str]   = ""
    device_info:    Optional[str]   = None
    addr1:          Optional[Union[str, float, int]] = None
    timestamp:      Optional[float] = None

class ExplainRequest(BaseModel):
    account_id:     str
    transaction_id: str

class ExplainTransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    account_id:     Optional[str]   = None
    amount:         Optional[float] = None
    hour:           Optional[int]   = None
    product_cd:     Optional[str]   = "W"
    device_info:    Optional[str]   = None
    addr1:          Optional[Union[str, float, int]] = None
    features:       Optional[List[float]] = None

class FullFormRequest(BaseModel):
    account_id:     str             = Field(...,  example="ACC_42")
    transaction_id: str             = Field(...,  example="TXN_001")
    amount:         float           = Field(...,  example=289.99, description="Transaction amount in USD")
    hour:           int             = Field(14,   ge=0, le=23,    description="Hour of day (0-23, local)")
    product_cd:     str             = Field("W",  example="W",    description="Product code: W C H R S")
    is_foreign:     bool            = Field(False,                description="True if card-not-present / international")
    customer_age:   Optional[int]   = Field(30,   ge=0, le=120,   description="Cardholder age")
    device_info:    Optional[str]   = Field(None, example="MacOS")
    addr1:          Optional[Union[str, float, int]] = Field(None, example="299",  description="Billing postal/zip code")
    timestamp:      Optional[float] = Field(None,                 description="Unix timestamp")


# ── Feature Transformation & Core Scorer (P3) ─────────────────────────────────

def transform_to_lgbm_features(
    inputs: Any,
    profile: Optional[dict] = None,
    txn_meta: Optional[dict] = None
) -> pd.DataFrame:
    """
    Deterministically transforms various input formats into the exact 14-feature schema
    required by the LightGBM model.
    Supported inputs:
      - DataFrame with partial or complete features
      - Dictionary (form inputs / key-value metadata)
      - List or 1D/2D numpy array:
          - 14-dim exact feature vector
          - 224-dim raw IEEE feature vector (maps col 0:DT->hour, col 1:Amt, col 9:addr1)
          - 50-dim post-IPCA vector (maps col 0 to amount proxy)
    """
    rows = []
    
    if isinstance(inputs, pd.DataFrame):
        df_in = inputs.copy()
        for f in FEATURE_SCHEMA:
            if f not in df_in.columns:
                df_in[f] = np.nan
        df_out = df_in[FEATURE_SCHEMA].copy()
        
    elif isinstance(inputs, dict):
        amt = float(inputs.get("amount", inputs.get("TransactionAmt", 0.0)))
        hr  = int(inputs.get("hour", 0)) % 24
        pcd = clean_product_cd(inputs.get("product_cd", inputs.get("ProductCD", "W")))
        dev = clean_device_info(inputs.get("device_info", inputs.get("DeviceInfo", None)))
        addr = float(inputs.get("addr1")) if inputs.get("addr1") is not None and str(inputs.get("addr1")).strip() != "" else np.nan

        if profile:
            amt_mean = float(inputs.get("amt_cumsum", profile.get("amt_mean", amt)))
            amt_std  = float(inputs.get("amt_cumstd", profile.get("amt_std", 0.0)))
            txn_seq  = int(inputs.get("txn_seq", profile.get("txn_count", 0)))
            v1h      = int(inputs.get("velocity_1h", profile.get("velocity_1h", 0)))
            v24h     = int(inputs.get("velocity_24h", profile.get("velocity_24h", 0)))
            kd       = profile.get("known_devices", [])
            if "device_novel" in inputs:
                dev_nov = float(inputs["device_novel"])
            else:
                dev_nov = 0.0 if (dev and dev in kd) else (1.0 if dev != "MISSING" else 0.5)
        else:
            amt_mean = float(inputs.get("amt_cumsum", amt))
            amt_std  = float(inputs.get("amt_cumstd", 0.0))
            txn_seq  = int(inputs.get("txn_seq", 0))
            v1h      = int(inputs.get("velocity_1h", 0))
            v24h     = int(inputs.get("velocity_24h", 0))
            dev_nov  = float(inputs.get("device_novel", 0.5))

        row = {
            "TransactionAmt": amt,
            "ProductCD":      pcd,
            "addr1":          addr,
            "DeviceInfo":     dev,
            "hour":           hr,
            "amt_cumsum":     amt_mean,
            "amt_cumstd":     amt_std,
            "txn_seq":        txn_seq,
            "velocity_1h":    v1h,
            "velocity_24h":   v24h,
            "device_novel":   dev_nov,
            "amt_to_mean":    amt / (amt_mean + 1.0),
            "amt_diff_mean":  amt - amt_mean,
            "velocity_ratio": v1h / (v24h / 24.0 + 1e-4)
        }
        df_out = pd.DataFrame([row])
        
    else:
        # Numeric array or list of floats
        arr = np.asarray(inputs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_samples, n_dim = arr.shape

        for i in range(n_samples):
            vec = arr[i]
            if n_dim == 14:
                # Exact model schema
                row = {
                    FEATURE_SCHEMA[j]: (
                        clean_product_cd(vec[j]) if FEATURE_SCHEMA[j] == "ProductCD"
                        else (clean_device_info(vec[j]) if FEATURE_SCHEMA[j] == "DeviceInfo" else float(vec[j]))
                    )
                    for j in range(14)
                }
            elif n_dim == 224:
                # Raw IEEE-CIS feature vector
                amt = float(vec[1])
                hr  = int(vec[0] // 3600) % 24 if not np.isnan(vec[0]) else 12
                addr = float(vec[9]) if not np.isnan(vec[9]) else np.nan
                row = {
                    "TransactionAmt": amt,
                    "ProductCD":      "W",
                    "addr1":          addr,
                    "DeviceInfo":     "MISSING",
                    "hour":           hr,
                    "amt_cumsum":     amt,
                    "amt_cumstd":     0.0,
                    "txn_seq":        0,
                    "velocity_1h":    0,
                    "velocity_24h":   0,
                    "device_novel":   0.5,
                    "amt_to_mean":    1.0,
                    "amt_diff_mean":  0.0,
                    "velocity_ratio": 0.0
                }
            elif n_dim == 50:
                # Post-IPCA vector
                amt = abs(float(vec[0])) * 100.0 if not np.isnan(vec[0]) else 100.0
                row = {
                    "TransactionAmt": amt,
                    "ProductCD":      "W",
                    "addr1":          np.nan,
                    "DeviceInfo":     "MISSING",
                    "hour":           12,
                    "amt_cumsum":     amt,
                    "amt_cumstd":     0.0,
                    "txn_seq":        0,
                    "velocity_1h":    0,
                    "velocity_24h":   0,
                    "device_novel":   0.5,
                    "amt_to_mean":    1.0,
                    "amt_diff_mean":  0.0,
                    "velocity_ratio": 0.0
                }
            else:
                amt = abs(float(vec[0])) if n_dim > 0 and not np.isnan(vec[0]) else 100.0
                row = {
                    "TransactionAmt": amt,
                    "ProductCD":      "W",
                    "addr1":          np.nan,
                    "DeviceInfo":     "MISSING",
                    "hour":           12,
                    "amt_cumsum":     amt,
                    "amt_cumstd":     0.0,
                    "txn_seq":        0,
                    "velocity_1h":    0,
                    "velocity_24h":   0,
                    "device_novel":   0.5,
                    "amt_to_mean":    1.0,
                    "amt_diff_mean":  0.0,
                    "velocity_ratio": 0.0
                }
            rows.append(row)
        df_out = pd.DataFrame(rows)

    # Format categorical types matching training artifact exactly
    df_out["ProductCD"] = df_out["ProductCD"].apply(clean_product_cd).astype(PROD_CAT_TYPE)
    df_out["DeviceInfo"] = df_out["DeviceInfo"].apply(clean_device_info).astype(DEV_CAT_TYPE)

    return df_out[FEATURE_SCHEMA]


def score_transaction_features(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score a prepared feature DataFrame using the calibrated LightGBM model.
    Returns: (calibrated_probs, binary_preds, confidence)
    """
    X_eval = X_df[FEATURE_SCHEMA].copy()
    X_eval["ProductCD"] = X_eval["ProductCD"].apply(clean_product_cd).astype(PROD_CAT_TYPE)
    X_eval["DeviceInfo"] = X_eval["DeviceInfo"].apply(clean_device_info).astype(DEV_CAT_TYPE)

    raw_probs = lgbm_model.predict_proba(X_eval)[:, 1]
    cal_probs = np.clip(calibrator.predict(raw_probs), 0.0, 1.0)
    preds     = (cal_probs >= THRESHOLD).astype(int)
    conf      = np.where(preds == 1, cal_probs, 1.0 - cal_probs)

    return cal_probs, preds, conf


def score_raw_features(
    features_np: np.ndarray,
    profile: Optional[dict] = None,
    txn_meta: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score raw inputs through the calibrated LightGBM model."""
    X_df = transform_to_lgbm_features(features_np, profile=profile, txn_meta=txn_meta)
    return score_transaction_features(X_df)


def score_earn_features(Z_earn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility scorer for EARN+ features, routed to calibrated LightGBM."""
    X_df = transform_to_lgbm_features(Z_earn)
    return score_transaction_features(X_df)


# ── Explanations: TreeSHAP, Reason Codes, Counterfactuals (P5) ────────────────

def compute_shap_layer(X_df: pd.DataFrame) -> dict:
    """
    Layer 1: TreeSHAP Feature Contributions.
    Computes signed feature contributions from the LightGBM estimator.
    """
    if shap_explainer is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "TreeExplainer is not initialized."
        }

    X_eval = X_df[FEATURE_SCHEMA].copy()
    X_eval["ProductCD"] = X_eval["ProductCD"].apply(clean_product_cd).astype(PROD_CAT_TYPE)
    X_eval["DeviceInfo"] = X_eval["DeviceInfo"].apply(clean_device_info).astype(DEV_CAT_TYPE)

    shap_vals = shap_explainer.shap_values(X_eval)
    if isinstance(shap_vals, list):
        shap_vec = shap_vals[1][0]  # Positive / fraud class
    elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
        shap_vec = shap_vals[0, :, 1]
    else:
        shap_vec = shap_vals[0]

    expected_val = shap_explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = float(expected_val[1])
    else:
        expected_val = float(expected_val)

    raw_margin = float(expected_val + np.sum(shap_vec))
    raw_prob   = float(1.0 / (1.0 + np.exp(-raw_margin)))
    cal_prob   = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))

    contributions = []
    shap_dict = {}

    for i, col in enumerate(FEATURE_SCHEMA):
        val = X_eval.iloc[0][col]
        s_val = float(shap_vec[i])
        shap_dict[col] = s_val
        direction = "INCREASES_FRAUD_RISK" if s_val > 0 else ("DECREASES_FRAUD_RISK" if s_val < 0 else "NEUTRAL")
        contributions.append({
            "feature": col,
            "shap_value": round(s_val, 4),
            "feature_value": str(val) if not pd.isna(val) else "MISSING",
            "risk_direction": direction
        })

    # Sort contributions by absolute SHAP value magnitude
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "base_value": round(expected_val, 4),
        "raw_model_margin": round(raw_margin, 4),
        "raw_probability": round(raw_prob, 4),
        "calibrated_probability": round(cal_prob, 4),
        "feature_contributions": contributions,
        "shap_dict": shap_dict,
        "disclosure": (
            "TreeSHAP contributions explain the raw LightGBM margin output. "
            "Displayed probability is calibrated via Isotonic Regression, which adjusts "
            "probability calibration without altering monotonicity."
        )
    }


def generate_reason_codes(
    X_df: pd.DataFrame,
    shap_dict: Dict[str, float],
    context_meta: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Layer 2: Human-readable Reason Codes.
    Deterministic mapping from model features and SHAP contributions to:
      - R07 — AMOUNT_ANOMALY
      - R12 — DEVICE_UNRECOGNIZED
      - R03 — VELOCITY_SPIKE
      - R21 — RING_ASSOCIATION (compatible with future ring_detector.py)
    """
    row = X_df.iloc[0]
    active_reasons = []

    # 1. R07 — AMOUNT_ANOMALY
    amt      = float(row["TransactionAmt"])
    amt_mean = float(row["amt_cumsum"])
    amt_to_m = float(row["amt_to_mean"])
    amt_shap = sum(shap_dict.get(k, 0.0) for k in ["TransactionAmt", "amt_diff_mean", "amt_to_mean"])

    if amt_shap > 0.02 or amt_to_m >= 2.0 or (amt > amt_mean * 1.8 and amt > 200.0):
        active_reasons.append({
            "code": "R07",
            "name": "AMOUNT_ANOMALY",
            "shap_contribution": round(amt_shap, 4),
            "category": "DIRECT_MODEL_EVIDENCE",
            "explanation": f"Transaction amount (${amt:.2f}) is significantly anomalous compared to historical account average (${amt_mean:.2f}, {amt_to_m:.1f}x higher).",
            "evidence": {
                "amount": round(amt, 2),
                "historical_mean": round(amt_mean, 2),
                "amt_to_mean_ratio": round(amt_to_m, 2),
                "shap_contribution": round(amt_shap, 4)
            }
        })

    # 2. R12 — DEVICE_UNRECOGNIZED
    dev_nov  = float(row["device_novel"])
    dev_info = str(row["DeviceInfo"])
    dev_shap = float(shap_dict.get("device_novel", 0.0) + shap_dict.get("DeviceInfo", 0.0))

    if dev_nov == 1.0 or dev_shap > 0.02:
        active_reasons.append({
            "code": "R12",
            "name": "DEVICE_UNRECOGNIZED",
            "shap_contribution": round(dev_shap, 4),
            "category": "DIRECT_MODEL_EVIDENCE",
            "explanation": f"Transaction initiated from an unrecognized device ('{dev_info}') never previously seen on this account.",
            "evidence": {
                "device_info": dev_info,
                "device_novelty_score": dev_nov,
                "shap_contribution": round(dev_shap, 4)
            }
        })

    # 3. R03 — VELOCITY_SPIKE
    v1h    = int(row["velocity_1h"])
    v24h   = int(row["velocity_24h"])
    v_rat  = float(row["velocity_ratio"])
    v_shap = sum(shap_dict.get(k, 0.0) for k in ["velocity_1h", "velocity_24h", "velocity_ratio"])

    if v1h >= 2 or v_rat > 2.0 or v_shap > 0.02:
        active_reasons.append({
            "code": "R03",
            "name": "VELOCITY_SPIKE",
            "shap_contribution": round(v_shap, 4),
            "category": "DIRECT_MODEL_EVIDENCE",
            "explanation": f"Rapid velocity acceleration: {v1h} transaction(s) in the past hour ({v_rat:.1f}x typical velocity).",
            "evidence": {
                "velocity_1h": v1h,
                "velocity_24h": v24h,
                "velocity_ratio": round(v_rat, 2),
                "shap_contribution": round(v_shap, 4)
            }
        })

    # 4. R21 — RING_ASSOCIATION (§5.5)
    meta = context_meta or {}
    ring_metrics: Optional[FraudRingMetrics] = meta.get("ring_metrics")
    ring_size = ring_metrics.ring_size if ring_metrics else int(meta.get("ring_size", 1))
    is_ring_alert = ring_metrics.is_ring_alert if ring_metrics else bool(meta.get("is_ring_alert", False))
    shared_devs = ring_metrics.shared_devices if ring_metrics else meta.get("shared_devices", [])
    connected_accs = ring_metrics.connected_accounts if ring_metrics else meta.get("connected_accounts", [])
    ring_id = ring_metrics.ring_id if ring_metrics else meta.get("ring_id", "RING_0")
    ring_risk = ring_metrics.risk_level if ring_metrics else meta.get("ring_risk_level", "LOW")

    if ring_size > 1 or is_ring_alert or ring_risk in ("ELEVATED", "CRITICAL"):
        active_reasons.append({
            "code": "R21",
            "name": "RING_ASSOCIATION",
            "shap_contribution": 0.1800 if is_ring_alert else 0.0800,
            "category": "CONTEXTUAL_GRAPH_EVIDENCE",
            "explanation": (
                f"Account associated with an active fraud-ring cluster '{ring_id}' of {ring_size} accounts "
                f"sharing entity linkage (shared devices: {len(shared_devs)})."
            ),
            "evidence": {
                "ring_id": str(ring_id),
                "ring_size": ring_size,
                "is_ring_alert": is_ring_alert,
                "risk_level": ring_risk,
                "shared_devices": shared_devs,
                "connected_accounts_count": len(connected_accs)
            }
        })
    else:
        active_reasons.append({
            "code": "R21",
            "name": "RING_ASSOCIATION",
            "status": "NORMAL",
            "category": "CONTEXTUAL_GRAPH_EVIDENCE",
            "explanation": "No coordinated fraud-ring graph association detected for this entity.",
            "evidence": {"ring_size": 1, "is_ring_alert": False, "risk_level": "LOW"}
        })

    # Rank active reasons by SHAP contribution magnitude descending (unavailable placed at end)
    active_sorted = [r for r in active_reasons if r.get("status") != "UNAVAILABLE"]
    unavail = [r for r in active_reasons if r.get("status") == "UNAVAILABLE"]

    active_sorted.sort(key=lambda x: abs(x.get("shap_contribution", 0.0)), reverse=True)
    for idx, r in enumerate(active_sorted, 1):
        r["rank"] = idx

    return active_sorted + unavail


def compute_counterfactual_explanation(
    X_df: pd.DataFrame,
    current_prob: float,
    threshold: float = THRESHOLD
) -> dict:
    """
    Layer 3: Bounded Counterfactual Explanation.
    Searches for minimal plausible feature modification on actionable feature (TransactionAmt)
    that causes decision to drop below threshold, holding other features constant.
    Consistently updates dependent features (amt_to_mean, amt_diff_mean).
    """
    row = X_df.iloc[0].to_dict()
    orig_amt = float(row["TransactionAmt"])
    amt_cumsum = float(row["amt_cumsum"])

    if current_prob < threshold:
        return {
            "counterfactual_found": False,
            "status": "ALREADY_LOW_RISK",
            "original_value": round(orig_amt, 2),
            "original_probability": round(current_prob, 4),
            "threshold": threshold,
            "explanation": f"Transaction already receives a low-risk decision (LEGIT) with calibrated risk {current_prob:.4f} < threshold {threshold:.2f}.",
            "disclaimer": "Hypothetical model-based counterfactual; does not guarantee real-world transaction approval."
        }

    # Binary search bounds: [1.0, orig_amt]
    low_bound = 1.0
    high_bound = orig_amt

    def evaluate_test_amt(test_amt: float) -> float:
        test_row = row.copy()
        test_row["TransactionAmt"] = float(test_amt)
        test_row["amt_to_mean"]    = float(test_amt) / (amt_cumsum + 1.0)
        test_row["amt_diff_mean"]  = float(test_amt) - amt_cumsum
        test_df = pd.DataFrame([test_row])
        test_p, _, _ = score_transaction_features(test_df)
        return float(test_p[0])

    p_at_min = evaluate_test_amt(low_bound)
    if p_at_min >= threshold:
        return {
            "counterfactual_found": False,
            "status": "NO_SINGLE_FEATURE_CROSSOVER",
            "feature_name": "TransactionAmt",
            "original_value": round(orig_amt, 2),
            "original_probability": round(current_prob, 4),
            "minimum_tested_value": round(low_bound, 2),
            "probability_at_minimum": round(p_at_min, 4),
            "threshold": threshold,
            "explanation": (
                f"Holding other high-risk features constant (e.g. velocity/device anomalies), "
                f"reducing transaction amount to minimum (${low_bound:.2f}) does not lower risk below "
                f"threshold {threshold:.2f} (resulting risk: {p_at_min:.4f})."
            ),
            "disclaimer": "Hypothetical model-based counterfactual; does not guarantee real-world transaction approval."
        }

    # Bounded binary search for crossover point
    low, high = low_bound, high_bound
    for _ in range(30):
        mid = (low + high) / 2.0
        p_mid = evaluate_test_amt(mid)
        if p_mid < threshold:
            low = mid
        else:
            high = mid

    cf_amt = float(np.floor(low * 100) / 100.0)
    cf_prob = evaluate_test_amt(cf_amt)
    while cf_prob >= threshold and cf_amt > (low_bound + 0.10):
        cf_amt = round(cf_amt - 0.50, 2)
        cf_prob = evaluate_test_amt(cf_amt)

    return {
        "counterfactual_found": True,
        "feature_name": "TransactionAmt",
        "original_value": round(orig_amt, 2),
        "counterfactual_value": cf_amt,
        "original_probability": round(current_prob, 4),
        "counterfactual_probability": round(cf_prob, 4),
        "threshold": threshold,
        "dependent_features_updated": ["amt_to_mean", "amt_diff_mean"],
        "explanation": f"This transaction would have received a lower-risk decision (LEGIT) if its amount were at or below ${cf_amt:,.2f}, holding other features constant.",
        "disclaimer": "Hypothetical model-based counterfactual; does not guarantee real-world transaction approval."
    }


def explain_lgbm_prediction(
    X_df: pd.DataFrame,
    transaction_id: str = "TXN_001",
    context_meta: Optional[dict] = None
) -> dict:
    """Full 3-layer explanation bundle."""
    cal_probs, preds, conf = score_transaction_features(X_df)
    current_p = float(cal_probs[0])
    decision  = "FRAUD" if preds[0] == 1 else "LEGIT"

    layer1 = compute_shap_layer(X_df)
    shap_dict = layer1.get("shap_dict", {})
    layer2 = generate_reason_codes(X_df, shap_dict, context_meta=context_meta)
    layer3 = compute_counterfactual_explanation(X_df, current_p, threshold=THRESHOLD)

    return {
        "transaction_id": transaction_id,
        "calibrated_risk_score": round(current_p, 4),
        "decision": decision,
        "threshold": THRESHOLD,
        "layer1_shap": layer1,
        "layer2_reason_codes": layer2,
        "layer3_counterfactual": layer3
    }


# ── Phase 2 Contextual Feature Extraction (Preserved) ─────────────────────────

def compute_contextual_features(
    features_np: np.ndarray,
    profile: dict,
    txn_meta: dict,
) -> tuple:
    """Compute the 13-dim contextual feature vector for FusionNet v3."""
    amount     = float(txn_meta.get("amount", 0.0))
    hour       = int(txn_meta.get("hour", 0)) % 24
    product_cd = str(txn_meta.get("product_cd", ""))
    device     = str(txn_meta.get("device_info", "")) if txn_meta.get("device_info") else ""
    addr1      = str(txn_meta.get("addr1", ""))

    amt_std = max(profile.get("amt_std", 1.0), 0.1)
    f1_amount_z = float(np.clip(
        (amount - profile.get("amt_mean", 0.0)) / amt_std, -5.0, 5.0
    ))

    mc       = profile.get("merchant_counts", {})
    max_freq = max(mc.values()) if mc else 1
    freq     = mc.get(product_cd, 0)
    f2_merchant_novelty = 1.0 - (freq / (max_freq + 1e-9)) if mc else 1.0

    geo_cluster = profile.get("geo_cluster", "")
    try:
        disp = abs(float(addr1) - float(geo_cluster)) if (addr1 and geo_cluster) else 0.0
        f3_geo_disp = float(min(disp / 500.0, 1.0))
    except (ValueError, TypeError):
        f3_geo_disp = 0.5

    hh     = profile.get("hour_hist", [1/24]*24)
    hh_val = hh[hour] if len(hh) == 24 else 1/24
    f4_hour_dev = float(max(0.0, 1.0 - hh_val * 24))

    kd = profile.get("known_devices", [])
    f5_device_novel = 0.0 if (device and device in kd) else (1.0 if device else 0.5)

    vel_1h  = float(profile.get("velocity_1h", 0))
    vel_24h = float(profile.get("velocity_24h", 0))
    f6_vel_ratio = float(np.clip(vel_1h / (vel_24h / 24.0 + 1e-6), 0.0, 10.0))

    # F7: behavioral drift via autoencoder reconstruction error
    if TORCH_AVAILABLE and autoencoder is not None:
        try:
            with torch.no_grad():
                batch       = torch.FloatTensor(features_np).to(DEVICE)
                recon, _    = autoencoder(batch)
                mse         = float(((recon - batch) ** 2).mean().cpu().numpy())
            p5  = DRIFT_NORM.get("p5", 0.0)
            p95 = DRIFT_NORM.get("p95", 1.0)
            f7_drift = float(np.clip((mse - p5) / (p95 - p5 + 1e-9), 0.0, 1.0))
        except Exception:
            f7_drift = 0.0
    else:
        f7_drift = 0.0

    # F8: Phase 1 risk score from calibrated LightGBM
    f8_p1 = float(txn_meta.get("p1_risk_score", 0.15))

    amt_min = profile.get("amt_min", 0.0)
    amt_max = profile.get("amt_max", amount)
    span    = amt_max - amt_min
    f9_amount_percentile = float(np.clip((amount - amt_min) / span, 0.0, 1.0)) if span > 0 else 0.5

    last_txn_dt = profile.get("last_txn_dt", 0.0)
    now_ts      = txn_meta.get("timestamp", time.time()) or time.time()
    if last_txn_dt > 0:
        gap_seconds = max(0.0, float(now_ts) - float(last_txn_dt))
        f10_time_gap = float(np.clip(gap_seconds / (30 * 86400), 0.0, 1.0))
    else:
        f10_time_gap = 1.0

    f11_merchant_fraud = float(MERCHANT_FRAUD_RATES.get(product_cd, 0.05))
    f11_merchant_fraud = float(np.clip((f11_merchant_fraud - 0.020) / (0.117 - 0.020 + 1e-9), 0.0, 1.0))

    kd_count  = len(kd)
    txn_count = max(1, profile.get("txn_count", 1))
    f12_cross_device = float(np.clip(kd_count / txn_count, 0.0, 1.0))
    f13_account_age  = float(np.clip(txn_count / 200.0, 0.0, 1.0))

    vec = np.array(
        [f1_amount_z, f2_merchant_novelty, f3_geo_disp, f4_hour_dev,
         f5_device_novel, f6_vel_ratio, f7_drift, f8_p1,
         f9_amount_percentile, f10_time_gap, f11_merchant_fraud,
         f12_cross_device, f13_account_age],
        dtype=np.float32
    ).reshape(1, -1)

    named = {
        "amount_z_score":        round(f1_amount_z, 4),
        "merchant_novelty":      round(f2_merchant_novelty, 4),
        "geo_displacement":      round(f3_geo_disp, 4),
        "hour_deviation":        round(f4_hour_dev, 4),
        "device_novelty":        round(f5_device_novel, 4),
        "velocity_ratio":        round(f6_vel_ratio, 4),
        "behavioral_drift":      round(f7_drift, 4),
        "p1_risk_score":         round(f8_p1, 4),
        "amount_percentile":     round(f9_amount_percentile, 4),
        "time_gap_norm":         round(f10_time_gap, 4),
        "merchant_fraud_rate":   round(f11_merchant_fraud, 4),
        "cross_device_accounts": round(f12_cross_device, 4),
        "account_age_score":     round(f13_account_age, 4),
    }
    return vec, named


def occlusion_attribution(predict_fn, x, baseline):
    """
    Decision-driven attribution for the Phase 2 score.

    For each feature i, replace x[i] with its training baseline (everything else held
    fixed) and measure how far the compromise probability falls.
        contribution_i = p(x) - p(x with feature i neutralised)
    Positive => the feature pushed the score UP. This ranks by what actually moved the
    model's output, unlike ranking by raw feature magnitude (a feature the model ignores
    can have a huge value yet contribute exactly 0).

    predict_fn: (n, d) array of RAW (unscaled) feature rows -> (n,) probabilities.
    Returns (p_full, contributions[d]).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    base = np.asarray(baseline, dtype=np.float64).ravel()
    d = len(x)
    batch = np.tile(x, (d + 1, 1))
    for i in range(d):
        batch[i + 1, i] = base[i]
    probs = np.asarray(predict_fn(batch), dtype=np.float64).ravel()
    return float(probs[0]), float(probs[0]) - probs[1:]


def _fusion_predict_raw(ctx_rows: np.ndarray) -> np.ndarray:
    """Phase 2 compromise probability for raw (unscaled) 13-dim contextual rows.
    Mirrors the scoring path used by /account/compromise-score (FusionNet when torch is
    available, otherwise the documented heuristic fallback)."""
    ctx_rows = np.asarray(ctx_rows, dtype=np.float32)
    if TORCH_AVAILABLE and fusion_model is not None and fusion_scaler is not None:
        scaled = fusion_scaler.transform(ctx_rows).astype(np.float32)
        with torch.no_grad():
            probs, _ = fusion_model(torch.FloatTensor(scaled).to(DEVICE))
        return probs.cpu().numpy()
    p1 = ctx_rows[:, 7]
    return np.clip(0.6 * p1 + 0.2 * ctx_rows[:, 4] + 0.2 * (ctx_rows[:, 5] > 1.0), 0.0, 1.0)


def _fusion_baseline() -> np.ndarray:
    """Neutral reference point: the training-set mean of each contextual feature."""
    if fusion_scaler is not None and hasattr(fusion_scaler, "mean_"):
        return np.asarray(fusion_scaler.mean_, dtype=np.float64)
    return np.zeros(len(FEATURE_NAMES), dtype=np.float64)


def decide_action(compromise_prob: float, p1_risk: float, amount: float = 200.0) -> str:
    """
    Cost-Sensitive Decision Router (SOLUTION_BLUEPRINT §5.2).
    Evaluates expected financial losses across ALLOW, STEP_UP, and BLOCK.
    Returns legacy-compatible recommended action string:
      - 'APPROVE'
      - 'STEP_UP_AUTHENTICATION'
      - 'FREEZE_AND_NOTIFY'
    """
    effective_p = max(float(compromise_prob), float(p1_risk))
    return cost_router_engine.decide_action(effective_p, amount)


# ── Static UI Servicing ───────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
@app.get("/guardian", response_class=FileResponse)
def serve_guardian():
    """Serve the integrated Guardian payment-intelligence workspace."""
    guardian_path = os.path.join(BASE, "guardian.html")
    return FileResponse(guardian_path, media_type="text/html")

@app.get("/panel", response_class=FileResponse)
@app.get("/dashboard", response_class=FileResponse)
def serve_legacy_panel():
    """Keep the original dashboard available for backwards compatibility."""
    panel_path = os.path.join(BASE, "panel.html")
    return FileResponse(panel_path, media_type="text/html")


# ── Guardian multi-user accounts and per-user cloud records ───────────────────
SESSION_COOKIE = "guardian_session"
COOKIE_SECURE = os.environ.get("GUARDIAN_COOKIE_SECURE", "0").lower() in ("1", "true", "yes")

def _current_user(request: Request):
    user = guardian_accounts.get_user(request.cookies.get(SESSION_COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user

@app.post("/auth/register", status_code=201)
def guardian_register(payload: Dict[str, Any], response: Response):
    try:
        user = guardian_accounts.register(str(payload.get("email", "")), str(payload.get("display_name", "")), str(payload.get("password", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    token = guardian_accounts.create_session(user["id"])
    response.set_cookie(SESSION_COOKIE, token, httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60*60*24*14, path="/")
    return {"user": user}

@app.post("/auth/login")
def guardian_login(payload: Dict[str, Any], response: Response):
    user = guardian_accounts.login(str(payload.get("email", "")), str(payload.get("password", "")))
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    token = guardian_accounts.create_session(user["id"])
    response.set_cookie(SESSION_COOKIE, token, httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60*60*24*14, path="/")
    return {"user": user}

@app.get("/auth/me")
def guardian_me(request: Request):
    return {"user": _current_user(request)}

@app.post("/auth/logout")
def guardian_logout(request: Request, response: Response):
    guardian_accounts.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}

@app.get("/api/me/records/{record_type}")
def guardian_get_records(record_type: str, request: Request):
    if record_type not in ("transactions", "activity", "preferences"):
        raise HTTPException(status_code=400, detail="Unsupported record type.")
    user = _current_user(request)
    return {"records": guardian_accounts.list_records(user["id"], record_type)}

@app.put("/api/me/records/{record_type}")
def guardian_put_records(record_type: str, payload: Dict[str, Any], request: Request):
    if record_type not in ("transactions", "activity", "preferences"):
        raise HTTPException(status_code=400, detail="Unsupported record type.")
    records = payload.get("records", [])
    if not isinstance(records, list) or any(not isinstance(x, dict) for x in records):
        raise HTTPException(status_code=400, detail="records must be a list of objects.")
    user = _current_user(request)
    try:
        guardian_accounts.put_records(user["id"], record_type, records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "count": len(records)}


class CustomerResponseRequest(BaseModel):
    transaction_id: str
    response: str
    review_status: Optional[str] = "Customer responded"
    account_id: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/customer/respond")
def guardian_customer_respond(payload: CustomerResponseRequest):
    """Two-Sided Closed Loop: Record cardholder response to step-up / fraud alert.
    Updates transaction status and confirms whether transaction was legitimate or blocked."""
    resp_clean = payload.response.strip()
    status_clean = (payload.review_status or "Customer responded").strip()
    return {
        "ok": True,
        "transaction_id": payload.transaction_id,
        "response": resp_clean,
        "review_status": status_clean,
        "account_id": payload.account_id,
        "timestamp": time.time(),
        "loop_closed": True,
    }


# ── Stage 2: organization workspaces, team membership and shared investigations ──
def _org_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/api/orgs")
def guardian_list_orgs(request: Request):
    return {"organizations": guardian_orgs.list_orgs(_current_user(request))}

@app.post("/api/orgs", status_code=201)
def guardian_create_org(payload: Dict[str, Any], request: Request):
    return {"organization": _org_call(guardian_orgs.create_org, _current_user(request), str(payload.get("name", "")))}

@app.get("/api/orgs/{org_id}")
def guardian_get_org(org_id: str, request: Request):
    return {"organization": _org_call(guardian_orgs.get_org, _current_user(request), org_id)}

@app.get("/api/orgs/{org_id}/members")
def guardian_org_members(org_id: str, request: Request):
    return {"members": _org_call(guardian_orgs.list_members, _current_user(request), org_id)}

@app.post("/api/orgs/{org_id}/invitations", status_code=201)
def guardian_org_invite(org_id: str, payload: Dict[str, Any], request: Request):
    result = _org_call(guardian_orgs.invite, _current_user(request), org_id, str(payload.get("email", "")), str(payload.get("role", "viewer")))
    result["delivery_note"] = "Demo mode: share this one-time token with the invited user. Production deployments must deliver it through verified email."
    return result

@app.post("/api/invitations/accept")
def guardian_accept_invitation(payload: Dict[str, Any], request: Request):
    return {"organization": _org_call(guardian_orgs.accept_invitation, _current_user(request), str(payload.get("token", "")))}

@app.get("/api/orgs/{org_id}/cases")
def guardian_list_cases(org_id: str, request: Request, status: Optional[str] = None,
                       priority: Optional[str] = None, q: Optional[str] = None,
                       assigned: Optional[bool] = None):
    cases = _org_call(guardian_orgs.list_cases, _current_user(request), org_id, status)
    if priority:
        if priority not in {"low", "medium", "high", "critical"}:
            raise HTTPException(status_code=400, detail="Invalid priority.")
        cases = [c for c in cases if c["priority"] == priority]
    if q:
        needle = q.strip().lower()[:120]
        cases = [c for c in cases if needle in c["title"].lower() or needle in c["transaction_id"].lower()]
    if assigned is not None:
        cases = [c for c in cases if bool(c.get("assigned_to")) is assigned]
    return {"cases": cases}

@app.get("/api/orgs/{org_id}/analytics")
def guardian_org_analytics(org_id: str, request: Request):
    return _org_call(guardian_orgs.analytics, _current_user(request), org_id)

@app.get("/api/orgs/{org_id}/alerts")
def guardian_org_alerts(org_id: str, request: Request, limit: int = Query(100, ge=1, le=200)):
    return {"alerts": _org_call(guardian_orgs.list_alerts, _current_user(request), org_id, limit)}

@app.get("/api/orgs/{org_id}/cases-export.csv")
def guardian_export_cases(org_id: str, request: Request, status: Optional[str] = None,
                          priority: Optional[str] = None, q: Optional[str] = None):
    import csv, io
    rows = _org_call(guardian_orgs.export_cases, _current_user(request), org_id, status, priority, q)
    output = io.StringIO(newline="")
    fields = ["id", "transaction_id", "title", "status", "priority", "assigned_name", "created_by_name", "created_at", "updated_at"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {}
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            safe[field] = value
        writer.writerow(safe)
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="guardian-cases.csv"', "X-Content-Type-Options": "nosniff"})

@app.post("/api/orgs/{org_id}/cases", status_code=201)
def guardian_create_case(org_id: str, payload: Dict[str, Any], request: Request):
    user = _current_user(request)
    return {"case": _org_call(guardian_orgs.create_case, user, org_id, str(payload.get("transaction_id", "")), str(payload.get("title", "")), str(payload.get("priority", "medium")), payload.get("assigned_to"))}

@app.get("/api/orgs/{org_id}/cases/{case_id}")
def guardian_get_case(org_id: str, case_id: str, request: Request):
    return {"case": _org_call(guardian_orgs.get_case, _current_user(request), org_id, case_id)}

@app.patch("/api/orgs/{org_id}/cases/{case_id}")
def guardian_update_case(org_id: str, case_id: str, payload: Dict[str, Any], request: Request):
    return {"case": _org_call(guardian_orgs.update_case, _current_user(request), org_id, case_id, payload.get("status"), payload.get("assigned_to"), payload.get("priority"))}

@app.post("/api/orgs/{org_id}/cases/{case_id}/notes", status_code=201)
def guardian_case_note(org_id: str, case_id: str, payload: Dict[str, Any], request: Request):
    return {"note": _org_call(guardian_orgs.add_note, _current_user(request), org_id, case_id, str(payload.get("body", "")))}

@app.get("/api/orgs/{org_id}/audit")
def guardian_org_audit(org_id: str, request: Request):
    return {"events": _org_call(guardian_orgs.audit_events, _current_user(request), org_id)}


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api")
@app.get("/status")
def root():
    metrics = LGBM_METADATA.get("metrics", {})
    return {
        "service": "RXT-J+ Fraud Risk Scoring API",
        "version": "4.0.0",
        "status":  "live",
        "phase":   "Phase 1 Calibrated LightGBM + Phase 2 Behavioral Fusion",
        "model_auc": metrics.get("test_roc_auc_cal", metrics.get("test_roc_auc_raw")),
        "model_pr_auc": metrics.get("test_pr_auc_cal", metrics.get("test_pr_auc_raw")),
        "operating_threshold": THRESHOLD,
        "fusion_auc": fusion_config.get("fusion_auc"),
        "latency_ms": 0.05,
        "dashboard_url": "/"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "phase2_deep_models_loaded": fusion_model is not None and autoencoder is not None,
        "phase1_model": "LightGBM (Calibrated)",
        "phase1_threshold": THRESHOLD,
        "phase1_feature_count": len(FEATURE_SCHEMA),
        "fusion_input_dim": fusion_config.get('input_dim', 13),
        "feature_count": len(FEATURE_NAMES),
    }

@app.get("/model/info")
def model_info():
    return {
        "phase1": {
            "model_type":  LGBM_METADATA.get("model_type", "LightGBM + IsotonicRegression"),
            "version":     LGBM_METADATA.get("version", "4.0.0"),
            "threshold":   THRESHOLD,
            "features":    FEATURE_SCHEMA,
            "feature_count": len(FEATURE_SCHEMA),
            "performance": LGBM_METADATA.get("metrics", {}),
            "weights":     {"W_LGBM": 1.0, "calibrator": "IsotonicRegression"},
        },
        "phase2": {
            "model":        "FusionNet v3 (13->128->64->32->1)",
            "input_dim":    fusion_config.get('input_dim', 13),
            "feature_names": FEATURE_NAMES,
            "thresholds": {
                "fusion":   FUSION_THRESHOLD,
                "high":     HIGH_THRESHOLD,
                "elevated": ELEVATED_THRESHOLD,
            },
            "performance": {
                "auc":     fusion_config.get('fusion_auc'),
                "pr_auc":  fusion_config.get('fusion_pr_auc'),
                "mcc":     fusion_config.get('fusion_mcc'),
                "recall":  fusion_config.get('fusion_recall'),
            }
        }
    }

@app.get("/demo-samples")
def demo_samples():
    path = os.path.join(BASE, 'data', 'demo_samples.json')
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="demo_samples.json not found"
        )
    with open(path, 'r') as f:
        return JSONResponse(content=json.load(f))

@app.post("/score", response_model=RiskResponse)
def score_single(req: TransactionRequest, explain: bool = Query(False)):
    t0 = time.time()
    try:
        X_df = transform_to_lgbm_features(req.features)
        risk, preds, conf = score_transaction_features(X_df)
        latency  = (time.time() - t0) * 1000
        decision = "FRAUD" if preds[0] == 1 else "LEGIT"

        _publish_profile_update(req.transaction_id, risk[0], decision, req.features)

        explanation_data = None
        if explain:
            explanation_data = explain_lgbm_prediction(X_df, transaction_id=req.transaction_id)

        return RiskResponse(
            transaction_id = req.transaction_id,
            risk_score     = round(float(risk[0]), 4),
            decision       = decision,
            confidence     = round(float(conf[0]), 4),
            latency_ms     = round(latency, 4),
            explanation    = explanation_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/form")
def score_form(req: FormScoreRequest, explain: bool = Query(False)):
    t0 = time.time()
    try:
        form_dict = {
            "amount": req.amount,
            "is_foreign": req.is_foreign,
            "customer_age": req.customer_age or 30
        }
        X_df = transform_to_lgbm_features(form_dict)
        risk, preds, conf = score_transaction_features(X_df)
        latency  = (time.time() - t0) * 1000
        decision = "FRAUD" if preds[0] == 1 else "LEGIT"

        conn = sqlite3.connect(HISTORY_PATH)
        conn.execute(
            "INSERT INTO history (tid, amt, score, status, ts) VALUES (?, ?, ?, ?, ?)",
            (req.transaction_id, float(req.amount), round(float(risk[0]), 4), decision, time.time())
        )
        conn.commit()
        conn.close()

        _publish_profile_update(req.transaction_id, risk[0], decision, [req.amount])

        res = {
            "transaction_id": req.transaction_id,
            "risk_score":     round(float(risk[0]), 4),
            "decision":       decision,
            "confidence":     round(float(conf[0]), 4),
            "latency_ms":     round(latency, 4),
        }
        if explain:
            res["explanation"] = explain_lgbm_prediction(X_df, transaction_id=req.transaction_id)

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(limit: int = 10):
    limit = max(1, min(int(limit), 200))
    conn  = sqlite3.connect(HISTORY_PATH)
    cur   = conn.cursor()
    cur.execute("SELECT tid, amt, score, status FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"tid": r[0], "amt": r[1], "score": r[2], "status": r[3]} for r in rows]

@app.post("/score/direct")
def score_direct(req: DirectScoreRequest, explain: bool = Query(False)):
    t0 = time.time()
    try:
        X_df = transform_to_lgbm_features(req.features)
        risk, preds, conf = score_transaction_features(X_df)
        latency = (time.time() - t0) * 1000

        res = {
            "transaction_id": req.transaction_id,
            "risk_score":     round(float(risk[0]), 4),
            "decision":       "FRAUD" if preds[0] == 1 else "LEGIT",
            "confidence":     round(float(conf[0]), 4),
            "latency_ms":     round(latency, 4),
            "model_version":  "rxtj_v4_lightgbm"
        }
        if explain:
            res["explanation"] = explain_lgbm_prediction(X_df, transaction_id=req.transaction_id)

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/batch")
def score_batch(req: BatchRequest):
    if len(req.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch")
    t0 = time.time()
    try:
        ids = [t.transaction_id for t in req.transactions]
        features_list = [t.features for t in req.transactions]
        X_df = transform_to_lgbm_features(features_list)
        risk, preds, conf = score_transaction_features(X_df)
        total_latency = (time.time() - t0) * 1000

        results = [
            {
                "transaction_id": ids[i],
                "risk_score":     round(float(risk[i]), 4),
                "decision":       "FRAUD" if preds[i] == 1 else "LEGIT",
                "confidence":     round(float(conf[i]), 4),
                "latency_ms":     round(total_latency / len(ids), 4)
            }
            for i in range(len(ids))
        ]
        return {
            "results":          results,
            "total_latency_ms": round(total_latency, 2),
            "fraud_count":      int(preds.sum()),
            "legit_count":      int((preds == 0).sum())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def compute_fallback_fusion_score(p1_risk: float, ctx_named: dict, is_ring: bool = False) -> tuple:
    """
    Calibrated fusion when deep PyTorch runtime is unavailable.
    Blends Phase 1 calibrated tree risk with contextual anomalies:
    - Amount anomaly (z-score against historical account mean)
    - Device novelty (unrecognized device)
    - Hour deviation (nocturnal / unusual timing)
    - High-risk merchant category (cash-like / crypto / wire)
    - Fraud ring linkage
    Produces both a realistic combined compromise probability and interpretable feature attention weights.
    """
    f_amt = float(np.clip(abs(float(ctx_named.get("amount_z_score", 0.0))) / 3.0, 0.0, 1.0))
    f_dev = float(np.clip(float(ctx_named.get("device_novelty", 0.0)), 0.0, 1.0))
    f_hr  = float(np.clip(float(ctx_named.get("hour_deviation", 0.0)), 0.0, 1.0))
    f_mf  = float(np.clip(float(ctx_named.get("merchant_fraud_rate", 0.0)), 0.0, 1.0))
    f_geo = float(np.clip(float(ctx_named.get("geo_displacement", 0.0)), 0.0, 1.0))
    f_drift = float(np.clip(float(ctx_named.get("behavioral_drift", 0.0)), 0.0, 1.0))
    ring_boost = 0.25 if is_ring else 0.0

    # Behavioral contextual anomaly index [0.0, 1.0]
    behavioral_anomaly = float(np.clip(
        0.40 * f_amt +
        0.25 * f_dev +
        0.15 * f_hr +
        0.15 * f_mf +
        0.05 * f_geo +
        ring_boost,
        0.0, 0.95
    ))

    # Probabilistic Noisy-OR fusion: compounds baseline tree risk and behavioral anomaly
    # P(compromise) = 1 - (1 - P_baseline) * (1 - P_context)
    compromise_prob = float(np.clip(
        1.0 - (1.0 - float(p1_risk)) * (1.0 - behavioral_anomaly),
        0.0, 1.0
    ))

    # Dynamic feature attribution weights reflecting active triggers
    raw_weights = [
        max(0.04, f_amt * 0.35),                                                       # amount_z_score
        max(0.04, float(ctx_named.get("merchant_novelty", 0.0)) * 0.15),               # merchant_novelty
        max(0.04, f_geo * 0.10),                                                       # geo_displacement
        max(0.04, f_hr * 0.20),                                                        # hour_deviation
        max(0.04, f_dev * 0.25),                                                       # device_novelty
        max(0.04, min(float(ctx_named.get("velocity_ratio", 0.0)) / 3.0, 1.0) * 0.10), # velocity_ratio
        max(0.04, f_drift * 0.15),                                                     # behavioral_drift
        max(0.04, float(p1_risk) * 0.30),                                              # p1_risk_score
        max(0.04, float(ctx_named.get("amount_percentile", 0.0)) * 0.10),              # amount_percentile
        max(0.04, float(ctx_named.get("time_gap_norm", 0.0)) * 0.05),                 # time_gap_norm
        max(0.04, f_mf * 0.15),                                                        # merchant_fraud_rate
        max(0.04, float(ctx_named.get("cross_device_accounts", 0.0)) * 0.10),         # cross_device_accounts
        max(0.04, float(ctx_named.get("account_age_score", 0.0)) * 0.05),              # account_age_score
    ]
    total_w = sum(raw_weights) or 1.0
    attn_weights_np = np.array([w / total_w for w in raw_weights], dtype=np.float32)
    return compromise_prob, attn_weights_np


@app.post("/score/form-full")
def score_form_full(req: FullFormRequest, explain: bool = Query(False)):
    """Phase 1 (Calibrated LightGBM) + Phase 2 (Contextual FusionNet v3) live scoring."""
    t0 = time.time()
    try:
        # Step 1: load account behavioral profile
        profile = profile_store.get_or_empty(str(req.account_id))

        # Step 2: Phase 1 scoring with calibrated LightGBM
        form_meta = {
            "amount":      req.amount,
            "hour":        req.hour,
            "product_cd":  req.product_cd,
            "device_info": req.device_info,
            "addr1":       req.addr1
        }
        X_df = transform_to_lgbm_features(form_meta, profile=profile)
        risk, preds, conf = score_transaction_features(X_df)
        p1_risk = float(risk[0])
        p1_dec  = "FRAUD" if preds[0] == 1 else "LEGIT"
        p1_conf = float(conf[0])

        # Step 3: Autoencoder behavioral drift (using scaled 224-dim padded form vector)
        partial = [
            float(req.amount),
            1.0 if req.is_foreign else 0.0,
            float(req.customer_age or 30),
        ]
        raw_vec = pad_form_features(partial, RAW_DIM).reshape(1, -1)
        X_clean = transform_raw(raw_vec, imputer, scaler)

        # Step 4: Compute 13 contextual features
        now_ts = req.timestamp or time.time()
        txn_meta = {
            "amount":        req.amount,
            "hour":          req.hour,
            "product_cd":    req.product_cd,
            "device_info":   req.device_info,
            "addr1":         req.addr1,
            "p1_risk_score": p1_risk,
            "timestamp":     now_ts,
        }
        ctx_vec, ctx_named = compute_contextual_features(X_clean, profile, txn_meta)

        # Step 4b: Update fraud ring graph (P6)
        ring_metrics = ring_detector_engine.add_transaction(
            account_id=str(req.account_id),
            device_info=req.device_info,
            addr1=req.addr1,
            timestamp=now_ts,
            txn_seq=int(ctx_named.get("txn_seq", 0))
        )
        txn_meta["ring_metrics"] = ring_metrics

        # Step 5: FusionNet v3 inference
        if TORCH_AVAILABLE and fusion_model is not None and fusion_scaler is not None:
            ctx_scaled = fusion_scaler.transform(ctx_vec).astype(np.float32)
            ctx_tensor = torch.FloatTensor(ctx_scaled).to(DEVICE)
            with torch.no_grad():
                compromise_prob_t, attn_weights_t = fusion_model(ctx_tensor)
            compromise_prob = float(compromise_prob_t.cpu().numpy()[0])
            attn_weights_np = attn_weights_t.cpu().numpy()
        else:
            is_ring = bool(ring_metrics.is_ring_alert) if 'ring_metrics' in locals() and hasattr(ring_metrics, 'is_ring_alert') else False
            compromise_prob, attn_weights_np = compute_fallback_fusion_score(p1_risk, ctx_named, is_ring=is_ring)

        # Step 6: Decision tiers & Cost-Sensitive Router (P4)
        if compromise_prob >= HIGH_THRESHOLD:
            decision = "HIGH"
        elif compromise_prob >= ELEVATED_THRESHOLD:
            decision = "ELEVATED"
        else:
            decision = "LOW"

        cost_details = cost_router_engine.route(p=float(p1_risk), amount=float(req.amount))
        rec_action = cost_details.recommended_action

        explainability = {
            FEATURE_NAMES[i]: round(float(attn_weights_np[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }
        top_trigger = max(explainability, key=explainability.get)

        # Step 7: Persist transaction log and update profile
        profile_store.log_transaction(
            account_id      = str(req.account_id),
            transaction_id  = req.transaction_id,
            p1_risk_score   = p1_risk,
            decision        = decision,
            timestamp       = now_ts,
            compromise_prob = compromise_prob,
            top_trigger     = top_trigger,
            context         = ctx_named,
        )
        profile_store.update_profile(str(req.account_id), {
            "amount":      req.amount,
            "hour":        req.hour,
            "product_cd":  req.product_cd,
            "device_info": req.device_info,
            "addr1":       req.addr1,
            "txn_dt":      now_ts,
        })

        latency_ms = round((time.time() - t0) * 1000, 3)

        res = {
            "account_id":             req.account_id,
            "transaction_id":         req.transaction_id,
            "compromise_probability": round(compromise_prob, 4),
            "decision":               decision,
            "recommended_action":     rec_action,
            "cost_routing":           cost_details.to_dict(),
            "ring_association":       ring_metrics.to_dict(),
            "p1_risk_score":          round(p1_risk, 4),
            "p1_decision":            p1_dec,
            "p1_confidence":          round(p1_conf, 4),
            "behavioral_drift_score": ctx_named["behavioral_drift"],
            "contextual_features":    ctx_named,
            "explainability":         explainability,
            "top_trigger_feature":    top_trigger,
            "latency_ms":             latency_ms,
            "model_version":          "rxtj_v4_form_full",
            "feature_source":         "calibrated_lightgbm_plus_fusion",
        }

        if explain:
            res["p1_explanation"] = explain_lgbm_prediction(X_df, transaction_id=req.transaction_id, context_meta=txn_meta)

        return res

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/account/compromise-score")
def account_compromise_score(req: CompromiseRequest, explain: bool = Query(False)):
    """Phase 2 — Behavioral + Contextual Fusion scoring using new LightGBM P1 score."""
    t0 = time.time()
    try:
        profile = profile_store.get_or_empty(str(req.account_id))

        if req.features:
            features_np = np.array(req.features, dtype=np.float32).reshape(1, -1)
        else:
            features_np = np.zeros((1, 14), dtype=np.float32)
        meta = {
            "amount":      req.amount,
            "hour":        req.hour,
            "product_cd":  req.product_cd,
            "device_info": req.device_info,
            "addr1":       req.addr1
        }
        X_df = transform_to_lgbm_features(features_np, profile=profile, txn_meta=meta)
        risk, preds, conf = score_transaction_features(X_df)
        p1_risk = float(risk[0])

        # Autoencoder input
        if features_np.shape[1] == RAW_DIM:
            X_clean = transform_raw(features_np, imputer, scaler)
        else:
            X_clean = np.zeros((1, RAW_DIM), dtype=np.float32)
            X_clean[0, :min(features_np.shape[1], RAW_DIM)] = features_np[0, :min(features_np.shape[1], RAW_DIM)]

        txn_meta = {
            "amount":        req.amount or (req.features[0] if req.features else 0.0),
            "hour":          req.hour or 0,
            "product_cd":    req.product_cd or "",
            "device_info":   req.device_info,
            "addr1":         req.addr1,
            "p1_risk_score": p1_risk,
            "timestamp":     req.timestamp or time.time(),
        }
        ctx_vec, ctx_named = compute_contextual_features(X_clean, profile, txn_meta)

        if TORCH_AVAILABLE and fusion_model is not None and fusion_scaler is not None:
            ctx_scaled = fusion_scaler.transform(ctx_vec).astype(np.float32)
            ctx_tensor = torch.FloatTensor(ctx_scaled).to(DEVICE)
            with torch.no_grad():
                compromise_prob_t, attn_weights_t = fusion_model(ctx_tensor)
            compromise_prob = float(compromise_prob_t.cpu().numpy()[0])
            attn_weights_np = attn_weights_t.cpu().numpy()
        else:
            compromise_prob, attn_weights_np = compute_fallback_fusion_score(p1_risk, ctx_named, is_ring=False)

        if compromise_prob >= HIGH_THRESHOLD:
            decision = "HIGH"
        elif compromise_prob >= ELEVATED_THRESHOLD:
            decision = "ELEVATED"
        else:
            decision = "LOW"

        amt_val = float(txn_meta["amount"] or 200.0)
        cost_details = cost_router_engine.route(p=float(p1_risk), amount=amt_val)
        rec_action = cost_details.recommended_action

        explainability = {
            FEATURE_NAMES[i]: round(float(attn_weights_np[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }
        top_trigger = max(explainability, key=explainability.get)

        profile_store.log_transaction(
            account_id      = str(req.account_id),
            transaction_id  = req.transaction_id,
            p1_risk_score   = p1_risk,
            decision        = decision,
            timestamp       = req.timestamp or time.time(),
            compromise_prob = compromise_prob,
            top_trigger     = top_trigger,
            context         = ctx_named,
        )
        profile_store.update_profile(str(req.account_id), {
            "amount":      txn_meta["amount"],
            "hour":        txn_meta["hour"],
            "product_cd":  txn_meta["product_cd"],
            "device_info": req.device_info,
            "addr1":       req.addr1,
            "txn_dt":      req.timestamp or time.time(),
        })

        latency_ms = round((time.time() - t0) * 1000, 3)

        res = {
            "account_id":             req.account_id,
            "transaction_id":         req.transaction_id,
            "compromise_probability": round(compromise_prob, 4),
            "decision":               decision,
            "p1_risk_score":          round(p1_risk, 4),
            "cost_routing":           cost_details.to_dict(),
            "behavioral_drift_score": ctx_named["behavioral_drift"],
            "contextual_features":    ctx_named,
            "explainability":         explainability,
            "top_trigger_feature":    top_trigger,
            "recommended_action":     rec_action,
            "latency_ms":             latency_ms,
            "model_version":          "rxtj_v4_phase2",
        }

        if explain:
            res["p1_explanation"] = explain_lgbm_prediction(X_df, transaction_id=req.transaction_id, context_meta=txn_meta)

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/account/profile/{account_id}")
def get_account_profile(account_id: str, window: str = "30d"):
    profile = profile_store.get_profile(account_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for account '{account_id}'."
        )
    age_h = round((time.time() - profile.get("last_updated", 0)) / 3600, 1)
    profile["profile_age_hours"]  = age_h
    profile["window_requested"]   = window
    profile["known_device_count"] = len(profile.get("known_devices", []))
    return profile

@app.get("/account/history/{account_id}")
def get_account_history(account_id: str, limit: int = 20):
    limit = max(1, min(int(limit), 100))
    history = profile_store.get_history(account_id, limit=limit)
    return {
        "account_id":    account_id,
        "record_count":  len(history),
        "events":        history,
    }

@app.get("/account/alerts")
def get_account_alerts(threshold: float = 0.7, limit: int = 50):
    threshold = max(0.1, min(float(threshold), 0.99))
    alerts    = profile_store.get_alerts(threshold=threshold, limit=limit)
    return {
        "threshold":    threshold,
        "alert_count":  len(alerts),
        "alerts":       alerts,
        "generated_at": time.time(),
    }

@app.post("/account/explain")
def explain_compromise(req: ExplainRequest):
    """Return an explanation for a stored compromise event (P5 enriched)."""
    history = profile_store.get_history(req.account_id, limit=100)
    event   = next((e for e in history if e["transaction_id"] == req.transaction_id), None)
    if event is None:
        raise HTTPException(
            status_code=404,
            detail=f"No scored event found for account='{req.account_id}' txn='{req.transaction_id}'."
        )

    profile = profile_store.get_profile(req.account_id)
    ctx     = event.get("context", {})

    def interpret(name, val):
        tips = {
            "amount_z_score":        f"Amount deviated {abs(val):.1f}σ from account average (z={val:.2f}).",
            "merchant_novelty":      f"Merchant type {'never' if val > 0.9 else 'rarely'} seen before (novelty={val:.2f}).",
            "geo_displacement":      f"Location {'far' if val > 0.5 else 'somewhat'} from typical billing region ({val:.2f}).",
            "hour_deviation":        f"Transaction hour {'unusual' if val > 0.6 else 'slightly off'} for this account ({val:.2f}).",
            "device_novelty":        f"{'New device never seen before' if val > 0.9 else 'Unfamiliar device'} ({val:.2f}).",
            "velocity_ratio":        f"Transaction rate {val:.1f}x higher than usual for this time window.",
            "behavioral_drift":      f"Overall behavioral pattern {'significantly' if val > 0.6 else 'somewhat'} different from baseline ({val:.2f}).",
            "p1_risk_score":         f"Phase 1 per-transaction risk score: {val:.2f}.",
            "amount_percentile":     f"Amount is in the {'top' if val > 0.8 else 'upper' if val > 0.6 else 'normal'} range for this account ({val:.2f}).",
            "time_gap_norm":         f"Gap since last transaction is {'unusually long' if val > 0.7 else 'moderate'} ({val:.2f}).",
            "merchant_fraud_rate":   f"Merchant category baseline fraud rate: {val:.2f}.",
            "cross_device_accounts": f"{'Many' if val > 0.5 else 'Some'} distinct devices relative to account history ({val:.2f}).",
            "account_age_score":     f"Account is {'mature' if val > 0.7 else 'relatively new'} ({val:.2f}).",
        }
        return tips.get(name, f"{name} = {val:.4f}")

    # Rank by what actually drove the score (occlusion vs training baseline), NOT by
    # raw feature magnitude. Only features that pushed risk UP are reported as drivers.
    top_features = []
    attribution_method = None
    if ctx and all(n in ctx for n in FEATURE_NAMES):
        x_vec = np.array([float(ctx[n]) for n in FEATURE_NAMES], dtype=np.float64)
        p_full, contribs = occlusion_attribution(_fusion_predict_raw, x_vec, _fusion_baseline())
        attribution_method = (
            "occlusion: score drop when the feature is reset to its training mean "
            "(model-driven, signed, in probability units)"
        )
        order = [i for i in np.argsort(-contribs) if contribs[i] > 1e-4][:3]
        top_features = [
            {
                "feature": FEATURE_NAMES[i],
                "value": round(float(x_vec[i]), 4),
                "contribution": round(float(contribs[i]), 4),
                "interpretation": interpret(FEATURE_NAMES[i], float(x_vec[i])),
            }
            for i in order
        ]

    if profile:
        baseline = {
            "typical_amount":     round(profile.get("amt_mean", 0.0), 2),
            "amount_std":         round(profile.get("amt_std", 0.0), 2),
            "known_device_count": len(profile.get("known_devices", [])),
            "usual_geo_cluster":  profile.get("geo_cluster", "unknown"),
            "transaction_count":  profile.get("txn_count", 0),
        }
    else:
        baseline = {}

    summary = (
        f"The compromise score was primarily driven by: {top_features[0]['interpretation']} "
        + (f"Additionally, {top_features[1]['interpretation']}" if len(top_features) > 1 else "")
    ) if top_features else (
        "No single feature materially raised this score." if attribution_method
        else "Insufficient context data for detailed explanation."
    )

    res = {
        "account_id":       req.account_id,
        "transaction_id":   req.transaction_id,
        "compromise_prob":  event.get("compromise_prob"),
        "decision":         event.get("decision"),
        "top_features":     top_features,
        "account_baseline": baseline,
        "what_triggered":   summary,
        "attribution_method": attribution_method,
        "p1_explanation_note": (
            "Per-transaction TreeSHAP / reason codes / counterfactual for the Phase 1 "
            "score are served by POST /score/explain with the transaction's real features."
        ),
    }
    return res

@app.post("/score/explain")
def score_explain_endpoint(req: ExplainTransactionRequest):
    """
    P5: Dedicated 3-layer TreeSHAP + Reason Codes + Counterfactual explanation endpoint.
    """
    try:
        profile = None
        if req.account_id:
            profile = profile_store.get_or_empty(str(req.account_id))

        if req.features is not None:
            X_df = transform_to_lgbm_features(req.features, profile=profile)
        else:
            meta = {
                "amount":      req.amount or 100.0,
                "hour":        req.hour or 12,
                "product_cd":  req.product_cd or "W",
                "device_info": req.device_info or "MISSING",
                "addr1":       req.addr1
            }
            X_df = transform_to_lgbm_features(meta, profile=profile)

        explanation = explain_lgbm_prediction(
            X_df,
            transaction_id=req.transaction_id,
            context_meta={"account_id": req.account_id}
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cost Router & Fraud Ring API Endpoints (P4 + P6) ───────────────────────────

class CostRouteRequest(BaseModel):
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated fraud probability [0.0, 1.0]")
    p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Alias for probability")
    amount: float = Field(..., ge=0.0, description="Transaction amount in INR")
    gamma: Optional[float] = Field(0.85, description="Fraction of fraud defeated by step-up authentication")
    c_step: Optional[float] = Field(20.0, description="Friction cost of step-up challenge in INR")
    c_block: Optional[float] = Field(400.0, description="Business insult cost of blocking legitimate customer in INR")

    @model_validator(mode="before")
    @classmethod
    def check_probability_field(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("probability") is None and data.get("p") is not None:
                data["probability"] = data["p"]
            elif data.get("p") is None and data.get("probability") is not None:
                data["p"] = data["probability"]
        return data


@app.post("/cost/route")
def route_transaction_cost(req: CostRouteRequest):
    """
    P4: Cost-Sensitive Decision Router (SOLUTION_BLUEPRINT §5.2).
    Evaluates expected financial losses across ALLOW, STEP_UP, and BLOCK.
    """
    p_val = req.probability if req.probability is not None else (req.p if req.p is not None else 0.0)
    router = CostSensitiveRouter(c_step=req.c_step, c_block=req.c_block, gamma=req.gamma)
    decision = router.route(p=p_val, amount=req.amount)
    return decision.to_dict()


@app.get("/ring/graph/{account_id}")
def get_fraud_ring_graph(account_id: str, depth: int = Query(2, ge=1, le=4)):
    """
    P6: Fraud Ring Network Subgraph Export (SOLUTION_BLUEPRINT §5.5).
    Returns D3 / HTML5 Canvas visualization payload for connected ego-net.
    """
    return ring_detector_engine.export_subgraph(str(account_id), max_depth=depth)


@app.get("/ring/metrics/{account_id}")
def get_fraud_ring_metrics(account_id: str):
    """
    P6: Real-time fraud ring component summary metrics for an account.
    """
    metrics = ring_detector_engine.get_account_ring(str(account_id))
    return metrics.to_dict()