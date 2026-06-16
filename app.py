# ============================================================
# app.py — RXT-J+ FastAPI Deployment  (Phase 2 — Zero-Bug)
# Run with: python -m uvicorn app:app --reload --port 8000
#
# FIXES applied vs previous version:
#   [BUG-1] FusionNet architecture: matched to trained weights
#           net: Linear(13,128)->ReLU->Dropout->Linear(128,64)->ReLU
#                ->Dropout->Linear(64,32)->ReLU->Linear(32,1)
#           (previous code had 32->16->1 which caused shape mismatch)
#   [BUG-2] compute_contextual_features now produces all 13 features:
#           added amount_percentile, time_gap_norm, merchant_fraud_rate,
#           cross_device_accounts, account_age_score
#   [BUG-3] explain endpoint interpret() updated for all 13 feature names
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from profile_store import ProfileStore
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import time
import os
import sys
import sqlite3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_preprocessors, transform_raw, pad_form_features, raw_feature_dim

DEVICE      = torch.device('cpu')
SEQ_LEN     = 8
CARDINALITY = 4
HISTORY_DB  = 'fraud_storage.db'
BASE        = os.path.dirname(os.path.abspath(__file__))

# ── Model architectures ───────────────────────────────────────────────────────

class ResNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cardinality=CARDINALITY):
        super().__init__()
        group_dim = out_dim // cardinality
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, group_dim), nn.BatchNorm1d(group_dim), nn.ReLU(),
                nn.Linear(group_dim, group_dim), nn.BatchNorm1d(group_dim)
            ) for _ in range(cardinality)
        ])
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.cat([p(x) for p in self.paths], dim=-1) + self.shortcut(x))

class ResNeXtExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResNeXtBlock(input_dim, 128), ResNeXtBlock(128, 128),
            ResNeXtBlock(128, 64),        ResNeXtBlock(64,  64)
        )
    def forward(self, x): return self.net(x)

class SelfAttentionGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len  = seq_len
        self.step_dim = input_dim // seq_len
        self.gru      = nn.GRU(self.step_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(),
                                       nn.Linear(32, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(32, 1), nn.Identity())
    def forward(self, x):
        x      = x.view(x.size(0), self.seq_len, self.step_dim)
        out, _ = self.gru(x)
        attn   = torch.softmax(self.attention(out), dim=1)
        ctx    = (attn * out).sum(dim=1)
        return self.classifier(ctx).squeeze(1), attn.squeeze(-1)

class AttentionRXTJ(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN):
        super().__init__()
        self.resnext  = ResNeXtExtractor(input_dim)
        self.attn_gru = SelfAttentionGRU(input_dim=64, seq_len=seq_len)
    def forward(self, x):
        return self.attn_gru(self.resnext(x))

class AutoencoderNet(nn.Module):
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


class ResBlock(nn.Module):
    """Residual block used by FraudResNet (must match resnet_extractor.pt weights)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class FraudResNet(nn.Module):
    """Feature extractor trained in notebook 03 (resnet_extractor.pt).
    .features() returns a 64-dim representation from 224-dim scaled input.
    """
    def __init__(self, input_dim, out_dim=64):
        super().__init__()
        self.stem   = nn.Linear(input_dim, 128)
        self.blocks = nn.Sequential(
            ResBlock(128, 128), ResBlock(128, 64),
            ResBlock(64,  64),  ResBlock(64, out_dim)
        )
        self.head = nn.Linear(out_dim, 2)
    def features(self, x):
        return self.blocks(torch.relu(self.stem(x)))
    def forward(self, x):
        return self.head(self.features(x))


# ── FusionNet v3 — architecture MUST match fusion_net.pt ─────────────────────
# Trained weights layout (verified from state_dict):
#   feature_attn : [13]
#   net.0.weight : [128, 13]   → Linear(13, 128)
#   net.3.weight : [64, 128]   → Linear(128, 64)
#   net.6.weight : [32, 64]    → Linear(64, 32)
#   net.8.weight : [1, 32]     → Linear(32, 1)
# ─────────────────────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    """FusionNet v3 — Attention-weighted MLP for account compromise scoring.

    Architecture is fixed to match the saved weights in fusion_net.pt.
    Input dim MUST be 13 (the 13 contextual features computed by
    compute_contextual_features()).
    """
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.feature_attn = nn.Parameter(torch.ones(input_dim))
        # Layers match the state_dict: 13→128→64→32→1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # net.0
            nn.ReLU(),                   # net.1
            nn.Dropout(0.3),             # net.2
            nn.Linear(128, 64),          # net.3
            nn.ReLU(),                   # net.4
            nn.Dropout(0.3),             # net.5
            nn.Linear(64, 32),           # net.6
            nn.ReLU(),                   # net.7
            nn.Linear(32, 1)             # net.8
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.feature_attn, dim=0)
        x_weighted   = x * attn_weights
        logit        = self.net(x_weighted).squeeze(1)
        return torch.sigmoid(logit), attn_weights


# ── Load artefacts ────────────────────────────────────────────────────────────

def load_models():
    config   = json.load(open(os.path.join(BASE, 'results/deployment_config.json')))
    imputer, scaler = load_preprocessors(BASE)
    nystroem = joblib.load(os.path.join(BASE, 'models/nystroem.pkl'))
    ipca     = joblib.load(os.path.join(BASE, 'models/incremental_pca.pkl'))
    ifm      = joblib.load(os.path.join(BASE, 'models/isolation_forest.pkl'))

    input_dim = ipca.n_components_
    model     = AttentionRXTJ(input_dim).to(DEVICE)
    state     = torch.load(os.path.join(BASE, 'models/attention_rxtj.pt'), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Autoencoder for Phase 2 behavioural drift scoring.
    ae_state  = torch.load(os.path.join(BASE, 'models/autoencoder.pt'), map_location=DEVICE)
    ae_input_dim = ae_state['encoder.0.weight'].shape[1]
    autoencoder  = AutoencoderNet(ae_input_dim).to(DEVICE)
    autoencoder.load_state_dict(ae_state)
    autoencoder.eval()

    # ResNet feature extractor — needed for raw→EARN+ pipeline (224→64 features).
    # Concatenated with AE latents (64) to form the 128-dim Nystroem input.
    rn_input_dim = ae_input_dim   # same 224-dim scaled input
    resnet_ext   = FraudResNet(rn_input_dim).to(DEVICE)
    rn_state     = torch.load(os.path.join(BASE, 'models/resnet_extractor.pt'), map_location=DEVICE)
    resnet_ext.load_state_dict(rn_state)
    resnet_ext.eval()

    # Phase 2: FusionNet v3 + scaler
    fusion_config_data = json.load(open(os.path.join(BASE, 'results/fusion_config.json')))
    fusion_scaler_obj  = joblib.load(os.path.join(BASE, 'models/fusion_scaler.pkl'))

    # Always use the input_dim from the saved config (13), not a hard-coded value
    f_input_dim      = fusion_config_data['input_dim']  # must be 13
    fusion_model_obj = FusionNet(f_input_dim).to(DEVICE)
    fusion_state     = torch.load(os.path.join(BASE, 'models/fusion_net.pt'), map_location=DEVICE)
    fusion_model_obj.load_state_dict(fusion_state)   # will raise if arch wrong
    fusion_model_obj.eval()

    return (model, imputer, scaler, nystroem, ipca, ifm, autoencoder, resnet_ext, config,
            fusion_model_obj, fusion_scaler_obj, fusion_config_data)


print("Loading RXT-J+ models...")
(model, imputer, scaler, nystroem, ipca, ifm,
 autoencoder, resnet_ext, config,
 fusion_model, fusion_scaler, fusion_config) = load_models()

RAW_DIM   = raw_feature_dim(imputer)
W_MODEL   = config['W_MODEL']
W_IFM     = config['W_IFM']
THRESHOLD = config['THRESHOLD']

FUSION_THRESHOLD   = fusion_config['fusion_threshold']
HIGH_THRESHOLD     = fusion_config['high_threshold']
ELEVATED_THRESHOLD = fusion_config['elevated_threshold']
FEATURE_NAMES      = fusion_config['feature_names']   # 13 names
DRIFT_NORM         = json.load(open(os.path.join(BASE, 'data/drift_norm_params.json')))
MERCHANT_FRAUD_RATES = json.load(open(os.path.join(BASE, 'data/merchant_fraud_rates.json')))

print(f"Models loaded. W_MODEL={W_MODEL:.4f}, W_IFM={W_IFM:.4f}, Threshold={THRESHOLD:.2f}")
print(f"Autoencoder loaded (input_dim={autoencoder.encoder[0].in_features}).")
print(f"ResNet extractor loaded (input_dim={resnet_ext.stem.in_features}, output_dim=64).")
print(f"EARN+ pipeline: {autoencoder.encoder[0].in_features}→AE(64)+RN(64)→concat(128)→Nystroem→IPCA({ipca.n_components_}).")
print(f"FusionNet v3 loaded (input_dim={fusion_config['input_dim']}). "
      f"High={HIGH_THRESHOLD}, Elevated={ELEVATED_THRESHOLD}")
print(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# ── SQLite history store ─────────────────────────────────────────────────────

HISTORY_PATH = os.path.join(BASE, HISTORY_DB)

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

profile_store = ProfileStore(
    db_path=os.path.join(BASE, "data", "behavioral_profiles.db")
)
print(f"  [Phase2] ProfileStore ready. Accounts: {profile_store.coverage_stats()['total_accounts']:,}")

# ── Kafka producer (optional, non-blocking) ──────────────────────────────────

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
        print(f"Kafka producer ready → {KAFKA_BOOTSTRAP} topic={KAFKA_TOPIC}")
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
    description="Real-time payment fraud detection — Attention-RXT-J+ with Jaya optimisation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],   # null allows file:// access from demo HTML
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    features: List[float] = Field(..., description="Preprocessed feature vector")

class DirectScoreRequest(BaseModel):
    transaction_id: str
    features: List[float] = Field(..., description="50-dim EARN+ feature vector (post-IPCA)")

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]

class FormScoreRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    amount:         float
    is_foreign:     bool = False
    customer_age:   Optional[int] = 30

class RiskResponse(BaseModel):
    transaction_id: str
    risk_score:  float
    decision:    str
    confidence:  float
    latency_ms:  float

class CompromiseRequest(BaseModel):
    account_id:     str
    transaction_id: str
    features:       List[float]
    amount:         Optional[float] = 0.0
    hour:           Optional[int]   = 0
    product_cd:     Optional[str]   = ""
    device_info:    Optional[str]   = None
    addr1:          Optional[str]   = None
    timestamp:      Optional[float] = None

class ExplainRequest(BaseModel):
    account_id:     str
    transaction_id: str

class FullFormRequest(BaseModel):
    """Human-readable inputs for /score/form-full.

    No pre-computed feature vector needed — the endpoint synthesises one
    from the form fields using the training-time imputer column means, so
    every field change produces a genuinely different model output.
    """
    account_id:     str             = Field(...,  example="ACC_42")
    transaction_id: str             = Field(...,  example="TXN_001")
    amount:         float           = Field(...,  example=289.99, description="Transaction amount in USD")
    hour:           int             = Field(14,   ge=0, le=23,    description="Hour of day (0-23, local)")
    product_cd:     str             = Field("W",  example="W",    description="Product code: W C H R S")
    is_foreign:     bool            = Field(False,                description="True if card-not-present / international")
    customer_age:   Optional[int]   = Field(30,   ge=0, le=120,   description="Cardholder age (used to fill raw features)")
    device_info:    Optional[str]   = Field(None, example="MacOS")
    addr1:          Optional[str]   = Field(None, example="299",  description="Billing postal/zip code")
    timestamp:      Optional[float] = Field(None,                 description="Unix timestamp (defaults to now)")

# ── Core Scorers & Helpers ────────────────────────────────────────────────────

def score_earn_features(Z_earn: np.ndarray):
    """Score 50-dim EARN+ features directly — no Nystroem/IPCA needed."""
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(Z_earn).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    ifm_raw  = ifm.score_samples(Z_earn)
    ifm_min, ifm_max = ifm_raw.min(), ifm_raw.max()
    ifm_norm = 1 - (ifm_raw - ifm_min) / (ifm_max - ifm_min + 1e-9)
    ifm_norm = np.clip(ifm_norm, 0, 1)

    total = W_MODEL + W_IFM + 1e-9
    risk  = np.clip((W_MODEL / total) * probs + (W_IFM / total) * ifm_norm, 0, 1)
    preds = (risk > THRESHOLD).astype(int)
    conf  = np.where(preds == 1, risk, 1 - risk)
    return risk, preds, conf


def extract_earn_features(X_clean: np.ndarray) -> np.ndarray:
    """Convert a 224-dim imputed+scaled feature matrix into 50-dim EARN+ space.

    Full pipeline (matches notebook 03 training exactly):
      1. AE encoder:    X_clean (224) → latent z (64)
      2. ResNet extractor: X_clean (224) → features (64)
      3. Concatenate:   [z | features] → (128)
      4. Nystroem RBF:  (128) → (300)
      5. IncrementalPCA:(300) → (50)  ← this is EARN+ / Z_earn

    Args:
        X_clean: float32 array of shape (n, 224) — imputer+scaler output.
    Returns:
        Z_earn: float32 array of shape (n, 50).
    """
    X_t = torch.FloatTensor(X_clean).to(DEVICE)
    with torch.no_grad():
        # AE latent (64-dim)
        _, z_ae = autoencoder(X_t)
        z_ae = z_ae.cpu().numpy()                    # (n, 64)
        # ResNet features (64-dim)
        z_rn = resnet_ext.features(X_t).cpu().numpy()  # (n, 64)

    Z_concat = np.concatenate([z_ae, z_rn], axis=1).astype(np.float32)  # (n, 128)
    Z_nys    = nystroem.transform(Z_concat)                               # (n, 300)
    Z_earn   = ipca.transform(Z_nys).astype(np.float32)                  # (n, 50)
    return Z_earn


def score_raw_features(features_np: np.ndarray):
    """Score raw 224-dim features through the full EARN+ pipeline.

    Pipeline: imputer+scaler → AE encoder+ResNet → concat 128 → Nystroem → IPCA → model+IFM
    """
    X_clean = transform_raw(features_np, imputer, scaler)   # (n, 224)
    Z_earn  = extract_earn_features(X_clean)                 # (n, 50)
    return score_earn_features(Z_earn)


def compute_contextual_features(
    features_np: np.ndarray,
    profile: dict,
    txn_meta: dict,
) -> tuple:
    """Compute the 13-dim contextual feature vector for FusionNet v3.

    Feature order (must match fusion_config['feature_names']):
      0  amount_z_score
      1  merchant_novelty
      2  geo_displacement
      3  hour_deviation
      4  device_novelty
      5  velocity_ratio
      6  behavioral_drift
      7  p1_risk_score
      8  amount_percentile        ← NEW (was missing)
      9  time_gap_norm            ← NEW (was missing)
      10 merchant_fraud_rate      ← NEW (was missing)
      11 cross_device_accounts    ← NEW (was missing)
      12 account_age_score        ← NEW (was missing)
    """
    amount     = float(txn_meta.get("amount", 0.0))
    hour       = int(txn_meta.get("hour", 0)) % 24
    product_cd = str(txn_meta.get("product_cd", ""))
    device     = str(txn_meta.get("device_info", "")) if txn_meta.get("device_info") else ""
    addr1      = str(txn_meta.get("addr1", ""))

    # ── F1: amount z-score ────────────────────────────────────────────────
    amt_std = max(profile.get("amt_std", 1.0), 0.1)
    f1_amount_z = float(np.clip(
        (amount - profile.get("amt_mean", 0.0)) / amt_std, -5.0, 5.0
    ))

    # ── F2: merchant novelty ──────────────────────────────────────────────
    mc       = profile.get("merchant_counts", {})
    max_freq = max(mc.values()) if mc else 1
    freq     = mc.get(product_cd, 0)
    f2_merchant_novelty = 1.0 - (freq / (max_freq + 1e-9)) if mc else 1.0

    # ── F3: geo displacement ──────────────────────────────────────────────
    geo_cluster = profile.get("geo_cluster", "")
    try:
        disp = abs(float(addr1) - float(geo_cluster)) if (addr1 and geo_cluster) else 0.0
        f3_geo_disp = float(min(disp / 500.0, 1.0))
    except (ValueError, TypeError):
        f3_geo_disp = 0.5

    # ── F4: hour deviation ────────────────────────────────────────────────
    hh     = profile.get("hour_hist", [1/24]*24)
    hh_val = hh[hour] if len(hh) == 24 else 1/24
    f4_hour_dev = float(max(0.0, 1.0 - hh_val * 24))

    # ── F5: device novelty ────────────────────────────────────────────────
    kd = profile.get("known_devices", [])
    f5_device_novel = 0.0 if (device and device in kd) else (1.0 if device else 0.5)

    # ── F6: velocity ratio ────────────────────────────────────────────────
    vel_1h  = float(profile.get("velocity_1h", 0))
    vel_24h = float(profile.get("velocity_24h", 0))
    f6_vel_ratio = float(np.clip(vel_1h / (vel_24h / 24.0 + 1e-6), 0.0, 10.0))

    # ── F7: behavioral drift via autoencoder reconstruction error ─────────
    with torch.no_grad():
        batch       = torch.FloatTensor(features_np).to(DEVICE)
        recon, _    = autoencoder(batch)
        mse         = float(((recon - batch) ** 2).mean().cpu().numpy())
    p5  = DRIFT_NORM.get("p5", 0.0)
    p95 = DRIFT_NORM.get("p95", 1.0)
    f7_drift = float(np.clip((mse - p5) / (p95 - p5 + 1e-9), 0.0, 1.0))

    # ── F8: Phase 1 risk score ────────────────────────────────────────────
    f8_p1 = float(txn_meta.get("p1_risk_score", 0.5))

    # ── F9: amount percentile (relative to account's observed range) ──────
    amt_min = profile.get("amt_min", 0.0)
    amt_max = profile.get("amt_max", amount)
    span    = amt_max - amt_min
    if span > 0:
        f9_amount_percentile = float(np.clip((amount - amt_min) / span, 0.0, 1.0))
    else:
        f9_amount_percentile = 0.5   # no history → neutral

    # ── F10: time gap norm (time since last transaction, normalised) ───────
    last_txn_dt = profile.get("last_txn_dt", 0.0)
    now_ts      = txn_meta.get("timestamp", time.time()) or time.time()
    if last_txn_dt > 0:
        gap_seconds = max(0.0, float(now_ts) - float(last_txn_dt))
        # clip at 30 days; gap > 30d is treated as 1.0 (very long gap)
        f10_time_gap = float(np.clip(gap_seconds / (30 * 86400), 0.0, 1.0))
    else:
        f10_time_gap = 1.0   # no prior transaction on record → maximum gap

    # ── F11: merchant fraud rate (static lookup from training data) ────────
    # product_cd maps to a known fraud-rate bucket (C, H, R, S, W)
    f11_merchant_fraud = float(MERCHANT_FRAUD_RATES.get(product_cd, 0.05))
    # normalise to [0, 1] using the known range [0.020, 0.117]
    f11_merchant_fraud = float(np.clip(
        (f11_merchant_fraud - 0.020) / (0.117 - 0.020 + 1e-9), 0.0, 1.0
    ))

    # ── F12: cross_device_accounts (fraction of shared devices in profile) ─
    # Proxy: count of known devices relative to transaction count.
    # More distinct devices per transaction → higher cross-device risk.
    kd_count  = len(kd)
    txn_count = max(1, profile.get("txn_count", 1))
    f12_cross_device = float(np.clip(kd_count / txn_count, 0.0, 1.0))

    # ── F13: account age score (normalised, 0=new, 1=very old) ────────────
    # We use txn_count as a proxy: saturates at 200 transactions (≈mature).
    f13_account_age = float(np.clip(txn_count / 200.0, 0.0, 1.0))

    # ── Assemble in feature_names order ───────────────────────────────────
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


def decide_action(compromise_prob: float, p1_risk: float) -> str:
    """Map compromise probability + P1 risk to a recommended action."""
    if compromise_prob >= HIGH_THRESHOLD:
        return "FREEZE_AND_NOTIFY" if p1_risk > 0.5 else "STEP_UP_AUTHENTICATION"
    if compromise_prob >= ELEVATED_THRESHOLD:
        return "MONITOR_AND_FLAG"
    return "APPROVE"

@app.get("/panel", response_class=FileResponse)
def serve_panel():
    """Serve the RXT-J+ FraudShield demo panel."""
    panel_path = os.path.join(BASE, "panel.html")
    return FileResponse(panel_path, media_type="text/html")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "RXT-J+ Fraud Risk Scoring API",
        "version": "2.0.0",
        "status":  "live",
        "phase":   "Phase 2 — Behavioral + Contextual Fusion",
        "model_auc": config['final_auc'],
        "model_mcc": config['final_mcc'],
        "fusion_auc": fusion_config['fusion_auc'],
        "latency_ms": config['latency_ms']
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "fusion_input_dim": fusion_config['input_dim'],
        "feature_count": len(FEATURE_NAMES),
    }

@app.get("/model/info")
def model_info():
    return {
        "phase1": {
            "weights":     {"W_MODEL": W_MODEL, "W_IFM": W_IFM},
            "threshold":   THRESHOLD,
            "performance": {
                "auc": config['final_auc'],
                "mcc": config['final_mcc'],
                "false_positives": config['false_positives'],
                "false_negatives": config['false_negatives'],
            },
            "speed": {
                "latency_ms":     config['latency_ms'],
                "throughput_tps": config['throughput_tps']
            }
        },
        "phase2": {
            "model":        "FusionNet v3 (13→128→64→32→1)",
            "input_dim":    fusion_config['input_dim'],
            "feature_names": FEATURE_NAMES,
            "thresholds": {
                "fusion":   FUSION_THRESHOLD,
                "high":     HIGH_THRESHOLD,
                "elevated": ELEVATED_THRESHOLD,
            },
            "performance": {
                "auc":     fusion_config['fusion_auc'],
                "pr_auc":  fusion_config['fusion_pr_auc'],
                "mcc":     fusion_config['fusion_mcc'],
                "recall":  fusion_config['fusion_recall'],
            }
        }
    }

@app.get("/demo-samples")
def demo_samples():
    path = os.path.join(BASE, 'data', 'demo_samples.json')
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="demo_samples.json not found — run save_demo_samples.py first"
        )
    with open(path, 'r') as f:
        return JSONResponse(content=json.load(f))

@app.post("/score", response_model=RiskResponse)
def score_single(req: TransactionRequest):
    t0 = time.time()
    try:
        features_np = np.array(req.features, dtype=np.float32).reshape(1, -1)
        risk, preds, conf = score_raw_features(features_np)
        latency  = (time.time() - t0) * 1000
        decision = "FRAUD" if preds[0] == 1 else "LEGIT"

        _publish_profile_update(req.transaction_id, risk[0], decision, req.features)

        return RiskResponse(
            transaction_id = req.transaction_id,
            risk_score     = round(float(risk[0]), 4),
            decision       = decision,
            confidence     = round(float(conf[0]), 4),
            latency_ms     = round(latency, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/form")
def score_form(req: FormScoreRequest):
    t0 = time.time()
    try:
        partial = [req.amount, 1.0 if req.is_foreign else 0.0, float(req.customer_age or 0)]
        padded  = pad_form_features(partial, RAW_DIM).reshape(1, -1)
        risk, preds, conf = score_raw_features(padded)
        latency  = (time.time() - t0) * 1000
        decision = "FRAUD" if preds[0] == 1 else "LEGIT"

        conn = sqlite3.connect(HISTORY_PATH)
        conn.execute(
            "INSERT INTO history (tid, amt, score, status, ts) VALUES (?, ?, ?, ?, ?)",
            (req.transaction_id, float(req.amount), round(float(risk[0]), 4), decision, time.time())
        )
        conn.commit()
        conn.close()

        _publish_profile_update(req.transaction_id, risk[0], decision, partial)

        return {
            "transaction_id": req.transaction_id,
            "risk_score":     round(float(risk[0]), 4),
            "decision":       decision,
            "confidence":     round(float(conf[0]), 4),
            "latency_ms":     round(latency, 4),
        }
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
def score_direct(req: DirectScoreRequest):
    t0 = time.time()
    try:
        Z_earn = np.array(req.features, dtype=np.float32).reshape(1, -1)
        if Z_earn.shape[1] != ipca.n_components_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {ipca.n_components_} features, got {Z_earn.shape[1]}"
            )

        risk, preds, conf = score_earn_features(Z_earn)
        latency = (time.time() - t0) * 1000

        return {
            "transaction_id": req.transaction_id,
            "risk_score":     round(float(risk[0]), 4),
            "decision":       "FRAUD" if preds[0] == 1 else "LEGIT",
            "confidence":     round(float(conf[0]), 4),
            "latency_ms":     round(latency, 4),
            "model_version":  "rxtj_plus_v1"
        }
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
        ids         = [t.transaction_id for t in req.transactions]
        features_np = np.array([t.features for t in req.transactions], dtype=np.float32)
        risk, preds, conf = score_raw_features(features_np)
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

@app.post("/score/form-full")
def score_form_full(req: FullFormRequest):
    """Phase 1 + Phase 2 scoring from plain human-readable form inputs.

    No pre-computed feature vector required.  The endpoint:
      1. Builds a padded raw (224-dim) feature vector from the form fields
         using pad_form_features() — missing columns filled with imputer means.
      2. Runs the full Phase 1 pipeline (imputer -> scaler -> Nystroem -> IPCA
         -> ResNeXt + Self-Attention GRU + IFM ensemble) to produce p1_risk_score.
      3. Uses transform_raw() output as the autoencoder input for behavioral drift.
      4. Computes all 13 contextual features using the live form fields + the
         account behavioral profile from ProfileStore.
      5. Runs FusionNet v3 to produce compromise probability + explainability.
      6. Persists the event to the transaction log and updates the account profile.

    Result: every change to amount / hour / device / addr produces a different
    Phase 1 score AND different contextual feature values — fully live model
    inference with no pre-saved vectors.
    """
    t0 = time.time()
    try:
        # Step 1: synthesise 224-dim raw feature vector from form fields.
        # col 0 -> TransactionAmt (strongest single predictor)
        # col 1 -> is_foreign flag (card-not-present proxy)
        # col 2 -> customer age
        # Every other column is NaN -> imputer fills with training-set mean.
        partial = [
            float(req.amount),
            1.0 if req.is_foreign else 0.0,
            float(req.customer_age or 30),
        ]
        raw_vec = pad_form_features(partial, RAW_DIM).reshape(1, -1)

        # Step 2: Phase 1 scoring
        # score_raw_features applies imputer -> scaler -> Nystroem -> IPCA -> model
        risk, preds, conf = score_raw_features(raw_vec)
        p1_risk = float(risk[0])
        p1_dec  = "FRAUD" if preds[0] == 1 else "LEGIT"
        p1_conf = float(conf[0])

        # Step 3: X_clean for autoencoder (behavioral drift)
        # transform_raw gives the imputed + scaled 224-dim vector the AE was trained on.
        X_clean = transform_raw(raw_vec, imputer, scaler)

        # Step 4: load account behavioral profile
        profile = profile_store.get_or_empty(str(req.account_id))

        # Step 5: compute all 13 contextual features
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

        if ctx_vec.shape != (1, 13):
            raise HTTPException(
                status_code=500,
                detail=f"Feature shape error: {ctx_vec.shape} != (1, 13)"
            )

        # Step 6: FusionNet v3 inference
        ctx_scaled = fusion_scaler.transform(ctx_vec).astype(np.float32)
        ctx_tensor = torch.FloatTensor(ctx_scaled).to(DEVICE)

        with torch.no_grad():
            compromise_prob_t, attn_weights_t = fusion_model(ctx_tensor)
        compromise_prob = float(compromise_prob_t.cpu().numpy()[0])
        attn_weights_np = attn_weights_t.cpu().numpy()

        # Step 7: decision tiers
        if compromise_prob >= HIGH_THRESHOLD:
            decision = "HIGH"
        elif compromise_prob >= ELEVATED_THRESHOLD:
            decision = "ELEVATED"
        else:
            decision = "LOW"

        rec_action = decide_action(compromise_prob, p1_risk)

        # Step 8: explainability
        explainability = {
            FEATURE_NAMES[i]: round(float(attn_weights_np[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }
        top_trigger = max(explainability, key=explainability.get)

        # Step 9: persist log + update profile
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

        return {
            "account_id":             req.account_id,
            "transaction_id":         req.transaction_id,
            "compromise_probability": round(compromise_prob, 4),
            "decision":               decision,
            "recommended_action":     rec_action,
            "p1_risk_score":          round(p1_risk, 4),
            "p1_decision":            p1_dec,
            "p1_confidence":          round(p1_conf, 4),
            "behavioral_drift_score": ctx_named["behavioral_drift"],
            "contextual_features":    ctx_named,
            "explainability":         explainability,
            "top_trigger_feature":    top_trigger,
            "latency_ms":             latency_ms,
            "model_version":          "rxtj_plus_v2_form_full",
            "feature_source":         "synthesised_from_form_inputs",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/account/compromise-score")
def account_compromise_score(req: CompromiseRequest):
    """Phase 2 — Full behavioral + contextual fusion scoring (13-feature FusionNet v3)."""
    t0 = time.time()
    try:
        # — Phase 1 scoring —
        # Accept either 50-dim EARN+ (post-IPCA) or full raw 224-dim features.
        features_np = np.array(req.features, dtype=np.float32).reshape(1, -1)
        n_feat = features_np.shape[1]

        if n_feat == ipca.n_components_:
            # Already EARN+ space — use direct scorer; synthesize X_clean for AE
            risk, preds, conf = score_earn_features(features_np)
            p1_risk = float(risk[0])
            # Reconstruct approximate 224-dim input for autoencoder via inverse_transform
            Z_nys_approx = ipca.inverse_transform(features_np)        # (1, nystroem_dim)
            # We can't invert Nystroem exactly; use the EARN+ vector padded to AE input
            ae_in_dim = autoencoder.encoder[0].in_features            # 224
            X_clean   = np.zeros((1, ae_in_dim), dtype=np.float32)
            X_clean[0, :min(n_feat, ae_in_dim)] = features_np[0, :min(n_feat, ae_in_dim)]
        else:
            # Full raw features path
            risk, preds, conf = score_raw_features(features_np)
            p1_risk = float(risk[0])
            # — Imputed+scaled for autoencoder —
            X_clean = transform_raw(features_np, imputer, scaler)

        # — Load account profile —
        profile = profile_store.get_or_empty(str(req.account_id))

        # — Contextual features (all 13) —
        txn_meta = {
            "amount":       req.amount   or (req.features[0] if req.features else 0.0),
            "hour":         req.hour     or 0,
            "product_cd":   req.product_cd or "",
            "device_info":  req.device_info,
            "addr1":        req.addr1,
            "p1_risk_score": p1_risk,
            "timestamp":    req.timestamp or time.time(),
        }
        ctx_vec, ctx_named = compute_contextual_features(X_clean, profile, txn_meta)

        # Safety check: shape must be (1, 13)
        assert ctx_vec.shape == (1, 13), f"ctx_vec shape {ctx_vec.shape} != (1,13)"

        # — FusionNet inference —
        ctx_scaled = fusion_scaler.transform(ctx_vec).astype(np.float32)
        ctx_tensor = torch.FloatTensor(ctx_scaled).to(DEVICE)

        with torch.no_grad():
            compromise_prob_t, attn_weights_t = fusion_model(ctx_tensor)
        compromise_prob  = float(compromise_prob_t.cpu().numpy()[0])
        attn_weights_np  = attn_weights_t.cpu().numpy()

        # — Decision —
        if compromise_prob >= HIGH_THRESHOLD:
            decision = "HIGH"
        elif compromise_prob >= ELEVATED_THRESHOLD:
            decision = "ELEVATED"
        else:
            decision = "LOW"

        rec_action = decide_action(compromise_prob, p1_risk)

        # — Explainability dict (attention weight × feature value magnitude) —
        explainability = {
            FEATURE_NAMES[i]: round(float(attn_weights_np[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }
        top_trigger = max(explainability, key=explainability.get)

        # — Persist to log —
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

        # — Update profile —
        profile_store.update_profile(str(req.account_id), {
            "amount":      txn_meta["amount"],
            "hour":        txn_meta["hour"],
            "product_cd":  txn_meta["product_cd"],
            "device_info": req.device_info,
            "addr1":       req.addr1,
            "txn_dt":      req.timestamp or time.time(),
        })

        latency_ms = round((time.time() - t0) * 1000, 3)

        return {
            "account_id":             req.account_id,
            "transaction_id":         req.transaction_id,
            "compromise_probability": round(compromise_prob, 4),
            "decision":               decision,
            "p1_risk_score":          round(p1_risk, 4),
            "behavioral_drift_score": ctx_named["behavioral_drift"],
            "contextual_features":    ctx_named,
            "explainability":         explainability,
            "top_trigger_feature":    top_trigger,
            "recommended_action":     rec_action,
            "latency_ms":             latency_ms,
            "model_version":          "rxtj_plus_v2_phase2",
        }
    except AssertionError as e:
        raise HTTPException(status_code=500, detail=f"Feature shape error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/account/profile/{account_id}")
def get_account_profile(account_id: str, window: str = "30d"):
    """Return stored behavioral profile for an account."""
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
    """Return the last N scored events for an account, newest first."""
    limit = max(1, min(int(limit), 100))
    history = profile_store.get_history(account_id, limit=limit)
    return {
        "account_id":    account_id,
        "record_count":  len(history),
        "events":        history,
    }


@app.get("/account/alerts")
def get_account_alerts(threshold: float = 0.7, limit: int = 50):
    """Return accounts with peak compromise score above threshold in last 24h."""
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
    """Return a human-readable explanation for a stored compromise score."""
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
        """Human-readable interpretation for all 13 feature names."""
        tips = {
            "amount_z_score":        f"Amount deviated {abs(val):.1f}σ from account's average (z={val:.2f}).",
            "merchant_novelty":      f"Merchant type {'never' if val > 0.9 else 'rarely'} seen before (novelty={val:.2f}).",
            "geo_displacement":      f"Location {'far' if val > 0.5 else 'somewhat'} from typical billing region ({val:.2f}).",
            "hour_deviation":        f"Transaction hour {'unusual' if val > 0.6 else 'slightly off'} for this account ({val:.2f}).",
            "device_novelty":        f"{'New device never seen before' if val > 0.9 else 'Unfamiliar device'} ({val:.2f}).",
            "velocity_ratio":        f"Transaction rate {val:.1f}× higher than usual for this time window.",
            "behavioral_drift":      f"Overall behavioral pattern {'significantly' if val > 0.6 else 'somewhat'} different from historical baseline ({val:.2f}).",
            "p1_risk_score":         f"Phase 1 per-transaction risk score: {val:.2f}.",
            "amount_percentile":     f"Amount is in the {'top' if val > 0.8 else 'upper' if val > 0.6 else 'normal'} range for this account (percentile={val:.2f}).",
            "time_gap_norm":         f"Gap since last transaction is {'unusually long' if val > 0.7 else 'moderate'} ({val:.2f}).",
            "merchant_fraud_rate":   f"Merchant category has {'high' if val > 0.6 else 'moderate'} baseline fraud rate ({val:.2f}).",
            "cross_device_accounts": f"{'Many' if val > 0.5 else 'Some'} distinct devices relative to account history ({val:.2f}).",
            "account_age_score":     f"Account is {'mature' if val > 0.7 else 'relatively new'} ({val:.2f}).",
        }
        return tips.get(name, f"{name} = {val:.4f}")

    if ctx:
        sorted_feats = sorted(ctx.items(), key=lambda x: abs(float(x[1])), reverse=True)[:3]
    else:
        sorted_feats = []

    top_features = [
        {"feature": k, "value": round(float(v), 4), "interpretation": interpret(k, float(v))}
        for k, v in sorted_feats
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

    if top_features:
        primary = top_features[0]
        summary = (
            f"The compromise score was primarily driven by: "
            f"{primary['interpretation']} "
            + (f"Additionally, {top_features[1]['interpretation']}" if len(top_features) > 1 else "")
        )
    else:
        summary = "Insufficient context data for detailed explanation."

    return {
        "account_id":       req.account_id,
        "transaction_id":   req.transaction_id,
        "compromise_prob":  event.get("compromise_prob"),
        "decision":         event.get("decision"),
        "top_features":     top_features,
        "account_baseline": baseline,
        "what_triggered":   summary,
    }