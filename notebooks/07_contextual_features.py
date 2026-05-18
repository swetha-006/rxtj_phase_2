# ============================================================
# notebooks/07_contextual_features.py
# Run after 06_behavioral_profiling.py
#
# PURPOSE: Compute 8 contextual deviation features per transaction
# by comparing each transaction against the account's cumulative
# history AT THAT POINT IN TIME. Also generates account-compromise
# labels and Phase 1 risk scores for fusion model training.
#
# RUNTIME: ~5–10 min (vectorised pandas).
# INPUTS:
#   data/tx_snapshot.parquet          — from NB06
#   models/imputer.pkl, scaler.pkl    — Phase 1 preprocessors
#   models/autoencoder.pt             — for behavioral drift scores
# OUTPUTS:
#   data/contextual_features.npy      — shape (N, 8)
#   data/fusion_labels.npy            — account compromise labels
#   data/fusion_feature_names.json    — feature name list
#   data/p1_scores_for_fusion.npy     — Phase 1 risk scores (N,)
# ============================================================

# %% Cell 1: Imports
import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
MODEL_DIR  = os.path.join(ROOT, "models")
SNAP_PATH  = os.path.join(DATA_DIR, "tx_snapshot.parquet")

print(f"[NB07] Root: {ROOT}")

# %% Cell 2: Load snapshot from NB06
print("\n[NB07] Loading snapshot …")
t0  = time.time()
tx  = pd.read_parquet(SNAP_PATH)
print(f"  {len(tx):,} rows, {len(tx.columns)} columns — {time.time()-t0:.1f}s")
print(f"  Columns: {list(tx.columns)}")

# %% Cell 3: Load Phase 1 preprocessors + autoencoder
print("\n[NB07] Loading Phase 1 artifacts …")
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"  imputer: {imputer.n_features_in_} features")
print(f"  scaler:  {scaler.n_features_in_} features")

# ── AutoencoderNet (must match app.py definition exactly) ────────────────────
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

ae_state    = torch.load(os.path.join(MODEL_DIR, "autoencoder.pt"), map_location="cpu")
ae_in_dim   = ae_state["encoder.0.weight"].shape[1]  # 224
autoencoder = AutoencoderNet(ae_in_dim)
autoencoder.load_state_dict(ae_state, strict=False)   # strict=False tolerates extra ReLU
autoencoder.eval()
print(f"  autoencoder: input_dim={ae_in_dim}")

# %% Cell 4: Feature 7 — Behavioral drift (autoencoder reconstruction error)
# The autoencoder operates on 224-dim imputed+scaled features.
# We reconstruct each transaction's feature vector and compute MSE error.
# Normalise to [0,1] using 5th–95th percentile of the training distribution.

print("\n[NB07] Computing behavioral drift scores (autoencoder recon error) …")

# Build a 224-dim raw feature matrix by padding TransactionAmt + basic features.
# Exact same padding logic as pad_form_features in preprocessing.py.
RAW_DIM = int(imputer.n_features_in_)

def make_raw_matrix(df, raw_dim):
    """Construct a (N, raw_dim) float32 matrix from available columns."""
    amt  = df["TransactionAmt"].fillna(0).values.astype(np.float32)
    hour = df["hour"].values.astype(np.float32)
    pcd_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
    pcd_enc = df["ProductCD"].map(pcd_map).fillna(-1).values.astype(np.float32)
    addr1   = pd.to_numeric(df["addr1"], errors="coerce").fillna(np.nan).values.astype(np.float32)

    X = np.full((len(df), raw_dim), np.nan, dtype=np.float32)
    X[:, 0] = amt
    X[:, 1] = hour
    X[:, 2] = pcd_enc
    X[:, 3] = addr1
    return X

X_raw = make_raw_matrix(tx, RAW_DIM)
print(f"  Raw matrix shape: {X_raw.shape}")

# Apply Phase 1 imputer + scaler (same chain as inference).
X_imp    = imputer.transform(X_raw)
X_scaled = scaler.transform(X_imp).astype(np.float32)
print(f"  Imputed+scaled: {X_scaled.shape}")

# Run autoencoder in batches.
BATCH = 2048
recon_errors = []
with torch.no_grad():
    for start in range(0, len(X_scaled), BATCH):
        batch = torch.FloatTensor(X_scaled[start:start+BATCH])
        recon, _ = autoencoder(batch)
        mse = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
        recon_errors.extend(mse.tolist())
        if (start // BATCH) % 50 == 0:
            print(f"    {start+BATCH:>7,}/{len(X_scaled):,}")

recon_errors = np.array(recon_errors, dtype=np.float32)
print(f"  Recon error — min={recon_errors.min():.4f}  "
      f"mean={recon_errors.mean():.4f}  max={recon_errors.max():.4f}")

# Normalise using 5th–95th percentile (clamp outliers).
p5, p95 = np.percentile(recon_errors, 5), np.percentile(recon_errors, 95)
drift_scores = np.clip((recon_errors - p5) / (p95 - p5 + 1e-9), 0.0, 1.0)
print(f"  Drift scores normalised — p5={p5:.4f}  p95={p95:.4f}")

# Save normalisation params for use at inference time.
drift_norm = {"p5": float(p5), "p95": float(p95)}
with open(os.path.join(DATA_DIR, "drift_norm_params.json"), "w") as f:
    json.dump(drift_norm, f)
print(f"  Drift norm params saved → data/drift_norm_params.json")

# %% Cell 5: Feature 1 — amount_z_score
print("\n[NB07] Building contextual features …")

amt      = tx["TransactionAmt"].fillna(0).values.astype(np.float32)
cum_mean = tx["amt_cumsum"].fillna(0).values.astype(np.float32)
cum_std  = tx["amt_cumstd"].fillna(1).values.astype(np.float32)
cum_std  = np.where(cum_std < 0.1, 1.0, cum_std)   # avoid div-by-zero on new accounts

f1_amount_z = np.clip((amt - cum_mean) / cum_std, -5.0, 5.0).astype(np.float32)
print(f"  F1 amount_z_score  — mean={f1_amount_z.mean():.3f}  std={f1_amount_z.std():.3f}")

# %% Cell 6: Feature 2 — merchant_novelty
# For each transaction, compute the proportion of times this ProductCD was
# NOT seen in earlier transactions for the same card1 (0 = common, 1 = never seen).

pcd_map_num = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
tx["pcd_int"] = tx["ProductCD"].map(pcd_map_num).fillna(-1).astype(int)

# Cumulative fraction of same-pcd transactions seen BEFORE this one.
def pcd_novelty(group):
    pcd   = group["pcd_int"].values
    novel = np.ones(len(pcd), dtype=np.float32)
    counts = {}
    for j, p in enumerate(pcd):
        total = j  # number of prior txns
        if total > 0 and p in counts:
            novel[j] = 1.0 - (counts[p] / total)
        counts[p] = counts.get(p, 0) + 1
    return pd.Series(novel, index=group.index)

print("  Computing merchant novelty (grouped apply) …")
f2_merchant_novelty = tx.groupby("card1", group_keys=False).apply(pcd_novelty).values.astype(np.float32)
print(f"  F2 merchant_novelty — mean={f2_merchant_novelty.mean():.3f}")

# %% Cell 7: Feature 3 — geo_displacement
# addr1 is a numeric billing region code.
# Displacement = normalised deviation from account's running mode.

addr1_num = pd.to_numeric(tx["addr1"], errors="coerce").fillna(0).values.astype(np.float32)

def geo_displacement(group):
    addrs  = group["addr1"].values
    disp   = np.zeros(len(addrs), dtype=np.float32)
    seen   = {}
    for j, a in enumerate(addrs):
        if j == 0: seen[a] = 1; continue
        mode  = max(seen, key=seen.get)
        raw   = abs(float(a) - float(mode)) if (a and mode) else 0.0
        disp[j] = min(raw / 500.0, 1.0)   # 500 normalisation constant
        seen[a]  = seen.get(a, 0) + 1
    return pd.Series(disp, index=group.index)

tx["addr1_str"] = tx["addr1"].fillna("0")
print("  Computing geo displacement …")
f3_geo_disp = tx.groupby("card1", group_keys=False).apply(geo_displacement).values.astype(np.float32)
print(f"  F3 geo_displacement — mean={f3_geo_disp.mean():.3f}")

# %% Cell 8: Feature 4 — hour_deviation
# 1 - (frequency of this hour in account's prior history).
# First transaction for an account → 0 (neutral).

def hour_deviation(group):
    hours  = group["hour"].values
    dev    = np.zeros(len(hours), dtype=np.float32)
    counts = [0] * 24
    for j, h in enumerate(hours):
        total = j
        if total > 0:
            dev[j] = 1.0 - (counts[int(h)] / total)
        counts[int(h)] += 1
    return pd.Series(dev, index=group.index)

print("  Computing hour deviation …")
f4_hour_dev = tx.groupby("card1", group_keys=False).apply(hour_deviation).values.astype(np.float32)
print(f"  F4 hour_deviation   — mean={f4_hour_dev.mean():.3f}")

# %% Cell 9: Feature 5 — device_novelty (already computed in NB06)
f5_device_novel = tx["device_novel"].fillna(0).values.astype(np.float32)
print(f"  F5 device_novelty   — mean={f5_device_novel.mean():.3f} (fraction novel)")

# %% Cell 10: Feature 6 — velocity_ratio
vel_1h   = tx["velocity_1h"].fillna(0).values.astype(np.float32)
vel_24h  = tx["velocity_24h"].fillna(0).values.astype(np.float32)
baseline = vel_24h / 24.0 + 1e-6
f6_vel_ratio = np.clip(vel_1h / baseline, 0.0, 10.0).astype(np.float32)
print(f"  F6 velocity_ratio   — mean={f6_vel_ratio.mean():.3f}")

# %% Cell 11: Feature 7 — behavioral_drift (already computed above)
f7_drift = drift_scores.astype(np.float32)

# %% Cell 12: Feature 8 — try to load Phase 1 risk scores, else compute
p1_npy = os.path.join(DATA_DIR, "model_probs.npy")
if os.path.exists(p1_npy):
    f8_p1_risk = np.load(p1_npy).astype(np.float32).ravel()
    if len(f8_p1_risk) != len(tx):
        print(f"  WARNING: model_probs.npy size ({len(f8_p1_risk)}) != tx size ({len(tx)})")
        print("  Falling back to neutral 0.5 scores — re-run after Phase 1 NB05.")
        f8_p1_risk = np.full(len(tx), 0.5, dtype=np.float32)
    else:
        print(f"  F8 P1 risk — loaded from model_probs.npy. mean={f8_p1_risk.mean():.3f}")
else:
    print(f"  model_probs.npy not found — using neutral 0.5 for F8.")
    print(f"  (Run NB05 first and re-run this notebook for best results.)")
    f8_p1_risk = np.full(len(tx), 0.5, dtype=np.float32)

# %% Cell 13: Stack all 8 features
FEATURE_NAMES = [
    "amount_z_score",
    "merchant_novelty",
    "geo_displacement",
    "hour_deviation",
    "device_novelty",
    "velocity_ratio",
    "behavioral_drift",
    "p1_risk_score",
]

X_fusion = np.column_stack([
    f1_amount_z,       # 0
    f2_merchant_novelty,  # 1
    f3_geo_disp,       # 2
    f4_hour_dev,       # 3
    f5_device_novel,   # 4
    f6_vel_ratio,      # 5
    f7_drift,          # 6
    f8_p1_risk,        # 7
]).astype(np.float32)

print(f"\n[NB07] Fusion feature matrix: {X_fusion.shape}")
print("  Feature stats:")
for i, name in enumerate(FEATURE_NAMES):
    col = X_fusion[:, i]
    print(f"    {name:<25}  min={col.min():.3f}  mean={col.mean():.3f}  max={col.max():.3f}")

# %% Cell 14: Build account-compromise labels
# Strategy: an account is "compromised" if it has ≥1 confirmed fraud
# transaction. We label ALL transactions for such accounts during and
# after the first fraud event as potential compromise events.
# This gives a more nuanced signal than per-transaction isFraud.

print("\n[NB07] Building account-compromise labels …")

fraud_accounts = set(
    tx.loc[tx["isFraud"] == 1, "card1"].astype(str).unique()
)
print(f"  Accounts with ≥1 fraud: {len(fraud_accounts):,}")

# Per-account: find first fraud TransactionDT.
first_fraud = (
    tx[tx["isFraud"] == 1]
    .groupby("card1")["TransactionDT"]
    .min()
    .rename("first_fraud_dt")
)
tx = tx.merge(first_fraud.reset_index(), on="card1", how="left")

# An account is "compromised" from its first fraud transaction onward.
# isFraud=1 transactions are positive regardless (they're actual fraud).
# Non-fraud transactions AFTER first fraud on a compromised account get
# label = 0.5 (uncertain) — we binarise at 0.5 → 0 for training.
# Simpler: positive label = isFraud=1 on a fraud account.
# This keeps the task tractable without noisy pseudo-labels.
y_fusion = tx["isFraud"].fillna(0).values.astype(np.float32)

print(f"  Label distribution: {int(y_fusion.sum()):,} positive "
      f"/ {int((y_fusion==0).sum()):,} negative "
      f"({100*y_fusion.mean():.2f}% positive)")

# Also build account_id array for stratified splitting in NB08.
account_ids = tx["card1"].fillna(-1).values.astype(np.int64)

# %% Cell 15: Save outputs
print("\n[NB07] Saving outputs …")

np.save(os.path.join(DATA_DIR, "contextual_features.npy"),   X_fusion)
np.save(os.path.join(DATA_DIR, "fusion_labels.npy"),          y_fusion)
np.save(os.path.join(DATA_DIR, "fusion_account_ids.npy"),     account_ids)
np.save(os.path.join(DATA_DIR, "p1_scores_for_fusion.npy"),   f8_p1_risk)

with open(os.path.join(DATA_DIR, "fusion_feature_names.json"), "w") as f:
    json.dump(FEATURE_NAMES, f, indent=2)

print(f"  contextual_features.npy  → {X_fusion.shape}")
print(f"  fusion_labels.npy        → {y_fusion.shape}")
print(f"  fusion_account_ids.npy   → {account_ids.shape}")
print(f"  fusion_feature_names.json → {FEATURE_NAMES}")

# %% Cell 16: Sanity check — feature correlation with label
print("\n[NB07] Feature–label correlations (positive = fraud-predictive):")
for i, name in enumerate(FEATURE_NAMES):
    corr = np.corrcoef(X_fusion[:, i], y_fusion)[0, 1]
    bar  = "█" * int(abs(corr) * 40)
    sign = "+" if corr >= 0 else "-"
    print(f"  {name:<25} {sign}{abs(corr):.4f}  {bar}")

print(f"\n[NB07] ✓ Done. NEXT STEP → run notebooks/08_fusion_model_training.py")