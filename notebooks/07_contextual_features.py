# ============================================================
# notebooks/07_contextual_features.py
# Run after 06_behavioral_profiling.py:
#   python notebooks/07_contextual_features.py
#
# PURPOSE: Compute 8 contextual deviation features per transaction
# by comparing each transaction against the account's cumulative
# history AT THAT POINT IN TIME. Also generates account-compromise
# labels for fusion model training.
#
# WHAT WAS FIXED (merged from fix_nb07_geodisp.py):
#   - geo_displacement: was using string comparison on float addr1 column
#     → replaced with float expanding-median baseline, abs diff / 500
#   - NaN guard: np.nan_to_num applied before saving any arrays
#   - Correlation table: wrapped with std check to handle NaN/zero-std cols
#   - Autoencoder: strict=False on load_state_dict (tolerates minor
#     architecture differences without crashing)
#
# OUTPUTS:
#   data/contextual_features.npy     — shape (N, 8), float32
#   data/fusion_labels.npy           — isFraud labels, float32
#   data/fusion_account_ids.npy      — card1 per row, int64
#   data/p1_scores_for_fusion.npy    — P1 risk scores (flat 0.5 if unavailable)
#   data/fusion_feature_names.json   — ordered feature name list
#   data/drift_norm_params.json      — {p5, p95} for autoencoder normalisation
#
# NOTE: p1_risk_score (feature 8) will be flat 0.5 until NB08 patches it.
# NB08 generates real scores and writes them back before training.
#
# RUNTIME: ~10–15 min
# ============================================================

# %% Cell 1: Imports
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
SNAP_PATH = os.path.join(DATA_DIR, "tx_snapshot.parquet")

sys.path.insert(0, ROOT)
print(f"[NB07] Root: {ROOT}")

# %% Cell 2: Load snapshot
print("\n[NB07] Loading snapshot...")
t0 = time.time()
tx = pd.read_parquet(SNAP_PATH)
print(f"  {len(tx):,} rows, {len(tx.columns)} columns — {time.time()-t0:.1f}s")
print(f"  Columns: {list(tx.columns)}")

# %% Cell 3: Load Phase 1 preprocessors
print("\n[NB07] Loading Phase 1 artifacts...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
RAW_DIM = int(imputer.n_features_in_)
print(f"  imputer : {RAW_DIM} features")
print(f"  scaler  : {scaler.n_features_in_} features")

# ── Autoencoder (strict=False tolerates minor arch differences) ───────────────
# Operates on the 224-dim imputed+scaled feature space (NOT the 128-dim EARN+
# features). Used purely for per-transaction reconstruction error = drift score.
class _Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
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
ae_in_dim   = ae_state["encoder.0.weight"].shape[1]
autoencoder = _Autoencoder(ae_in_dim)
autoencoder.load_state_dict(ae_state, strict=False)
autoencoder.eval()
print(f"  autoencoder : input_dim={ae_in_dim} (strict=False)")

# %% Cell 4: Feature 7 — behavioral drift (autoencoder reconstruction error)
# Build a (N, 224) float32 matrix with the 4 available columns filled in;
# remaining 220 positions are NaN → imputed to training-set means by imputer.
print("\n[NB07] Computing behavioral drift scores...")
pcd_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
X_raw   = np.full((len(tx), RAW_DIM), np.nan, dtype=np.float32)
X_raw[:, 0] = tx["TransactionAmt"].fillna(0).values.astype(np.float32)
X_raw[:, 1] = tx["hour"].values.astype(np.float32)
X_raw[:, 2] = tx["ProductCD"].map(pcd_map).fillna(-1).values.astype(np.float32)
X_raw[:, 3] = pd.to_numeric(tx["addr1"], errors="coerce").values.astype(np.float32)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_imp    = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp).astype(np.float32)
print(f"  Imputed+scaled: {X_scaled.shape}")

BATCH = 2048
recon_errors = []
with torch.no_grad():
    for s in range(0, len(X_scaled), BATCH):
        batch     = torch.FloatTensor(X_scaled[s:s+BATCH])
        recon, _  = autoencoder(batch)
        mse       = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
        recon_errors.extend(mse.tolist())
        if (s // BATCH) % 50 == 0:
            print(f"    {min(s+BATCH, len(X_scaled)):>7,}/{len(X_scaled):,}")

recon_errors = np.array(recon_errors, dtype=np.float32)
p5, p95      = np.percentile(recon_errors, 5), np.percentile(recon_errors, 95)
drift_scores = np.clip((recon_errors - p5) / (p95 - p5 + 1e-9), 0.0, 1.0)
print(f"  Recon error — min={recon_errors.min():.4f}  "
      f"mean={recon_errors.mean():.4f}  max={recon_errors.max():.4f}")
print(f"  Drift normalised — p5={p5:.4f}  p95={p95:.4f}")

with open(os.path.join(DATA_DIR, "drift_norm_params.json"), "w") as f:
    json.dump({"p5": float(p5), "p95": float(p95)}, f)
print(f"  drift_norm_params.json saved")

# %% Cell 5: Feature 1 — amount_z_score
print("\n[NB07] Building contextual features...")
amt      = tx["TransactionAmt"].fillna(0).values.astype(np.float32)
cum_mean = tx["amt_cumsum"].fillna(0).values.astype(np.float32)
cum_std  = np.where(
    tx["amt_cumstd"].fillna(0).values.astype(np.float32) < 0.1,
    1.0,
    tx["amt_cumstd"].values.astype(np.float32)
)
f1_amount_z = np.clip((amt - cum_mean) / cum_std, -5.0, 5.0).astype(np.float32)
print(f"  F1 amount_z_score   — mean={f1_amount_z.mean():.3f}  std={f1_amount_z.std():.3f}")

# %% Cell 6: Feature 2 — merchant_novelty
tx["pcd_int"] = tx["ProductCD"].map({"W":0,"H":1,"C":2,"S":3,"R":4}).fillna(-1).astype(int)

def _pcd_novelty(group):
    pcd, novel, counts = group["pcd_int"].values, np.ones(len(group), np.float32), {}
    for j, p in enumerate(pcd):
        if j > 0 and p in counts:
            novel[j] = 1.0 - (counts[p] / j)
        counts[p] = counts.get(p, 0) + 1
    return pd.Series(novel, index=group.index)

print("  Computing merchant novelty...")
f2_merchant_novelty = (
    tx.groupby("card1", group_keys=False)
    .apply(_pcd_novelty)
    .values.astype(np.float32)
)
print(f"  F2 merchant_novelty — mean={f2_merchant_novelty.mean():.3f}")

# %% Cell 7: Feature 3 — geo_displacement (FIXED)
# addr1 is a FLOAT numeric region code. We compute the expanding median of
# prior transactions as the account's "home cluster" baseline, then take
# abs(current - baseline) / 500 clipped to [0, 1].
print("  Computing geo displacement (fixed: float addr1)...")

tx_snap          = tx.copy()
tx_snap["addr1_f"] = pd.to_numeric(tx_snap["addr1"], errors="coerce").fillna(0.0)
tx_snap["addr1_baseline"] = (
    tx_snap.groupby("card1")["addr1_f"]
    .transform(lambda s: s.expanding().median().shift(1))
    .fillna(tx_snap["addr1_f"])   # first transaction: baseline = own value → disp = 0
)
raw_disp = np.abs(
    tx_snap["addr1_f"].values - tx_snap["addr1_baseline"].values
) / 500.0
f3_geo_disp = np.clip(raw_disp, 0.0, 1.0).astype(np.float32)
nan_count   = np.isnan(f3_geo_disp).sum()
if nan_count:
    print(f"  WARNING: {nan_count} NaN in geo_displacement — replacing with 0")
    f3_geo_disp = np.nan_to_num(f3_geo_disp, nan=0.0)
print(f"  F3 geo_displacement — mean={f3_geo_disp.mean():.3f}  NaN={nan_count}")

# %% Cell 8: Feature 4 — hour_deviation
def _hour_deviation(group):
    hours, dev, counts = group["hour"].values, np.zeros(len(group), np.float32), [0]*24
    for j, h in enumerate(hours):
        if j > 0:
            dev[j] = 1.0 - (counts[int(h)] / j)
        counts[int(h)] += 1
    return pd.Series(dev, index=group.index)

print("  Computing hour deviation...")
f4_hour_dev = (
    tx.groupby("card1", group_keys=False)
    .apply(_hour_deviation)
    .values.astype(np.float32)
)
print(f"  F4 hour_deviation   — mean={f4_hour_dev.mean():.3f}")

# %% Cell 9: Feature 5 — device_novelty (from NB06 snapshot)
f5_device_novel = tx["device_novel"].fillna(0.5).values.astype(np.float32)
print(f"  F5 device_novelty   — mean={f5_device_novel.mean():.3f}")

# %% Cell 10: Feature 6 — velocity_ratio
vel_1h   = tx["velocity_1h"].fillna(0).values.astype(np.float32)
vel_24h  = tx["velocity_24h"].fillna(0).values.astype(np.float32)
f6_vel_ratio = np.clip(vel_1h / (vel_24h / 24.0 + 1e-6), 0.0, 10.0).astype(np.float32)
print(f"  F6 velocity_ratio   — mean={f6_vel_ratio.mean():.3f}")

# %% Cell 11: Feature 7 — behavioral_drift (computed above)
f7_drift = drift_scores.astype(np.float32)
print(f"  F7 behavioral_drift — mean={f7_drift.mean():.3f}")

# %% Cell 12: Feature 8 — p1_risk_score (placeholder; NB08 patches with real scores)
p1_npy = os.path.join(DATA_DIR, "model_probs_full.npy")
if os.path.exists(p1_npy):
    f8_p1 = np.load(p1_npy).astype(np.float32).ravel()
    if len(f8_p1) != len(tx):
        print(f"  model_probs_full.npy size mismatch ({len(f8_p1)} ≠ {len(tx)}) — using 0.5")
        f8_p1 = np.full(len(tx), 0.5, np.float32)
    else:
        print(f"  F8 p1_risk_score — loaded from model_probs_full.npy  mean={f8_p1.mean():.3f}")
else:
    # Fall back to model_probs.npy (from NB05, may be subset)
    p1_fallback = os.path.join(DATA_DIR, "model_probs.npy")
    if os.path.exists(p1_fallback):
        tmp = np.load(p1_fallback).astype(np.float32).ravel()
        if len(tmp) == len(tx):
            f8_p1 = tmp
            print(f"  F8 p1_risk_score — loaded from model_probs.npy  mean={f8_p1.mean():.3f}")
        else:
            f8_p1 = np.full(len(tx), 0.5, np.float32)
            print(f"  F8 p1_risk_score — using neutral 0.5 (NB08 will patch with real scores)")
    else:
        f8_p1 = np.full(len(tx), 0.5, np.float32)
        print(f"  F8 p1_risk_score — using neutral 0.5 (NB08 will patch with real scores)")

# %% Cell 13: Stack all 8 features
FEATURE_NAMES = [
    "amount_z_score",    # 0
    "merchant_novelty",  # 1
    "geo_displacement",  # 2
    "hour_deviation",    # 3
    "device_novelty",    # 4
    "velocity_ratio",    # 5
    "behavioral_drift",  # 6
    "p1_risk_score",     # 7
]

X_fusion = np.column_stack([
    f1_amount_z, f2_merchant_novelty, f3_geo_disp, f4_hour_dev,
    f5_device_novel, f6_vel_ratio, f7_drift, f8_p1,
]).astype(np.float32)

print(f"\n[NB07] Fusion feature matrix: {X_fusion.shape}")
print("  Feature stats (min / mean / max / NaN):")
for i, name in enumerate(FEATURE_NAMES):
    col = X_fusion[:, i]
    nan_c = int(np.isnan(col).sum())
    print(f"    {name:<25}  {col.min():.3f} / {col.mean():.3f} / {col.max():.3f} / NaN={nan_c}")

# NaN safety guard — replace any residual NaN with 0 before saving
if np.isnan(X_fusion).any():
    total_nan = int(np.isnan(X_fusion).sum())
    print(f"\n  WARNING: Replacing {total_nan} residual NaN values with 0")
    X_fusion = np.nan_to_num(X_fusion, nan=0.0)

# %% Cell 14: Account-compromise labels
print("\n[NB07] Building account-compromise labels...")
fraud_accounts = set(tx.loc[tx["isFraud"] == 1, "card1"].astype(str).unique())
print(f"  Accounts with ≥1 fraud: {len(fraud_accounts):,}")

y_fusion    = tx["isFraud"].fillna(0).values.astype(np.float32)
account_ids = tx["card1"].fillna(-1).values.astype(np.int64)

print(f"  Positive: {int(y_fusion.sum()):,} / {len(y_fusion):,}  "
      f"({100*y_fusion.mean():.2f}%)")

# %% Cell 15: Save all outputs
print("\n[NB07] Saving outputs...")
np.save(os.path.join(DATA_DIR, "contextual_features.npy"),  X_fusion)
np.save(os.path.join(DATA_DIR, "fusion_labels.npy"),         y_fusion)
np.save(os.path.join(DATA_DIR, "fusion_account_ids.npy"),    account_ids)
np.save(os.path.join(DATA_DIR, "p1_scores_for_fusion.npy"),  f8_p1)
with open(os.path.join(DATA_DIR, "fusion_feature_names.json"), "w") as f:
    json.dump(FEATURE_NAMES, f, indent=2)

print(f"  contextual_features.npy  → {X_fusion.shape}")
print(f"  fusion_labels.npy        → {y_fusion.shape}")
print(f"  fusion_account_ids.npy   → {account_ids.shape}")
print(f"  fusion_feature_names.json saved")

# %% Cell 16: Feature–label correlations (NaN-safe)
print("\n[NB07] Feature–label correlations (positive = fraud-predictive):")
for i, name in enumerate(FEATURE_NAMES):
    col = X_fusion[:, i]
    if np.std(col) > 0 and not np.isnan(col).any():
        corr = float(np.corrcoef(col, y_fusion)[0, 1])
        bar  = "█" * int(abs(corr) * 40)
        sign = "+" if corr >= 0 else "-"
    else:
        corr, bar, sign = 0.0, "", "?"
    print(f"  {name:<25} {sign}{abs(corr):.4f}  {bar}")

print(f"\n[NB07] ✓ Done.")
print(f"  Total NaN in saved matrix: {int(np.isnan(X_fusion).sum())}")
print(f"\n[NB07] NEXT STEP → python notebooks/08_fusion_model_training.py")