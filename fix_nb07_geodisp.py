# ============================================================
# fix_nb07_geodisp.py  —  run from F:\rxtj_phase_2\
#   python fix_nb07_geodisp.py
#
# WHAT IT FIXES:
#   1. geo_displacement column (index 2) is all NaN in
#      contextual_features.npy — addr1 is float, not string,
#      so the original function failed silently.
#   2. Rewrites contextual_features.npy with correct values.
#   3. Prints the correlation table that NB07 crashed on.
#
# RUNTIME: ~1 min
# ============================================================

import os, sys, json
import numpy as np
import pandas as pd

ROOT     = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")

print("[PATCH] Loading saved arrays...")
X = np.load(os.path.join(DATA_DIR, "contextual_features.npy"))
y = np.load(os.path.join(DATA_DIR, "fusion_labels.npy"))

with open(os.path.join(DATA_DIR, "fusion_feature_names.json")) as f:
    NAMES = json.load(f)

print(f"  X shape: {X.shape}  —  geo_displacement NaN count before fix: {np.isnan(X[:,2]).sum():,}")

# ── Recompute geo_displacement correctly ──────────────────────────────────────
print("\n[PATCH] Loading snapshot to recompute geo_displacement...")
snap = pd.read_parquet(os.path.join(DATA_DIR, "tx_snapshot.parquet"))

# addr1 is a FLOAT column (numeric region code, nullable).
# Convert to float, fill NaN with 0.
addr1_float = pd.to_numeric(snap["addr1"], errors="coerce").fillna(0.0).values.astype(np.float32)

print("[PATCH] Computing per-account geo cluster (rolling mode via median)...")

# Sort by card1 + TransactionDT (already sorted in snapshot but re-confirm)
snap = snap.copy()
snap["addr1_f"] = addr1_float

# Per-account expanding median of addr1 up to (but not including) this transaction.
# Shift(1) gives the median of all PRIOR transactions = the account's "home" cluster.
snap["addr1_baseline"] = (
    snap.groupby("card1")["addr1_f"]
    .transform(lambda s: s.expanding().median().shift(1))
    .fillna(snap["addr1_f"])   # first transaction: no prior → use own value → disp=0
)

# Displacement = abs(current - baseline) / 500, clipped to [0, 1].
raw_disp = np.abs(snap["addr1_f"].values - snap["addr1_baseline"].values) / 500.0
geo_disp = np.clip(raw_disp, 0.0, 1.0).astype(np.float32)

print(f"  geo_displacement — min={geo_disp.min():.4f}  "
      f"mean={geo_disp.mean():.4f}  max={geo_disp.max():.4f}  "
      f"NaN count: {np.isnan(geo_disp).sum()}")

# Patch column 2 in X
X[:, 2] = geo_disp

# Final NaN check across all columns
nan_counts = np.isnan(X).sum(axis=0)
print("\n[PATCH] NaN counts per feature after fix:")
for i, name in enumerate(NAMES):
    flag = "  ← STILL HAS NaN — CHECK" if nan_counts[i] > 0 else ""
    print(f"  {name:<25}  NaN={nan_counts[i]}{flag}")

# Replace any residual NaN with 0 (safety net)
if np.isnan(X).any():
    print("\n[PATCH] Replacing residual NaN values with 0...")
    X = np.nan_to_num(X, nan=0.0)

# ── Save patched array ────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "contextual_features.npy")
np.save(out_path, X)
print(f"\n[PATCH] Saved patched contextual_features.npy → {out_path}")

# ── Print correlation table (what NB07 crashed on) ───────────────────────────
print("\n[PATCH] Feature–label correlations (positive = fraud-predictive):")
for i, name in enumerate(NAMES):
    col  = X[:, i]
    corr = float(np.corrcoef(col, y)[0, 1]) if np.std(col) > 0 else 0.0
    bar  = "█" * int(abs(corr) * 40)
    sign = "+" if corr >= 0 else "-"
    print(f"  {name:<25} {sign}{abs(corr):.4f}  {bar}")

# ── Final shape check ─────────────────────────────────────────────────────────
print(f"\n[PATCH] Final feature matrix stats:")
print(f"  Shape          : {X.shape}")
print(f"  Total NaN      : {np.isnan(X).sum()}")
print(f"  Positive labels: {int(y.sum()):,}  ({100*y.mean():.2f}%)")
print(f"\n[PATCH] ✓ Done. NEXT STEP → python notebooks/08_fusion_model_training.py")