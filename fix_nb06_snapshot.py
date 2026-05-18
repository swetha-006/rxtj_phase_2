# ============================================================
# fix_nb06_snapshot.py
# Run from F:\rxtj_phase_2\
#   python fix_nb06_snapshot.py
#
# PURPOSE: NB06 crashed during device_novelty computation before
# saving tx_snapshot.parquet. This script skips the 2.5-hour
# profile loop (already done) and only rebuilds the snapshot.
#
# RUNTIME: ~5-8 min
# REQUIRES: data/behavioral_profiles.db already exists (it does)
# PRODUCES: data/tx_snapshot.parquet  data/profile_stats.json
# ============================================================

import os, sys, json, time
import pandas as pd
import numpy as np

ROOT     = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TX_PATH  = os.path.join(DATA_DIR, "IEEE CIS", "train_transaction.csv")
ID_PATH  = os.path.join(DATA_DIR, "IEEE CIS", "train_identity.csv")

print("[FIX] Starting snapshot rebuild (profile DB already saved — skipping loop)")
print(f"[FIX] Loading CSV...")
t0 = time.time()

TX_COLS = ["TransactionID", "isFraud", "TransactionDT",
           "TransactionAmt", "ProductCD", "card1", "addr1"]

tx = pd.read_csv(TX_PATH, usecols=lambda c: c in TX_COLS)
print(f"  {len(tx):,} rows loaded in {time.time()-t0:.1f}s")

# Merge DeviceInfo from identity file
try:
    id_df = pd.read_csv(ID_PATH, usecols=["TransactionID", "DeviceInfo"])
    tx    = tx.merge(id_df, on="TransactionID", how="left")
    print(f"  DeviceInfo merged. Non-null: {tx['DeviceInfo'].notna().sum():,}")
except FileNotFoundError:
    tx["DeviceInfo"] = None
    print("  train_identity.csv not found — DeviceInfo set to null")

# ── Derived columns ───────────────────────────────────────────────────────────

tx["hour"]       = (tx["TransactionDT"] // 3600).astype(int) % 24
tx["device_str"] = tx["DeviceInfo"].fillna("").astype(str)

# Sort by account + time (required for temporal correctness in NB07)
print("[FIX] Sorting by card1, TransactionDT...")
tx = tx.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

# ── Rolling amount stats (vectorised) ────────────────────────────────────────
print("[FIX] Computing rolling amount stats...")
tx["amt_cumsum"] = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().mean().shift(1).fillna(s)
)
tx["amt_cumstd"] = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().std().shift(1).fillna(0)
)
tx["txn_seq"] = tx.groupby("card1").cumcount()
print(f"  Done.")

# ── Velocity features ─────────────────────────────────────────────────────────
print("[FIX] Computing velocity features (this takes 3-5 min)...")

def rolling_velocity(group, window_sec):
    """Count transactions within window_sec before each transaction."""
    dts = group["TransactionDT"].values
    vel = np.zeros(len(dts), dtype=np.int32)
    for j in range(len(dts)):
        # Only look back 200 rows max for speed
        start = max(0, j - 200)
        vel[j] = int(np.sum(dts[start:j] >= dts[j] - window_sec))
    out = group.copy()
    out["_vel"] = vel
    return out

t_vel = time.time()
tx_sorted = tx.sort_values("TransactionDT")

vel_1h = (
    tx_sorted.groupby("card1", group_keys=False)
    .apply(lambda g: rolling_velocity(g, 3600))
    [["TransactionID", "_vel"]]
    .rename(columns={"_vel": "velocity_1h"})
)
print(f"  1h done ({time.time()-t_vel:.0f}s)")

vel_24h = (
    tx_sorted.groupby("card1", group_keys=False)
    .apply(lambda g: rolling_velocity(g, 86400))
    [["TransactionID", "_vel"]]
    .rename(columns={"_vel": "velocity_24h"})
)
print(f"  24h done ({time.time()-t_vel:.0f}s)")

tx = tx.merge(vel_1h,  on="TransactionID", how="left")
tx = tx.merge(vel_24h, on="TransactionID", how="left")
tx["velocity_1h"]  = tx["velocity_1h"].fillna(0).astype(int)
tx["velocity_24h"] = tx["velocity_24h"].fillna(0).astype(int)

# ── Device novelty (FIXED — no string expanding().apply()) ────────────────────
# Mark first occurrence of each (card1, device) pair as novel (1.0).
# All subsequent occurrences = seen before (0.0).
# Empty device string = neutral (0.5).
print("[FIX] Computing device novelty (fixed approach)...")

tx["device_key"] = tx["card1"].astype(str) + "|||" + tx["device_str"]

first_seen = (
    tx.groupby("device_key")["TransactionDT"]
    .min()
    .rename("first_dt")
    .reset_index()
)
tx = tx.merge(first_seen, on="device_key", how="left")

# Novel = this transaction IS the first occurrence of this (card1, device) pair
# Empty device → neutral 0.5
tx["device_novel"] = np.where(
    tx["device_str"] == "",        # no device info
    0.5,
    np.where(
        tx["TransactionDT"] == tx["first_dt"],  # first time this device seen
        1.0,
        0.0                                      # device was seen before
    )
)

novel_pct = tx["device_novel"].mean()
print(f"  device_novel mean={novel_pct:.3f}  "
      f"(novel=1.0: {(tx['device_novel']==1.0).sum():,}  "
      f"seen=0.0: {(tx['device_novel']==0.0).sum():,}  "
      f"unknown=0.5: {(tx['device_novel']==0.5).sum():,})")

# ── Save snapshot ─────────────────────────────────────────────────────────────
SNAP_COLS = [
    "TransactionID", "card1", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "addr1", "DeviceInfo", "hour",
    "amt_cumsum", "amt_cumstd", "txn_seq",
    "velocity_1h", "velocity_24h", "device_novel",
]

snap      = tx[[c for c in SNAP_COLS if c in tx.columns]].copy()
snap_path = os.path.join(DATA_DIR, "tx_snapshot.parquet")
snap.to_parquet(snap_path, index=False)
print(f"\n[FIX] Snapshot saved → {snap_path}")
print(f"  Shape: {snap.shape}")
print(snap[["TransactionAmt","amt_cumsum","velocity_1h","device_novel"]].describe().round(3))

# ── Load profile stats from existing DB ──────────────────────────────────────
try:
    from profile_store import ProfileStore
    ps    = ProfileStore(db_path=os.path.join(DATA_DIR, "behavioral_profiles.db"))
    stats = ps.coverage_stats()
except Exception as e:
    stats = {"note": f"ProfileStore import error: {e}"}

diag = {
    "total_transactions": int(len(tx)),
    "unique_accounts":    int(tx["card1"].nunique()),
    "fraud_count":        int(tx["isFraud"].sum()),
    "fraud_rate_pct":     round(100 * tx["isFraud"].mean(), 3),
    "profile_coverage":   stats,
    "snapshot_path":      snap_path,
    "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(os.path.join(DATA_DIR, "profile_stats.json"), "w") as f:
    json.dump(diag, f, indent=2)

print(f"\n[FIX] profile_stats.json saved")
print(f"\n[FIX] ✓ Complete in {time.time()-t0:.0f}s")
print(f"[FIX] NEXT STEP → python notebooks/07_contextual_features.py")