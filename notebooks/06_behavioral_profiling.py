# ============================================================
# notebooks/06_behavioral_profiling.py
# Run as a Jupyter notebook OR as a plain script:
#   python notebooks/06_behavioral_profiling.py
#
# PURPOSE: Build per-account behavioral profiles from the IEEE-CIS
# training data. Profiles are stored in data/behavioral_profiles.db
# (SQLite) and used by Phase 2 contextual feature computation.
#
# RUNTIME: ~3–8 min on 590K rows (pure pandas, no GPU needed).
# OUTPUTS:
#   data/behavioral_profiles.db  — SQLite profile store
#   data/profile_stats.json      — coverage diagnostics
# ============================================================

# %% [markdown]
# ## 06 — Behavioral Profiling
# Groups IEEE-CIS transactions by `card1` (account proxy) and computes
# rolling behavioral statistics used by Phase 2 contextual scoring.

# %% Cell 1: Imports & paths
import os, sys, json, time
import pandas as pd
import numpy as np

# Ensure project root is on path so profile_store imports correctly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from profile_store import ProfileStore

DATA_DIR  = os.path.join(ROOT, "data")
TX_PATH   = os.path.join(DATA_DIR, "IEEE CIS", "train_transaction.csv")
ID_PATH   = os.path.join(DATA_DIR, "IEEE CIS", "train_identity.csv")
DB_PATH   = os.path.join(DATA_DIR, "behavioral_profiles.db")

print(f"[NB06] Root : {ROOT}")
print(f"[NB06] TX   : {TX_PATH}")
print(f"[NB06] DB   : {DB_PATH}")

# %% Cell 2: Load data
print("\n[NB06] Loading transaction CSV …")
t0 = time.time()

# Load only the columns we actually need to keep memory low.
USE_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "addr1", "DeviceInfo",
]

# DeviceInfo lives in identity file — we'll merge later.
TX_COLS = [c for c in USE_COLS if c != "DeviceInfo"]

tx = pd.read_csv(TX_PATH, usecols=lambda c: c in TX_COLS)
print(f"  Transactions loaded: {len(tx):,} rows in {time.time()-t0:.1f}s")

# Try to merge DeviceInfo from identity file.
try:
    id_df = pd.read_csv(ID_PATH, usecols=["TransactionID", "DeviceInfo"])
    tx    = tx.merge(id_df, on="TransactionID", how="left")
    print(f"  DeviceInfo merged. Non-null: {tx['DeviceInfo'].notna().sum():,}")
except FileNotFoundError:
    tx["DeviceInfo"] = None
    print("  train_identity.csv not found — DeviceInfo set to null.")

print(f"  Columns: {list(tx.columns)}")
print(tx.head(3))

# %% Cell 3: Derive hour from TransactionDT
# TransactionDT is seconds elapsed from a reference point (not epoch).
# Hour-of-day: (seconds // 3600) % 24 gives a synthetic hour signal.
tx["hour"] = (tx["TransactionDT"] // 3600).astype(int) % 24

# %% Cell 4: Sort by account + time (critical for temporal lookups in NB07)
print("\n[NB06] Sorting by card1, TransactionDT …")
tx = tx.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

# %% Cell 5: Build profiles using ProfileStore
# We iterate account by account and call update_profile() for every
# transaction IN TIME ORDER. This guarantees the stored profile at the
# end of this pass reflects cumulative history.
#
# NOTE: For NB07 temporal lookups, we will use pre-computed snapshot
# arrays rather than the live DB (too slow at inference for 590K rows).
# This pass builds the FINAL profile for each account which is used
# at API inference time by the live endpoints.

print("\n[NB06] Building profiles …")
ps = ProfileStore(db_path=DB_PATH)

total    = len(tx)
accounts = tx["card1"].nunique()
print(f"  Total transactions: {total:,}")
print(f"  Unique accounts (card1): {accounts:,}")

batch_size = 5000
t_start = time.time()

for i, row in tx.iterrows():
    acct = str(int(row["card1"])) if pd.notna(row["card1"]) else "unknown"
    txn  = {
        "amount":      float(row["TransactionAmt"]) if pd.notna(row["TransactionAmt"]) else 0.0,
        "hour":        int(row["hour"]),
        "product_cd":  str(row["ProductCD"])  if pd.notna(row["ProductCD"])  else "",
        "device_info": str(row["DeviceInfo"]) if pd.notna(row["DeviceInfo"]) else None,
        "addr1":       str(int(row["addr1"])) if pd.notna(row["addr1"])      else "",
        "txn_dt":      float(row["TransactionDT"]),
    }
    ps.update_profile(acct, txn)

    if (i + 1) % batch_size == 0:
        pct = 100 * (i+1) / total
        elapsed = time.time() - t_start
        eta = elapsed / (i+1) * (total - i - 1)
        print(f"  {i+1:>7,}/{total:,}  ({pct:.1f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

print(f"\n[NB06] Profile building complete in {time.time()-t_start:.1f}s")

# %% Cell 6: Validate coverage
print("\n[NB06] Coverage statistics:")
stats = ps.coverage_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# Also build a fast numpy snapshot for NB07.
# For each transaction index, we store the account's CUMULATIVE statistics
# AT THAT POINT. This is expensive but only needs to run once.
# We use a vectorised group-by approach instead of row-by-row DB reads.

print("\n[NB06] Building snapshot arrays for NB07 …")

# Per-account rolling stats (vectorised with pandas groupby + expanding).
tx["amt_cumsum"]  = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().mean().shift(1).fillna(s)
)
tx["amt_cumstd"]  = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().std().shift(1).fillna(0)
)
tx["txn_seq"]     = tx.groupby("card1").cumcount()  # 0-indexed position

# Velocity: transactions by same card1 within 1h and 24h windows.
# Approximate using rank within 3600s / 86400s window.
print("  Computing velocity features …")
tx_sorted = tx.sort_values("TransactionDT")

def rolling_velocity(group, window_sec):
    dts = group["TransactionDT"].values
    vel = np.zeros(len(dts), dtype=np.int32)
    for j in range(len(dts)):
        vel[j] = int(np.sum(dts[max(0,j-200):j] >= dts[j] - window_sec))
    group = group.copy()
    group["vel"] = vel
    return group

print("  (velocity computation may take 3-5 min on 590K rows)")
vel_1h  = tx_sorted.groupby("card1", group_keys=False).apply(
    lambda g: rolling_velocity(g, 3600)
)[["TransactionID", "vel"]].rename(columns={"vel": "velocity_1h"})
vel_24h = tx_sorted.groupby("card1", group_keys=False).apply(
    lambda g: rolling_velocity(g, 86400)
)[["TransactionID", "vel"]].rename(columns={"vel": "velocity_24h"})

tx = tx.merge(vel_1h,  on="TransactionID", how="left")
tx = tx.merge(vel_24h, on="TransactionID", how="left")
tx["velocity_1h"]  = tx["velocity_1h"].fillna(0).astype(int)
tx["velocity_24h"] = tx["velocity_24h"].fillna(0).astype(int)

# %% Cell 7: Build device novelty flag
print("\n[NB06] Building device novelty flags …")
# For each transaction, was the device seen in any EARLIER transaction
# for the same account?
tx["device_str"] = tx["DeviceInfo"].fillna("").astype(str)
tx["device_seen_before"] = (
    tx.groupby("card1")["device_str"]
    .transform(lambda s: s.shift(1).expanding().apply(
        lambda x, cur=None: 1.0 if s.name in x.values else 0.0,
        raw=False
    ))
).fillna(0.0)

# Simpler and faster: mark first occurrence per (card1, device) pair.
tx["device_key"] = tx["card1"].astype(str) + "_" + tx["device_str"]
first_seen = tx.groupby("device_key")["TransactionDT"].min().rename("first_dt")
tx = tx.merge(first_seen.reset_index(), on="device_key", how="left")
tx["device_novel"] = (tx["TransactionDT"] == tx["first_dt"]).astype(float)

# %% Cell 8: Save snapshot for NB07
SNAP_COLS = [
    "TransactionID", "card1", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "addr1", "DeviceInfo", "hour",
    "amt_cumsum", "amt_cumstd", "txn_seq",
    "velocity_1h", "velocity_24h", "device_novel",
]

snap = tx[[c for c in SNAP_COLS if c in tx.columns]].copy()
snap_path = os.path.join(DATA_DIR, "tx_snapshot.parquet")
snap.to_parquet(snap_path, index=False)
print(f"\n[NB06] Snapshot saved → {snap_path}  ({len(snap):,} rows)")
print(snap.describe())

# %% Cell 9: Save diagnostics JSON
diag = {
    "total_transactions": int(len(tx)),
    "unique_accounts":    int(accounts),
    "fraud_count":        int(tx["isFraud"].sum()),
    "fraud_rate_pct":     round(100 * tx["isFraud"].mean(), 3),
    "profile_coverage":   stats,
    "snapshot_path":      snap_path,
    "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(os.path.join(DATA_DIR, "profile_stats.json"), "w") as f:
    json.dump(diag, f, indent=2)

print("\n[NB06] ✓ Done. Summary:")
for k, v in diag.items():
    print(f"  {k}: {v}")
print(f"\n[NB06] NEXT STEP → run notebooks/07_contextual_features.py")