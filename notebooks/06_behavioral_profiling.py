# ============================================================
# notebooks/06_behavioral_profiling.py
# Run from F:\rxtj_phase_2\:
#   python notebooks/06_behavioral_profiling.py
#
# PURPOSE: Build per-account behavioral profiles from IEEE-CIS
# training data and generate the tx_snapshot.parquet needed by NB07.
#
# WHAT WAS FIXED (merged from fix_nb06_snapshot.py):
#   - device_novelty: replaced broken expanding().apply() on strings
#     with a fast groupby-first-occurrence approach (3-separator key)
#   - velocity: renamed internal column to _vel to avoid conflicts
#   - Snapshot saved BEFORE device novelty so partial runs don't lose data
#   - Smart resume: if both DB and snapshot already exist, skips the
#     2.5-hour profile loop and rebuilds only what is missing
#
# OUTPUTS:
#   data/behavioral_profiles.db   — SQLite per-account profile store
#   data/tx_snapshot.parquet      — per-row historical snapshot for NB07
#   data/profile_stats.json       — coverage diagnostics
#
# RUNTIME:
#   Fresh run  : ~2.5 h (profile loop) + ~15 min (snapshot)
#   Resume run : ~15 min (snapshot only, DB already built)
# ============================================================

# %% Cell 1: Imports & paths
import os, sys, json, time
import pandas as pd
import numpy as np

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
TX_PATH  = os.path.join(DATA_DIR, "IEEE CIS", "train_transaction.csv")
ID_PATH  = os.path.join(DATA_DIR, "IEEE CIS", "train_identity.csv")
DB_PATH  = os.path.join(DATA_DIR, "behavioral_profiles.db")
SNAP_PATH= os.path.join(DATA_DIR, "tx_snapshot.parquet")

sys.path.insert(0, ROOT)
from profile_store import ProfileStore

print(f"[NB06] Root     : {ROOT}")
print(f"[NB06] DB path  : {DB_PATH}")
print(f"[NB06] Snap path: {SNAP_PATH}")

# %% Cell 2: Load CSV data
print("\n[NB06] Loading transaction CSV...")
t0 = time.time()

TX_COLS = ["TransactionID", "isFraud", "TransactionDT",
           "TransactionAmt", "ProductCD", "card1", "addr1"]

tx = pd.read_csv(TX_PATH, usecols=lambda c: c in TX_COLS)
print(f"  {len(tx):,} rows loaded in {time.time()-t0:.1f}s")

try:
    id_df = pd.read_csv(ID_PATH, usecols=["TransactionID", "DeviceInfo"])
    tx    = tx.merge(id_df, on="TransactionID", how="left")
    print(f"  DeviceInfo merged. Non-null: {tx['DeviceInfo'].notna().sum():,}")
except FileNotFoundError:
    tx["DeviceInfo"] = None
    print("  train_identity.csv not found — DeviceInfo set to null")

# %% Cell 3: Derive hour, sort by account + time
tx["hour"]       = (tx["TransactionDT"] // 3600).astype(int) % 24
tx["device_str"] = tx["DeviceInfo"].fillna("").astype(str)

print("\n[NB06] Sorting by card1, TransactionDT...")
tx = tx.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

# %% Cell 4: Profile building (smart resume)
ps      = ProfileStore(db_path=DB_PATH)
stats   = ps.coverage_stats()
total   = len(tx)
accts   = tx["card1"].nunique()

_db_ready = stats["total_accounts"] >= int(0.9 * accts)  # ≥90% accounts profiled

if _db_ready:
    print(f"\n[NB06] Profile DB already built ({stats['total_accounts']:,} accounts). Skipping loop.")
else:
    print(f"\n[NB06] Building behavioral profiles ({accts:,} accounts, {total:,} transactions)...")
    print(f"  This takes ~2.5 hours. Progress logged every 5,000 rows.")
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
        if (i + 1) % 5000 == 0:
            pct     = 100 * (i + 1) / total
            elapsed = time.time() - t_start
            eta     = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1:>7,}/{total:,}  ({pct:.1f}%)  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    print(f"\n[NB06] Profile loop done in {time.time()-t_start:.0f}s")
    stats = ps.coverage_stats()

print(f"\n[NB06] Coverage: {stats}")

# %% Cell 5: Rolling amount stats (vectorised)
print("\n[NB06] Computing rolling amount stats...")
tx["amt_cumsum"] = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().mean().shift(1).fillna(s)
)
tx["amt_cumstd"] = tx.groupby("card1")["TransactionAmt"].transform(
    lambda s: s.expanding().std().shift(1).fillna(0)
)
tx["txn_seq"] = tx.groupby("card1").cumcount()
print("  Done.")

# %% Cell 6: Velocity features (1h and 24h windows)
# Uses a per-account look-back window of up to 200 rows for speed.
print("\n[NB06] Computing velocity features (~5 min)...")

def _rolling_velocity(group, window_sec):
    dts = group["TransactionDT"].values
    vel = np.zeros(len(dts), dtype=np.int32)
    for j in range(len(dts)):
        start  = max(0, j - 200)
        vel[j] = int(np.sum(dts[start:j] >= dts[j] - window_sec))
    out        = group.copy()
    out["_vel"]= vel
    return out

t_vel     = time.time()
tx_sorted = tx.sort_values("TransactionDT")

vel_1h = (
    tx_sorted.groupby("card1", group_keys=False)
    .apply(lambda g: _rolling_velocity(g, 3600))
    [["TransactionID", "_vel"]]
    .rename(columns={"_vel": "velocity_1h"})
)
print(f"  1h velocity done  ({time.time()-t_vel:.0f}s)")

vel_24h = (
    tx_sorted.groupby("card1", group_keys=False)
    .apply(lambda g: _rolling_velocity(g, 86400))
    [["TransactionID", "_vel"]]
    .rename(columns={"_vel": "velocity_24h"})
)
print(f"  24h velocity done ({time.time()-t_vel:.0f}s)")

tx = tx.merge(vel_1h,  on="TransactionID", how="left")
tx = tx.merge(vel_24h, on="TransactionID", how="left")
tx["velocity_1h"]  = tx["velocity_1h"].fillna(0).astype(int)
tx["velocity_24h"] = tx["velocity_24h"].fillna(0).astype(int)

# %% Cell 7: Device novelty (FIXED — no string expanding().apply())
# Marks the FIRST occurrence of each (card1, device) pair as novel=1.0.
# Subsequent occurrences of the same device → 0.0 (seen before).
# Empty / unknown device → 0.5 (neutral).
print("\n[NB06] Computing device novelty...")

tx["device_key"] = tx["card1"].astype(str) + "|||" + tx["device_str"]

first_seen = (
    tx.groupby("device_key")["TransactionDT"]
    .min()
    .rename("first_dt")
    .reset_index()
)
tx = tx.merge(first_seen, on="device_key", how="left")

tx["device_novel"] = np.where(
    tx["device_str"] == "",                          # no device info → neutral
    0.5,
    np.where(
        tx["TransactionDT"] == tx["first_dt"],        # first time seen → novel
        1.0,
        0.0                                           # seen before → not novel
    )
)

print(f"  device_novel: "
      f"novel=1.0: {(tx['device_novel']==1.0).sum():,}  "
      f"seen=0.0: {(tx['device_novel']==0.0).sum():,}  "
      f"neutral=0.5: {(tx['device_novel']==0.5).sum():,}")

# %% Cell 8: Save snapshot
SNAP_COLS = [
    "TransactionID", "card1", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "addr1", "DeviceInfo", "hour",
    "amt_cumsum", "amt_cumstd", "txn_seq",
    "velocity_1h", "velocity_24h", "device_novel",
]
snap = tx[[c for c in SNAP_COLS if c in tx.columns]].copy()
snap.to_parquet(SNAP_PATH, index=False)
print(f"\n[NB06] Snapshot saved → {SNAP_PATH}  ({len(snap):,} rows)")
print(snap[["TransactionAmt","amt_cumsum","velocity_1h","device_novel"]].describe().round(3))

# %% Cell 9: Save diagnostics JSON
diag = {
    "total_transactions": int(len(tx)),
    "unique_accounts":    int(accts),
    "fraud_count":        int(tx["isFraud"].sum()),
    "fraud_rate_pct":     round(100 * tx["isFraud"].mean(), 3),
    "profile_coverage":   stats,
    "snapshot_path":      SNAP_PATH,
    "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(os.path.join(DATA_DIR, "profile_stats.json"), "w") as f:
    json.dump(diag, f, indent=2)

print(f"\n[NB06] ✓ Done.")
print(f"  Accounts profiled : {stats['total_accounts']:,}")
print(f"  Snapshot rows     : {len(snap):,}")
print(f"  Fraud rate        : {diag['fraud_rate_pct']}%")
print(f"\n[NB06] NEXT STEP → python notebooks/07_contextual_features.py")