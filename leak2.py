import pandas as pd, numpy as np

df = pd.read_parquet("/home/vishu/Projects/rxtj_phase_2/data/tx_snapshot.parquet")
d = df.sort_values(["card1", "TransactionDT"]).copy()
g = d.groupby("card1", sort=False)

d["exp_mean"] = g["TransactionAmt"].expanding().mean().reset_index(level=0, drop=True)
d["exp_std"] = g["TransactionAmt"].expanding().std().reset_index(level=0, drop=True).fillna(0)

print("amt_cumsum == expanding MEAN?  frac=%.4f"
      % np.isclose(d.amt_cumsum, d.exp_mean, rtol=1e-3, atol=1e-2).mean())
print("amt_cumstd == expanding STD?   frac=%.4f"
      % np.isclose(d.amt_cumstd, d.exp_std, rtol=1e-3, atol=1e-2).mean())

# first txn: does amt_cumsum == TransactionAmt ?
first = g.head(1)
print("first txn: amt_cumsum == TransactionAmt? frac=%.4f"
      % np.isclose(first.amt_cumsum, first.TransactionAmt, rtol=1e-3, atol=1e-2).mean())

print("\nsample card1 trace (first card with >=6 txns):")
counts = d.card1.value_counts()
c = counts[counts >= 6].index[0]
cols = ["TransactionDT", "TransactionAmt", "amt_cumsum", "amt_cumstd",
        "txn_seq", "velocity_1h", "velocity_24h", "device_novel", "isFraud"]
print(d[d.card1 == c][cols].head(8).to_string(index=False))

print("\nDeviceInfo null pct: %.1f%%" % (100 * df.DeviceInfo.isnull().mean()))
print("addr1      null pct: %.1f%%" % (100 * df.addr1.isnull().mean()))
