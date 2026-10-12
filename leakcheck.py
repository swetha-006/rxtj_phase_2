import os, pandas as pd, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(BASE_DIR, "data", "tx_snapshot.parquet")
df = pd.read_parquet(P)
print("shape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nisFraud counts:\n", df.isFraud.value_counts())
print("fraud rate: %.4f%%" % (100 * df.isFraud.mean()))
print("\nnulls:\n", df.isnull().sum())
print("\nTransactionDT range:", df.TransactionDT.min(), "->", df.TransactionDT.max())
print("unique card1:", df.card1.nunique())
print("unique addr1:", df.addr1.nunique())
print("unique DeviceInfo:", df.DeviceInfo.nunique())
print("unique ProductCD:", df.ProductCD.unique())

print("\n" + "=" * 60)
print("LEAK CHECK: are derived cols computed causally in DT order?")
print("=" * 60)

d = df.sort_values(["card1", "TransactionDT"]).copy()
g = d.groupby("card1", sort=False)

# txn_seq should increment by 1 within card1
d["seq_diff"] = g["txn_seq"].diff()
sd = d["seq_diff"].dropna()
print("\ntxn_seq diff within card1: min=%s max=%s  ==1 frac=%.4f"
      % (sd.min(), sd.max(), (sd == 1).mean()))

# amt_cumsum should be non-decreasing within card1
d["cs_diff"] = g["amt_cumsum"].diff()
cd = d["cs_diff"].dropna()
print("amt_cumsum diff within card1: min=%.4f  neg frac=%.6f"
      % (cd.min(), (cd < 0).mean()))

# does amt_cumsum equal running sum of TransactionAmt (inclusive or exclusive)?
d["run_incl"] = g["TransactionAmt"].cumsum()
d["run_excl"] = d["run_incl"] - d["TransactionAmt"]
print("amt_cumsum == inclusive running sum? frac=%.4f"
      % np.isclose(d.amt_cumsum, d.run_incl, rtol=1e-4, atol=1e-3).mean())
print("amt_cumsum == EXCLUSIVE running sum? frac=%.4f"
      % np.isclose(d.amt_cumsum, d.run_excl, rtol=1e-4, atol=1e-3).mean())

# first txn of each card: what is cumsum / seq?
first = d.groupby("card1", sort=False).head(1)
print("\nfirst-txn-per-card: txn_seq values ->", first.txn_seq.value_counts().head().to_dict())
print("first-txn-per-card: amt_cumsum==0 frac=%.4f" % (first.amt_cumsum == 0).mean())
print("first-txn-per-card: amt_cumstd==0 frac=%.4f" % (first.amt_cumstd == 0).mean())
print("first-txn-per-card: device_novel value counts ->",
      first.device_novel.value_counts().head().to_dict())

# device_novel / velocity sanity
print("\ndevice_novel uniques:", np.sort(df.device_novel.unique())[:10])
print("velocity_1h  describe:\n", df.velocity_1h.describe())
print("velocity_24h describe:\n", df.velocity_24h.describe())

# correlation of each numeric col with label (full data)
print("\n" + "=" * 60)
print("Pearson corr with isFraud (full data)")
print("=" * 60)
num = df.select_dtypes(include=[np.number]).drop(columns=["isFraud"])
for c in num.columns:
    print("  %-16s %+.4f" % (c, np.corrcoef(num[c].fillna(-999), df.isFraud)[0, 1]))
