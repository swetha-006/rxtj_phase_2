import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tx_snapshot.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm.pkl")

df = pd.read_parquet(DATA_PATH).sort_values("TransactionDT").reset_index(drop=True)
n = len(df)
tr = df.iloc[:int(n * 0.70)]
te = df.iloc[int(n * 0.85):].copy()

seen = te.card1.isin(set(tr.card1.unique()))
print("test rows whose card1 was SEEN in train: %.2f%%" % (100 * seen.mean()))
print("  seen  : n=%6d fraud=%.2f%%" % (seen.sum(), 100 * te.isFraud[seen].mean()))
print("  unseen: n=%6d fraud=%.2f%%" % ((~seen).sum(), 100 * te.isFraud[~seen].mean()))

bundle = joblib.load(MODEL_PATH)
m = bundle["model"]
FE = bundle["features"]
top_devices = bundle["top_devices"]

te["amt_to_mean"] = te["TransactionAmt"] / (te["amt_cumsum"] + 1.0)
te["amt_diff_mean"] = te["TransactionAmt"] - te["amt_cumsum"]
te["velocity_ratio"] = te["velocity_1h"] / (te["velocity_24h"] / 24.0 + 1e-4)

te["ProductCD"] = te["ProductCD"].fillna("MISSING").astype(str).astype("category")
te["DeviceInfo"] = te["DeviceInfo"].apply(
    lambda x: x if x in top_devices else ("MISSING" if pd.isna(x) else "OTHER")
).astype("category")

p = m.predict_proba(te[FE])[:, 1]
y = te.isFraud.values
print("\nOVERALL      AUC=%.4f  PR-AUC=%.4f" % (roc_auc_score(y, p), average_precision_score(y, p)))
s = seen.values
print("SEEN cards   AUC=%.4f  PR-AUC=%.4f  (n=%d)" % (roc_auc_score(y[s], p[s]), average_precision_score(y[s], p[s]), s.sum()))
if (~s).sum() > 50 and y[~s].sum() > 0:
    print("UNSEEN cards AUC=%.4f  PR-AUC=%.4f  (n=%d, frauds=%d)" % (roc_auc_score(y[~s], p[~s]), average_precision_score(y[~s], p[~s]), (~s).sum(), y[~s].sum()))
else:
    print("UNSEEN cards: too few rows/frauds to score (n=%d, frauds=%d)" % ((~s).sum(), y[~s].sum()))

