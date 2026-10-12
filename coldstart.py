import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
df = pd.read_parquet("/home/vishu/Projects/rxtj_phase_2/data/tx_snapshot.parquet").sort_values("TransactionDT").reset_index(drop=True)
n=len(df); tr=df.iloc[:int(n*.70)]; te=df.iloc[int(n*.85):]
seen = te.card1.isin(set(tr.card1.unique()))
print("test rows whose card1 was SEEN in train: %.2f%%" % (100*seen.mean()))
print("  seen  : n=%6d fraud=%.2f%%" % (seen.sum(), 100*te.isFraud[seen].mean()))
print("  unseen: n=%6d fraud=%.2f%%" % ((~seen).sum(), 100*te.isFraud[~seen].mean()))
m = lgb.Booster(model_file='/tmp/claude-1000/-home-vishu-Projects-rxtj-phase-2/a7ed3607-22cf-4810-8778-6b58fa94efe4/scratchpad/lgbm_baseline.txt')
CATS=["card1","addr1","ProductCD","DeviceInfo"]
NUMS=["TransactionAmt","hour","amt_cumsum","amt_cumstd","txn_seq","velocity_1h","velocity_24h","device_novel"]
tr=tr.copy(); te=te.copy()
for c in CATS:
    f=tr[c].astype(str).value_counts(normalize=True)
    for p in (tr,te): p[c+"_freq"]=p[c].astype(str).map(f).fillna(0.0).astype(float)
for p in (tr,te):
    p["amt_vs_hist"]=p.TransactionAmt/(p.amt_cumsum+1e-6)
    p["amt_z_hist"]=(p.TransactionAmt-p.amt_cumsum)/(p.amt_cumstd+1.0)
FE=NUMS+[c+"_freq" for c in CATS]+["amt_vs_hist","amt_z_hist"]+CATS
for c in CATS:
    tr[c]=tr[c].astype(str).astype("category")
    te[c]=pd.Categorical(te[c].astype(str), categories=tr[c].cat.categories)
p=m.predict(te[FE])
y=te.isFraud.values
print("\nOVERALL      AUC=%.4f  PR-AUC=%.4f" % (roc_auc_score(y,p), average_precision_score(y,p)))
s=seen.values
print("SEEN cards   AUC=%.4f  PR-AUC=%.4f  (n=%d)" % (roc_auc_score(y[s],p[s]), average_precision_score(y[s],p[s]), s.sum()))
if (~s).sum()>50 and y[~s].sum()>0:
    print("UNSEEN cards AUC=%.4f  PR-AUC=%.4f  (n=%d, frauds=%d)" % (roc_auc_score(y[~s],p[~s]), average_precision_score(y[~s],p[~s]), (~s).sum(), y[~s].sum()))
else:
    print("UNSEEN cards: too few rows/frauds to score (n=%d, frauds=%d)" % ((~s).sum(), y[~s].sum()))
