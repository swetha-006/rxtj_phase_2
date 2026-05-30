# ============================================================
# fix_retrain_nb08.py  —  run from F:\rxtj_phase_2\
#   python fix_retrain_nb08.py
#
# Uses the EXACT model architecture copied from app.py so the
# state_dict loads without any key mismatches.
# RUNTIME: ~35-45 min total (Step 1: ~20 min, Step 2: ~15 min)
# ============================================================

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              precision_score, recall_score, f1_score,
                              roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = os.path.abspath(os.path.dirname(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
MODEL_DIR   = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
DEVICE      = torch.device("cpu")

print("=" * 60)
print("  PHASE 2 FIX — Real P1 Scores + Retrain FusionNet v2")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# EXACT model classes copied from app.py
# (ResNeXtBlock uses paths + shortcut + cardinality BN structure)
# ════════════════════════════════════════════════════════════

SEQ_LEN     = 8
CARDINALITY = 4

class ResNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cardinality=CARDINALITY):
        super().__init__()
        group_dim = out_dim // cardinality
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, group_dim), nn.BatchNorm1d(group_dim), nn.ReLU(),
                nn.Linear(group_dim, group_dim), nn.BatchNorm1d(group_dim)
            ) for _ in range(cardinality)
        ])
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.cat([p(x) for p in self.paths], dim=-1) + self.shortcut(x))


class ResNeXtExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResNeXtBlock(input_dim, 128), ResNeXtBlock(128, 128),
            ResNeXtBlock(128, 64),        ResNeXtBlock(64,  64)
        )
    def forward(self, x):
        return self.net(x)


class SelfAttentionGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len  = seq_len
        self.step_dim = input_dim // seq_len
        self.gru      = nn.GRU(self.step_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.3)
        self.attention  = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(),
                                        nn.Linear(32, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(32, 1),
                                        nn.Identity())

    def forward(self, x):
        x        = x.view(x.size(0), self.seq_len, self.step_dim)
        out, _   = self.gru(x)
        attn_w   = torch.softmax(self.attention(out), dim=1)
        context  = (attn_w * out).sum(dim=1)
        return torch.sigmoid(self.classifier(context)).squeeze(1), attn_w


class AttentionRXTJ(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN):
        super().__init__()
        self.resnext  = ResNeXtExtractor(input_dim)
        self.attn_gru = SelfAttentionGRU(input_dim=64, seq_len=seq_len)

    def forward(self, x):
        return self.attn_gru(self.resnext(x))


# ════════════════════════════════════════════════════════════
# STEP 1 — Load Phase 1 artifacts and generate real P1 scores
# ════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading Phase 1 preprocessors...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imputer  = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    nystroem = joblib.load(os.path.join(MODEL_DIR, "nystroem.pkl"))
    ipca     = joblib.load(os.path.join(MODEL_DIR, "incremental_pca.pkl"))
    ifm      = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

RAW_DIM  = int(imputer.n_features_in_)
IPCA_DIM = int(ipca.n_components_)
print(f"  imputer  : {RAW_DIM} features")
print(f"  ipca     : {IPCA_DIM} components")

# Load Phase 1 model with CORRECT architecture
print("  Loading attention_rxtj.pt with exact architecture from app.py...")
p1_model = AttentionRXTJ(input_dim=IPCA_DIM, seq_len=SEQ_LEN).to(DEVICE)
state    = torch.load(os.path.join(MODEL_DIR, "attention_rxtj.pt"),
                      map_location=DEVICE)
p1_model.load_state_dict(state)
p1_model.eval()
print(f"  ✓ attention_rxtj.pt loaded successfully (IPCA_DIM={IPCA_DIM})")

cfg_p1    = json.load(open(os.path.join(RESULTS_DIR, "deployment_config.json")))
W_MODEL   = float(cfg_p1["W_MODEL"])
W_IFM     = float(cfg_p1["W_IFM"])
THRESHOLD = float(cfg_p1["THRESHOLD"])
print(f"  W_MODEL={W_MODEL}  W_IFM={W_IFM}  THRESHOLD={THRESHOLD}")

# ── Build feature matrix from snapshot ───────────────────────────────────────
print("\n[STEP 1] Loading snapshot...")
snap = pd.read_parquet(os.path.join(DATA_DIR, "tx_snapshot.parquet"))
print(f"  {len(snap):,} rows")

pcd_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
X_raw   = np.full((len(snap), RAW_DIM), np.nan, dtype=np.float32)
X_raw[:, 0] = snap["TransactionAmt"].fillna(0).values.astype(np.float32)
X_raw[:, 1] = snap["hour"].values.astype(np.float32)
X_raw[:, 2] = snap["ProductCD"].map(pcd_map).fillna(-1).values.astype(np.float32)
X_raw[:, 3] = pd.to_numeric(snap["addr1"], errors="coerce").values.astype(np.float32)

print("[STEP 1] Applying imputer + scaler...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_imp = imputer.transform(X_raw)
    X_sc  = scaler.transform(X_imp).astype(np.float32)
print(f"  Scaled: {X_sc.shape}")

# Nystroem + IPCA in chunks (~3 min)
print("[STEP 1] Applying Nystroem + IPCA in chunks (~3 min)...")
CHUNK  = 8000
X_ipca = np.zeros((len(X_sc), IPCA_DIM), dtype=np.float32)
t0     = time.time()
for s in range(0, len(X_sc), CHUNK):
    e = min(s + CHUNK, len(X_sc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_ipca[s:e] = ipca.transform(
            nystroem.transform(X_sc[s:e])
        ).astype(np.float32)
    if s % 80000 == 0:
        print(f"  {e:>7,}/{len(X_sc):,}  ({time.time()-t0:.0f}s)")
print(f"  IPCA done in {time.time()-t0:.0f}s")

# Score all rows with Phase 1 ensemble
print("[STEP 1] Scoring with AttentionRXTJ + IFM...")
BATCH    = 2048
all_risk = []
t0       = time.time()

with torch.no_grad():
    for s in range(0, len(X_ipca), BATCH):
        e          = min(s + BATCH, len(X_ipca))
        probs, _   = p1_model(torch.FloatTensor(X_ipca[s:e]).to(DEVICE))
        probs_np   = probs.cpu().numpy()
        ifm_norm   = 1.0 / (1.0 + np.exp(ifm.decision_function(X_ipca[s:e])))
        all_risk.extend((W_MODEL * probs_np + W_IFM * ifm_norm).tolist())
        if (s // BATCH) % 50 == 0:
            print(f"  {e:>7,}/{len(X_ipca):,}")

p1_scores = np.array(all_risk, dtype=np.float32)
print(f"\n  P1 scores — min={p1_scores.min():.4f}  mean={p1_scores.mean():.4f}  "
      f"max={p1_scores.max():.4f}  std={p1_scores.std():.4f}")
print(f"  Predicted fraud @ threshold: {(p1_scores>=THRESHOLD).mean()*100:.2f}%")

if p1_scores.std() < 0.01:
    print("  WARNING: std too low — model may not have loaded correctly")
    sys.exit(1)

# Save and patch
np.save(os.path.join(DATA_DIR, "model_probs_full.npy"), p1_scores)
X_ctx       = np.load(os.path.join(DATA_DIR, "contextual_features.npy"))
X_ctx[:, 7] = p1_scores
np.save(os.path.join(DATA_DIR, "contextual_features.npy"), X_ctx)
print(f"  ✓ contextual_features.npy col-7 patched with real P1 scores")

# ════════════════════════════════════════════════════════════
# STEP 2 — Retrain FusionNet v2
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Retraining FusionNet v2")
print("=" * 60)

X        = np.load(os.path.join(DATA_DIR, "contextual_features.npy"))
y        = np.load(os.path.join(DATA_DIR, "fusion_labels.npy"))
acct_ids = np.load(os.path.join(DATA_DIR, "fusion_account_ids.npy"))
with open(os.path.join(DATA_DIR, "fusion_feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

print(f"\n  X={X.shape}  positives={int(y.sum()):,} ({100*y.mean():.2f}%)")
print(f"  P1 col: min={X[:,7].min():.4f}  max={X[:,7].max():.4f}  "
      f"std={X[:,7].std():.4f}  ← must be > 0.01")

X = np.nan_to_num(X, nan=0.0)

# Scale
fusion_scaler = StandardScaler()
X_sc2 = fusion_scaler.fit_transform(X).astype(np.float32)
joblib.dump(fusion_scaler, os.path.join(MODEL_DIR, "fusion_scaler.pkl"))
print("  fusion_scaler.pkl updated")

# Account-stratified split
unique_a = np.unique(acct_ids)
fa       = set(acct_ids[y == 1])
al       = np.array([1 if a in fa else 0 for a in unique_a])
a_tr, a_tmp, _, _ = train_test_split(unique_a, al, test_size=0.30,
                                      stratify=al, random_state=42)
a_va, a_te, _, _  = train_test_split(
    a_tmp,
    np.array([1 if a in fa else 0 for a in a_tmp]),
    test_size=0.50, random_state=42)

X_tr, y_tr = X_sc2[np.isin(acct_ids, a_tr)], y[np.isin(acct_ids, a_tr)]
X_va, y_va = X_sc2[np.isin(acct_ids, a_va)], y[np.isin(acct_ids, a_va)]
X_te, y_te = X_sc2[np.isin(acct_ids, a_te)], y[np.isin(acct_ids, a_te)]
print(f"  Train={len(X_tr):,}  Val={len(X_va):,}  Test={len(X_te):,}")


# ── FusionNet v2: wider architecture ─────────────────────────────────────────
class FusionNet(nn.Module):
    """Attention-weighted MLP: 8 → 64 → 32 → 16 → 1."""
    def __init__(self, input_dim=8):
        super().__init__()
        self.feature_attn = nn.Parameter(torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.feature_attn, dim=0)
        return torch.sigmoid(self.net(x * w)).squeeze(1), w


def raw_logits(m, x):
    return m.net(x * torch.softmax(m.feature_attn, dim=0)).squeeze(1)


model_f   = FusionNet(X_tr.shape[1]).to(DEVICE)
pw        = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).float()
criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(DEVICE))
opt       = torch.optim.Adam(model_f.parameters(), lr=5e-4, weight_decay=1e-4)
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

print(f"\n  FusionNet v2  input={X_tr.shape[1]}  pos_weight={pw.item():.2f}")

loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
    batch_size=4096, shuffle=True)
Xv_t = torch.FloatTensor(X_va).to(DEVICE)

best_auc, best_st, no_imp = 0.0, None, 0
t_tr = time.time()
print("\n  Training (target: AUC ≥ 0.85)...")

for ep in range(1, 201):
    model_f.train()
    ep_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        l = criterion(raw_logits(model_f, xb), yb)
        l.backward()
        nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
        opt.step()
        ep_loss += l.item() * len(xb)
    sched.step()

    model_f.eval()
    with torch.no_grad():
        vp = torch.sigmoid(raw_logits(model_f, Xv_t)).cpu().numpy()
    vauc = roc_auc_score(y_va, vp)

    if ep % 10 == 0 or ep <= 5:
        print(f"  Epoch {ep:>3}  loss={ep_loss/len(X_tr):.4f}  "
              f"AUC={vauc:.4f}  lr={opt.param_groups[0]['lr']:.6f}")

    if vauc > best_auc:
        best_auc = vauc; no_imp = 0
        best_st = {k: v.clone() for k, v in model_f.state_dict().items()}
    else:
        no_imp += 1
        if no_imp >= 25:
            print(f"  Early stopping at epoch {ep}")
            break

model_f.load_state_dict(best_st)
model_f.eval()
print(f"\n  Best val AUC: {best_auc:.4f}  ({time.time()-t_tr:.0f}s)")

# ── Jaya threshold ────────────────────────────────────────────────────────────
print("\n  Jaya threshold optimisation...")
with torch.no_grad():
    vp_np = torch.sigmoid(raw_logits(model_f, Xv_t)).cpu().numpy()

def cfn(t, p, l):
    pred = (p >= t).astype(int)
    fp = ((pred==1)&(l==0)).sum(); fn = ((pred==0)&(l==1)).sum()
    tp = ((pred==1)&(l==1)).sum(); tn = ((pred==0)&(l==0)).sum()
    return 2.0*fp/(fp+tn+1e-9) + fn/(fn+tp+1e-9)

pop = np.random.uniform(0.1, 0.9, 40)
c   = np.array([cfn(t, vp_np, y_va) for t in pop])
for _ in range(150):
    bi, wi = np.argmin(c), np.argmax(c)
    r1, r2 = np.random.rand(40), np.random.rand(40)
    np2    = np.clip(pop + r1*(pop[bi]-abs(pop)) - r2*(pop[wi]-abs(pop)), 0.05, 0.95)
    nc     = np.array([cfn(t, vp_np, y_va) for t in np2])
    m      = nc < c; pop = np.where(m, np2, pop); c = np.where(m, nc, c)

OPT_T  = float(pop[np.argmin(c)])
HIGH_T = min(OPT_T + 0.15, 0.90)
ELEV_T = max(OPT_T - 0.10, 0.25)
print(f"  Optimal={OPT_T:.4f}  HIGH={HIGH_T:.4f}  ELEVATED={ELEV_T:.4f}")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\n  Test evaluation...")
Xt_t = torch.FloatTensor(X_te).to(DEVICE)
with torch.no_grad():
    tp_p = torch.sigmoid(raw_logits(model_f, Xt_t)).cpu().numpy()
    attn = torch.softmax(model_f.feature_attn, dim=0).cpu().numpy()

preds = (tp_p >= OPT_T).astype(int)
auc   = roc_auc_score(y_te, tp_p)
mcc   = matthews_corrcoef(y_te, preds)
prec  = precision_score(y_te, preds, zero_division=0)
rec   = recall_score(y_te, preds, zero_division=0)
f1    = f1_score(y_te, preds, zero_division=0)
cm    = confusion_matrix(y_te, preds)
TP, FP, FN, TN = int(cm[1,1]), int(cm[0,1]), int(cm[1,0]), int(cm[0,0])

print(f"\n  ┌──────────────────────────────────────┐")
print(f"  │  AUC-ROC   : {auc:.4f}                  │")
print(f"  │  MCC       : {mcc:.4f}                  │")
print(f"  │  Precision : {prec:.4f}                  │")
print(f"  │  Recall    : {rec:.4f}                  │")
print(f"  │  F1        : {f1:.4f}                  │")
print(f"  │  TP={TP:<5} FP={FP:<6} TN={TN:<6} FN={FN:<4}  │")
print(f"  └──────────────────────────────────────┘")

print(f"\n  Learned attention weights:")
for i, name in enumerate(FEATURE_NAMES):
    bar = "█" * int(attn[i] * 60)
    print(f"    {name:<25}  {attn[i]:.4f}  {bar}")

# ── Save everything ───────────────────────────────────────────────────────────
fpr_a, tpr_a, _ = roc_curve(y_te, tp_p)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(fpr_a, tpr_a, color="#0099cc", lw=2,
             label=f"FusionNet v2 AUC={auc:.4f}")
axes[0].plot([0,1],[0,1],"k--",lw=0.8)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("FusionNet v2 ROC"); axes[0].legend()
axes[1].bar(FEATURE_NAMES, attn, color="#00d4ff", edgecolor="#0099cc")
axes[1].set_title("Feature Attention Weights v2")
axes[1].set_ylabel("Weight"); axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fusion_roc_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()

torch.save(model_f.state_dict(), os.path.join(MODEL_DIR, "fusion_net.pt"))

cfg_out = {
    "input_dim":            X_tr.shape[1],
    "fusion_threshold":     round(OPT_T, 6),
    "high_threshold":       round(HIGH_T, 6),
    "elevated_threshold":   round(ELEV_T, 6),
    "fusion_auc":           round(float(auc), 6),
    "fusion_mcc":           round(float(mcc), 6),
    "fusion_precision":     round(float(prec), 6),
    "fusion_recall":        round(float(rec), 6),
    "fusion_f1":            round(float(f1), 6),
    "true_positives":       TP, "false_positives": FP,
    "true_negatives":       TN, "false_negatives": FN,
    "feature_names":        FEATURE_NAMES,
    "optimal_attn_weights": [round(float(w), 6) for w in attn],
    "model_version":        "fusionnet_v2",
    "best_val_auc":         round(float(best_auc), 6),
}
with open(os.path.join(RESULTS_DIR, "fusion_config.json"), "w") as f:
    json.dump(cfg_out, f, indent=2)

print(f"\n  fusion_net.pt        → updated")
print(f"  fusion_scaler.pkl    → updated")
print(f"  fusion_config.json   → updated")
print(f"  fusion_roc_v2.png    → saved")
print(f"\n{'='*60}")
print(f"  ✓ Done.  AUC={auc:.4f}  MCC={mcc:.4f}  threshold={OPT_T:.4f}")
print(f"  NEXT STEP → merge app_phase2_endpoints.py into app.py")
print(f"{'='*60}")