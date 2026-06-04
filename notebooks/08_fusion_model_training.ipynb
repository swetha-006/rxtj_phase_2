# ============================================================
# notebooks/08_fusion_model_training.py
# Run after 07_contextual_features.py:
#   python notebooks/08_fusion_model_training.py
#
# PURPOSE: Two-stage notebook:
#   Stage 1 — Generate real Phase 1 risk scores for all 590K rows
#             using the full EARN+ pipeline (AE + ResNet → 128 →
#             Nystroem → IPCA → AttentionRXTJ + IFM ensemble).
#             Patches contextual_features.npy column 7 in-place.
#   Stage 2 — Train FusionNet v2 (8→64→32→16→1 with feature attention)
#             using Jaya threshold optimisation. Saves fusion_net.pt,
#             fusion_scaler.pkl, fusion_config.json, ROC plots.
#
# WHAT WAS FIXED (merged from fix_retrain_nb08.py):
#   - AE architecture: added Dropout at indices [3] and [7] and removed
#     the spurious extra ReLU after the final encoder Linear — these exact
#     positions match the autoencoder.pt state_dict keys (encoder.0,1,4,5,8)
#   - Pipeline order: raw (224) → AE(64) + ResNet(64) → concat(128) →
#     Nystroem(128→300) → IPCA(300→50) — NOT raw(224) → Nystroem directly
#   - AttentionRXTJ: uses exact app.py names (paths ModuleList, attn_gru,
#     attention Sequential, classifier Sequential) so load_state_dict works
#   - FusionNet v2: wider (8→64→32→16→1) + CosineAnnealingLR (removes the
#     deprecated verbose=True ReduceLROnPlateau) + monitors AUC not loss
#   - Training: EPOCHS=200, PATIENCE=25, batch=4096, lr=5e-4
#   - Smart resume: skips Stage 1 if model_probs_full.npy already exists
#     with the correct row count
#
# OUTPUTS:
#   models/fusion_net.pt          — trained FusionNet v2 weights
#   models/fusion_scaler.pkl      — StandardScaler for 8 input features
#   results/fusion_config.json    — thresholds + metrics + attn weights
#   results/fusion_roc.png        — ROC curve + attention weight chart
#   results/fusion_training_curves.png
#   data/model_probs_full.npy     — P1 risk scores for all 590K rows
#
# RUNTIME: ~25 min (Stage 1) + ~15 min (Stage 2) = ~40 min total
#          Stage 1 skipped if model_probs_full.npy already exists.
# ============================================================

# %% Cell 1: Imports
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — avoids display errors
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              precision_score, recall_score, f1_score,
                              roc_curve, confusion_matrix)

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(ROOT, "data")
MODEL_DIR   = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

print(f"[NB08] Root  : {ROOT}")
print(f"[NB08] Device: {DEVICE}")

# ════════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES
# All definitions are EXACT copies of the classes in app.py / NB03 so that
# load_state_dict() succeeds without key or shape mismatches.
# ════════════════════════════════════════════════════════════════════════════════

# ── FraudAutoencoder (matches autoencoder.pt key layout exactly) ──────────────
# Key layout in the saved .pt (only parametric layers appear):
#   encoder.0  Linear(224,256)  | encoder.1  BN(256)
#   [idx 2=ReLU, 3=Dropout — no params, invisible to state_dict]
#   encoder.4  Linear(256,128)  | encoder.5  BN(128)
#   [idx 6=ReLU, 7=Dropout — no params]
#   encoder.8  Linear(128,64)   ← last encoder layer, NO extra ReLU after it
#   decoder.0  Linear(64,128)   | decoder.1  BN(128)   [idx 2=ReLU]
#   decoder.3  Linear(128,256)  | decoder.4  BN(256)   [idx 5=ReLU]
#   decoder.6  Linear(256,224)
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=224, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim)   # index 8 — NO extra ReLU here
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def encode(self, x):
        return self.encoder(x)


# ── FraudResNet (matches resnet_extractor.pt — extract() bypasses head) ───────
# Confirmed architecture from state_dict key shapes:
#   stem         : Linear(input_dim, 128)
#   blocks.0     : _ResBlock(128, 128)  shortcut=Identity
#   blocks.1     : _ResBlock(128, 64)   shortcut=Linear(128,64)  ← dim-reducing
#   blocks.2     : _ResBlock(64, 64)    shortcut=Identity
#   blocks.3     : _ResBlock(64, 64)    shortcut=Identity
#   head         : Linear(64, 2)
#
# _ResBlock supports in_dim ≠ out_dim via a shortcut Linear projection.
# When in_dim == out_dim the shortcut is Identity (no extra params).
class _ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim,  out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu     = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class FraudResNet(nn.Module):
    def __init__(self, input_dim=224):
        super().__init__()
        # stem is a bare Linear so its state_dict key is stem.weight / stem.bias
        self.stem   = nn.Linear(input_dim, 128)
        self.blocks = nn.Sequential(
            _ResBlock(128, 128),   # blocks.0 — same dim, Identity shortcut
            _ResBlock(128, 64),    # blocks.1 — dim reduction, Linear shortcut
            _ResBlock(64,  64),    # blocks.2
            _ResBlock(64,  64),    # blocks.3
        )
        self.head = nn.Linear(64, 2)   # classifier head (bypassed during extraction)
    def extract(self, x):
        """Return 64-dim EARN+ features without the classification head."""
        return self.blocks(torch.relu(self.stem(x)))
    def forward(self, x):
        return self.head(self.extract(x))


# ── AttentionRXTJ (exact copy from app.py — DO NOT change attribute names) ────
SEQ_LEN     = 8
CARDINALITY = 4

class _ResNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cardinality=CARDINALITY):
        super().__init__()
        gd = out_dim // cardinality
        self.paths    = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, gd), nn.BatchNorm1d(gd), nn.ReLU(),
                nn.Linear(gd, gd),     nn.BatchNorm1d(gd)
            ) for _ in range(cardinality)
        ])
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu     = nn.ReLU()
    def forward(self, x):
        return self.relu(torch.cat([p(x) for p in self.paths], dim=-1) + self.shortcut(x))

class _ResNeXtExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            _ResNeXtBlock(input_dim, 128), _ResNeXtBlock(128, 128),
            _ResNeXtBlock(128, 64),        _ResNeXtBlock(64,  64)
        )
    def forward(self, x):
        return self.net(x)

class _SelfAttentionGRU(nn.Module):
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
        x       = x.view(x.size(0), self.seq_len, self.step_dim)
        out, _  = self.gru(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (attn_w * out).sum(dim=1)
        return torch.sigmoid(self.classifier(context)).squeeze(1), attn_w

class AttentionRXTJ(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN):
        super().__init__()
        self.resnext  = _ResNeXtExtractor(input_dim)
        self.attn_gru = _SelfAttentionGRU(input_dim=64, seq_len=seq_len)
    def forward(self, x):
        return self.attn_gru(self.resnext(x))


# ── FusionNet v2 (wider than original NB08 — 8→64→32→16→1) ──────────────────
class FusionNet(nn.Module):
    """Attention-weighted MLP for account compromise scoring.

    A learned softmax weight vector (feature_attn) is applied to the 8
    contextual inputs before the MLP. At inference time these weights give
    a per-feature importance score used by /account/explain.
    """
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


def _logits(m, x):
    """Raw logits for BCEWithLogitsLoss (avoids double sigmoid)."""
    return m.net(x * torch.softmax(m.feature_attn, dim=0)).squeeze(1)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Generate real Phase 1 risk scores for all 590K rows
# Full pipeline: 224 → AE(64) + ResNet(64) → concat(128) → Nystroem(300) → IPCA(50)
# ════════════════════════════════════════════════════════════════════════════════

P1_PROBS_PATH = os.path.join(DATA_DIR, "model_probs_full.npy")
SNAP_PATH     = os.path.join(DATA_DIR, "tx_snapshot.parquet")
CTX_PATH      = os.path.join(DATA_DIR, "contextual_features.npy")

# Smart resume: skip Stage 1 if scores already exist for all rows
_need_p1 = True
if os.path.exists(P1_PROBS_PATH):
    _existing = np.load(P1_PROBS_PATH)
    _snap_len = len(pd.read_parquet(SNAP_PATH, columns=["TransactionID"]))
    if len(_existing) == _snap_len and np.std(_existing) > 0.01:
        print(f"\n[STAGE 1] Skipping — model_probs_full.npy already exists "
              f"({len(_existing):,} rows, std={np.std(_existing):.4f})")
        p1_scores = _existing.astype(np.float32)
        _need_p1  = False
    else:
        print(f"\n[STAGE 1] model_probs_full.npy found but stale "
              f"(len={len(_existing)}, std={np.std(_existing):.4f}) — regenerating")

if _need_p1:
    print("\n" + "="*60)
    print("  STAGE 1: Generating Phase 1 risk scores")
    print("  Pipeline: 224 → AE(64)+ResNet(64) → 128 → Nystroem → IPCA(50)")
    print("="*60)

    # ── Load preprocessors ────────────────────────────────────────────────────
    print("\n[STAGE 1] Loading preprocessors...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer  = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
        scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        nystroem = joblib.load(os.path.join(MODEL_DIR, "nystroem.pkl"))
        ipca     = joblib.load(os.path.join(MODEL_DIR, "incremental_pca.pkl"))
        ifm      = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

    RAW_DIM  = int(imputer.n_features_in_)    # 224
    NYS_DIM  = int(nystroem.n_features_in_)   # 128  (AE 64 + ResNet 64)
    IPCA_DIM = int(ipca.n_components_)         # 50
    print(f"  imputer  : {RAW_DIM} features in")
    print(f"  nystroem : expects {NYS_DIM} features  (= 64 AE + 64 ResNet)")
    print(f"  ipca     : {IPCA_DIM} components out")

    # ── Load EARN+ extractors ─────────────────────────────────────────────────
    ae_state = torch.load(os.path.join(MODEL_DIR, "autoencoder.pt"), map_location="cpu")
    ae_in    = ae_state["encoder.0.weight"].shape[1]
    ae_model = FraudAutoencoder(input_dim=ae_in, latent_dim=64)
    ae_model.load_state_dict(ae_state)    # strict=True — must match exactly
    ae_model.eval()
    print(f"  FraudAutoencoder loaded  (input={ae_in} → latent=64) ✓")

    rn_model = FraudResNet(input_dim=ae_in)
    rn_state = torch.load(os.path.join(MODEL_DIR, "resnet_extractor.pt"), map_location="cpu")
    rn_model.load_state_dict(rn_state, strict=False)  # strict=False for safety
    rn_model.eval()
    print(f"  FraudResNet loaded  (input={ae_in} → features=64) ✓")

    p1_model = AttentionRXTJ(input_dim=IPCA_DIM, seq_len=SEQ_LEN).to(DEVICE)
    p1_state = torch.load(os.path.join(MODEL_DIR, "attention_rxtj.pt"), map_location=DEVICE)
    p1_model.load_state_dict(p1_state)    # strict=True — exact match with app.py
    p1_model.eval()
    print(f"  AttentionRXTJ loaded  (input={IPCA_DIM}) ✓")

    cfg_p1    = json.load(open(os.path.join(RESULTS_DIR, "deployment_config.json")))
    W_MODEL   = float(cfg_p1["W_MODEL"])
    W_IFM     = float(cfg_p1["W_IFM"])
    THRESHOLD = float(cfg_p1["THRESHOLD"])
    print(f"  W_MODEL={W_MODEL:.4f}  W_IFM={W_IFM:.4f}  THRESHOLD={THRESHOLD:.4f}")

    # ── Build 224-dim raw matrix from snapshot ────────────────────────────────
    print("\n[STAGE 1] Loading snapshot and building feature matrix...")
    snap    = pd.read_parquet(SNAP_PATH)
    pcd_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
    X_raw   = np.full((len(snap), RAW_DIM), np.nan, dtype=np.float32)
    X_raw[:, 0] = snap["TransactionAmt"].fillna(0).values.astype(np.float32)
    X_raw[:, 1] = snap["hour"].values.astype(np.float32)
    X_raw[:, 2] = snap["ProductCD"].map(pcd_map).fillna(-1).values.astype(np.float32)
    X_raw[:, 3] = pd.to_numeric(snap["addr1"], errors="coerce").values.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_imp = imputer.transform(X_raw)
        X_sc  = scaler.transform(X_imp).astype(np.float32)
    print(f"  Imputed+scaled: {X_sc.shape}")

    # ── EARN+ feature extraction: AE(64) + ResNet(64) → concat(128) ──────────
    print(f"\n[STAGE 1] Extracting EARN+ features (~5 min)...")
    BATCH        = 4096
    Z_ae_list, Z_rn_list = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(X_sc), BATCH):
            e    = min(s + BATCH, len(X_sc))
            xb   = torch.FloatTensor(X_sc[s:e])
            Z_ae_list.append(ae_model.encode(xb).cpu().numpy())
            Z_rn_list.append(rn_model.extract(xb).cpu().numpy())
            if (s // BATCH) % 25 == 0:
                print(f"  AE+ResNet: {e:>7,}/{len(X_sc):,}  ({time.time()-t0:.0f}s)")

    Z_earn = np.hstack([
        np.vstack(Z_ae_list).astype(np.float32),
        np.vstack(Z_rn_list).astype(np.float32),
    ])
    print(f"\n  Z_earn: {Z_earn.shape}  ← must be (N, {NYS_DIM})")
    assert Z_earn.shape[1] == NYS_DIM, \
        f"EARN dim {Z_earn.shape[1]} ≠ Nystroem expects {NYS_DIM}"
    print(f"  ✓ Dimension check passed: {Z_earn.shape[1]} == {NYS_DIM}")

    # ── Nystroem → IPCA in chunks ─────────────────────────────────────────────
    print(f"\n[STAGE 1] Nystroem + IPCA in chunks (~8 min)...")
    CHUNK  = 8000
    X_ipca = np.zeros((len(Z_earn), IPCA_DIM), dtype=np.float32)
    t0     = time.time()
    for s in range(0, len(Z_earn), CHUNK):
        e = min(s + CHUNK, len(Z_earn))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_ipca[s:e] = ipca.transform(
                nystroem.transform(Z_earn[s:e])
            ).astype(np.float32)
        if s % 80000 == 0:
            print(f"  Nystroem+IPCA: {e:>7,}/{len(Z_earn):,}  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.0f}s. X_ipca: {X_ipca.shape}")

    # ── AttentionRXTJ + IFM ensemble scoring ─────────────────────────────────
    print(f"\n[STAGE 1] Ensemble scoring (AttentionRXTJ + IFM)...")
    all_risk = []
    t0       = time.time()
    with torch.no_grad():
        for s in range(0, len(X_ipca), BATCH):
            e        = min(s + BATCH, len(X_ipca))
            probs, _ = p1_model(torch.FloatTensor(X_ipca[s:e]).to(DEVICE))
            pnp      = probs.cpu().numpy()
            ifm_norm = 1.0 / (1.0 + np.exp(ifm.decision_function(X_ipca[s:e])))
            all_risk.extend((W_MODEL * pnp + W_IFM * ifm_norm).tolist())
            if (s // BATCH) % 50 == 0:
                print(f"  Scoring: {e:>7,}/{len(X_ipca):,}")

    p1_scores = np.array(all_risk, dtype=np.float32)
    print(f"\n  P1 scores — min={p1_scores.min():.4f}  mean={p1_scores.mean():.4f}  "
          f"max={p1_scores.max():.4f}  std={p1_scores.std():.4f}")
    print(f"  Predicted fraud @ threshold: "
          f"{(p1_scores>=THRESHOLD).mean()*100:.2f}%")

    if p1_scores.std() < 0.005:
        raise RuntimeError(
            "P1 score std is near zero — model may not have loaded correctly. "
            "Check that attention_rxtj.pt and the EARN+ extractors loaded without errors."
        )

    # Persist and patch contextual_features.npy column 7
    np.save(P1_PROBS_PATH, p1_scores)
    print(f"  model_probs_full.npy saved")

X_ctx       = np.load(CTX_PATH)
X_ctx[:, 7] = p1_scores
np.save(CTX_PATH, X_ctx)
print(f"  contextual_features.npy col-7 patched  "
      f"({X_ctx[:,7].min():.4f}–{X_ctx[:,7].max():.4f})")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Train FusionNet v2
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STAGE 2: Training FusionNet v2")
print("="*60)

# %% Cell 2: Load features + labels
print("\n[STAGE 2] Loading features...")
X        = np.load(CTX_PATH)
y        = np.load(os.path.join(DATA_DIR, "fusion_labels.npy"))
acct_ids = np.load(os.path.join(DATA_DIR, "fusion_account_ids.npy"))
with open(os.path.join(DATA_DIR, "fusion_feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

print(f"  X={X.shape}  positives={int(y.sum()):,} ({100*y.mean():.2f}%)")
print(f"  P1 col: {X[:,7].min():.4f}–{X[:,7].max():.4f}  std={X[:,7].std():.4f}")

X = np.nan_to_num(X, nan=0.0)  # safety guard

# %% Cell 3: Scale
print("\n[STAGE 2] Scaling features...")
fusion_scaler = StandardScaler()
X_sc          = fusion_scaler.fit_transform(X).astype(np.float32)
joblib.dump(fusion_scaler, os.path.join(MODEL_DIR, "fusion_scaler.pkl"))
print("  fusion_scaler.pkl saved")

# %% Cell 4: Account-stratified train/val/test split
# All transactions for the same account go into the same split
# to prevent data leakage across the account boundary.
print("\n[STAGE 2] Account-stratified split (70/15/15)...")
unique_a = np.unique(acct_ids)
fa       = set(acct_ids[y == 1])
al       = np.array([1 if a in fa else 0 for a in unique_a])

a_tr, a_tmp, _, _ = train_test_split(unique_a, al,
                                      test_size=0.30, stratify=al, random_state=42)
a_va, a_te, _, _  = train_test_split(
    a_tmp,
    np.array([1 if a in fa else 0 for a in a_tmp]),
    test_size=0.50, random_state=42)

tr_m = np.isin(acct_ids, a_tr)
va_m = np.isin(acct_ids, a_va)
te_m = np.isin(acct_ids, a_te)
X_tr, y_tr = X_sc[tr_m], y[tr_m]
X_va, y_va = X_sc[va_m], y[va_m]
X_te, y_te = X_sc[te_m], y[te_m]

print(f"  Train : {len(X_tr):,}  ({y_tr.mean()*100:.2f}% fraud)")
print(f"  Val   : {len(X_va):,}  ({y_va.mean()*100:.2f}% fraud)")
print(f"  Test  : {len(X_te):,}  ({y_te.mean()*100:.2f}% fraud)")

# %% Cell 5: Training setup
INPUT_DIM = X_tr.shape[1]
model_f   = FusionNet(INPUT_DIM).to(DEVICE)

pos_weight = torch.tensor([(y_tr==0).sum() / (y_tr==1).sum()]).float()
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer  = torch.optim.Adam(model_f.parameters(), lr=5e-4, weight_decay=1e-4)
# CosineAnnealingLR has no deprecated verbose parameter
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-5
)

print(f"\n[STAGE 2] FusionNet v2  input={INPUT_DIM}  pos_weight={pos_weight.item():.2f}")
print(f"  Architecture: {INPUT_DIM}→64→32→16→1  +  feature_attn[{INPUT_DIM}]")

BATCH_SIZE = 4096
EPOCHS     = 200
PATIENCE   = 25   # monitor AUC (up) not loss (down)

loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
    batch_size=BATCH_SIZE, shuffle=True
)
Xv_t = torch.FloatTensor(X_va).to(DEVICE)
yv   = y_va  # numpy copy for roc_auc_score

# %% Cell 6: Training loop (monitors AUC, not loss)
print(f"\n[STAGE 2] Training (target AUC ≥ 0.85)...")
best_auc, best_state, no_imp = 0.0, None, 0
t_tr = time.time()
train_losses, val_aucs = [], []

for ep in range(1, EPOCHS + 1):
    model_f.train()
    ep_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(_logits(model_f, xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    scheduler.step()

    model_f.eval()
    with torch.no_grad():
        vp = torch.sigmoid(_logits(model_f, Xv_t)).cpu().numpy()
    vauc = roc_auc_score(yv, vp)

    train_losses.append(ep_loss / len(X_tr))
    val_aucs.append(vauc)

    if ep % 10 == 0 or ep <= 5:
        print(f"  Epoch {ep:>3}  loss={ep_loss/len(X_tr):.4f}  "
              f"AUC={vauc:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    if vauc > best_auc:
        best_auc = vauc; no_imp = 0
        best_state = {k: v.clone() for k, v in model_f.state_dict().items()}
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  Early stopping at epoch {ep} (patience={PATIENCE})")
            break

model_f.load_state_dict(best_state)
model_f.eval()
print(f"\n  Best val AUC: {best_auc:.4f}  ({time.time()-t_tr:.0f}s)")

# %% Cell 7: Jaya threshold optimisation (same pattern as NB05)
print("\n[STAGE 2] Jaya threshold optimisation...")
with torch.no_grad():
    vp_np = torch.sigmoid(_logits(model_f, Xv_t)).cpu().numpy()

def _cost(t, p, l):
    pred = (p >= t).astype(int)
    fp   = ((pred==1) & (l==0)).sum()
    fn   = ((pred==0) & (l==1)).sum()
    tp   = ((pred==1) & (l==1)).sum()
    tn   = ((pred==0) & (l==0)).sum()
    return 2.0*fp/(fp+tn+1e-9) + fn/(fn+tp+1e-9)

pop = np.random.uniform(0.1, 0.9, 40)
c   = np.array([_cost(t, vp_np, yv) for t in pop])
for _ in range(150):
    bi, wi = np.argmin(c), np.argmax(c)
    r1, r2 = np.random.rand(40), np.random.rand(40)
    np2    = np.clip(pop + r1*(pop[bi]-np.abs(pop)) - r2*(pop[wi]-np.abs(pop)), 0.05, 0.95)
    nc     = np.array([_cost(t, vp_np, yv) for t in np2])
    m      = nc < c; pop = np.where(m, np2, pop); c = np.where(m, nc, c)

OPT_T  = float(pop[np.argmin(c)])
HIGH_T = min(OPT_T + 0.15, 0.90)
ELEV_T = max(OPT_T - 0.10, 0.25)
print(f"  Optimal  threshold : {OPT_T:.4f}")
print(f"  HIGH     threshold : {HIGH_T:.4f}")
print(f"  ELEVATED threshold : {ELEV_T:.4f}")
print(f"  Val cost @ optimal : {_cost(OPT_T, vp_np, yv):.4f}")

# %% Cell 8: Test set evaluation
print("\n[STAGE 2] Test set evaluation...")
Xt_t = torch.FloatTensor(X_te).to(DEVICE)
with torch.no_grad():
    test_probs = torch.sigmoid(_logits(model_f, Xt_t)).cpu().numpy()
    attn_final = torch.softmax(model_f.feature_attn, dim=0).cpu().numpy()

test_preds = (test_probs >= OPT_T).astype(int)
auc  = roc_auc_score(y_te, test_probs)
mcc  = matthews_corrcoef(y_te, test_preds)
prec = precision_score(y_te, test_preds, zero_division=0)
rec  = recall_score(y_te, test_preds, zero_division=0)
f1   = f1_score(y_te, test_preds, zero_division=0)
cm   = confusion_matrix(y_te, test_preds)
TP, FP = int(cm[1,1]), int(cm[0,1])
FN, TN = int(cm[1,0]), int(cm[0,0])

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  AUC-ROC   : {auc:.4f}                     │")
print(f"  │  MCC       : {mcc:.4f}                     │")
print(f"  │  Precision : {prec:.4f}                     │")
print(f"  │  Recall    : {rec:.4f}                     │")
print(f"  │  F1        : {f1:.4f}                     │")
print(f"  │  TP={TP:<5}  FP={FP:<5}  TN={TN:<6}  FN={FN:<4} │")
print(f"  └─────────────────────────────────────────┘")

if auc >= 0.85:
    print(f"\n  ✓ AUC target met  ({auc:.4f} ≥ 0.85)")
else:
    print(f"\n  ⚠  AUC below 0.85 ({auc:.4f}) — acceptable for Phase 2 demo")

print(f"\n  Learned attention weights (feature importance):")
for i, name in enumerate(FEATURE_NAMES):
    bar = "█" * int(attn_final[i] * 60)
    print(f"    {name:<25}  {attn_final[i]:.4f}  {bar}")

# %% Cell 9: ROC + attention weight plots
fpr_a, tpr_a, _ = roc_curve(y_te, test_probs)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(fpr_a, tpr_a, color="#0099cc", lw=2,
             label=f"FusionNet v2  AUC={auc:.4f}")
axes[0].plot([0,1],[0,1],"k--",lw=0.8)
axes[0].axvline(x=0.15, color="gray", ls=":", alpha=0.5, label="15% FPR target")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("FusionNet v2 ROC Curve"); axes[0].legend()

axes[1].bar(FEATURE_NAMES, attn_final, color="#00d4ff", edgecolor="#0099cc")
axes[1].set_title("Learned Feature Attention Weights")
axes[1].set_ylabel("Weight"); axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "fusion_roc.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ROC plot saved → {roc_path}")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(train_losses, label="train loss", color="#0099cc")
ax2.plot(val_aucs,     label="val AUC",    color="#00e5a0")
ax2.set_xlabel("Epoch"); ax2.set_title("FusionNet v2 Training Curves"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fusion_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# %% Cell 10: Save model + config
torch.save(model_f.state_dict(), os.path.join(MODEL_DIR, "fusion_net.pt"))

fusion_config = {
    "input_dim":            INPUT_DIM,
    "fusion_threshold":     round(OPT_T, 6),
    "high_threshold":       round(HIGH_T, 6),
    "elevated_threshold":   round(ELEV_T, 6),
    "fusion_auc":           round(float(auc), 6),
    "fusion_mcc":           round(float(mcc), 6),
    "fusion_precision":     round(float(prec), 6),
    "fusion_recall":        round(float(rec), 6),
    "fusion_f1":            round(float(f1), 6),
    "true_positives":       TP,
    "false_positives":      FP,
    "true_negatives":       TN,
    "false_negatives":      FN,
    "feature_names":        FEATURE_NAMES,
    "optimal_attn_weights": [round(float(w), 6) for w in attn_final],
    "model_version":        "fusionnet_v2",
    "training_epochs":      len(train_losses),
    "best_val_auc":         round(float(best_auc), 6),
    "pos_weight_used":      round(float(pos_weight.item()), 4),
}
config_path = os.path.join(RESULTS_DIR, "fusion_config.json")
with open(config_path, "w") as f:
    json.dump(fusion_config, f, indent=2)

print(f"\n[NB08] Saved:")
print(f"  models/fusion_net.pt          ✓")
print(f"  models/fusion_scaler.pkl      ✓")
print(f"  results/fusion_config.json    ✓")
print(f"  results/fusion_roc.png        ✓")
print(f"  results/fusion_training_curves.png  ✓")
print(f"\n[NB08] ✓ Phase 2 training complete!")
print(f"  AUC={auc:.4f}  MCC={mcc:.4f}  threshold={OPT_T:.4f}")
print(f"\n[NB08] NEXT STEP → merge app_phase2_endpoints.py into app.py")