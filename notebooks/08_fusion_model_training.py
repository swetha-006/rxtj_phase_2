# ============================================================
# notebooks/08_fusion_model_training.py
# Run after 07_contextual_features.py
#
# PURPOSE: Train FusionNet — an attention-weighted MLP that combines
# Phase 1 + behavioral signals to output compromise_probability.
# Applies Jaya algorithm (same as NB05) for threshold optimization.
#
# RUNTIME: ~10–20 min (CPU training on 590K samples).
# INPUTS:
#   data/contextual_features.npy
#   data/fusion_labels.npy
#   data/fusion_account_ids.npy
#   data/fusion_feature_names.json
# OUTPUTS:
#   models/fusion_net.pt
#   models/fusion_scaler.pkl
#   results/fusion_config.json
#   results/fusion_roc.png
# ============================================================

# %% Cell 1: Imports
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              precision_score, recall_score, f1_score,
                              roc_curve, confusion_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA_DIR    = os.path.join(ROOT, "data")
MODEL_DIR   = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
print(f"[NB08] Root: {ROOT}")
print(f"[NB08] Device: {DEVICE}")

# %% Cell 2: Load data
print("\n[NB08] Loading features …")
X = np.load(os.path.join(DATA_DIR, "contextual_features.npy"))
y = np.load(os.path.join(DATA_DIR, "fusion_labels.npy"))
acct_ids = np.load(os.path.join(DATA_DIR, "fusion_account_ids.npy"))

with open(os.path.join(DATA_DIR, "fusion_feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Positive: {int(y.sum()):,} ({100*y.mean():.2f}%)")
print(f"  Features: {FEATURE_NAMES}")

# %% Cell 3: Scale features
print("\n[NB08] Scaling features …")
fusion_scaler = StandardScaler()
X_scaled = fusion_scaler.fit_transform(X).astype(np.float32)

joblib.dump(fusion_scaler, os.path.join(MODEL_DIR, "fusion_scaler.pkl"))
print(f"  fusion_scaler saved → models/fusion_scaler.pkl")

# %% Cell 4: Train/Val/Test split (stratified, account-aware)
# Use account_ids to prevent data leakage: all transactions from the
# same account go into the same split.
print("\n[NB08] Splitting data …")

unique_accts  = np.unique(acct_ids)
fraud_accts   = set(acct_ids[y == 1])
acct_labels   = np.array([1 if a in fraud_accts else 0 for a in unique_accts])

a_train, a_tmp, _, _ = train_test_split(
    unique_accts, acct_labels, test_size=0.30, stratify=acct_labels, random_state=42
)
a_val, a_test, _, _  = train_test_split(
    a_tmp,
    np.array([1 if a in fraud_accts else 0 for a in a_tmp]),
    test_size=0.50, random_state=42
)

train_mask = np.isin(acct_ids, a_train)
val_mask   = np.isin(acct_ids, a_val)
test_mask  = np.isin(acct_ids, a_test)

X_tr, y_tr = X_scaled[train_mask], y[train_mask]
X_va, y_va = X_scaled[val_mask],   y[val_mask]
X_te, y_te = X_scaled[test_mask],  y[test_mask]

print(f"  Train: {len(X_tr):,} ({y_tr.mean()*100:.2f}% fraud)")
print(f"  Val  : {len(X_va):,} ({y_va.mean()*100:.2f}% fraud)")
print(f"  Test : {len(X_te):,} ({y_te.mean()*100:.2f}% fraud)")

# %% Cell 5: FusionNet architecture
class FusionNet(nn.Module):
    """Attention-weighted MLP for account compromise scoring.

    A learned softmax weight vector applied to the 8 input features
    before the MLP acts as a feature importance (explainability) layer.
    The weights are inspected at inference time to explain which
    contextual feature drove the compromise score.
    """

    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.feature_attn = nn.Parameter(torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.feature_attn, dim=0)
        x_weighted   = x * attn_weights
        logit        = self.net(x_weighted).squeeze(1)
        return torch.sigmoid(logit), attn_weights


# %% Cell 6: Training setup
INPUT_DIM = X_tr.shape[1]
model     = FusionNet(INPUT_DIM).to(DEVICE)

pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, verbose=True
)

print(f"\n[NB08] FusionNet  input={INPUT_DIM}  pos_weight={pos_weight.item():.2f}")

BATCH_SIZE  = 2048
EPOCHS      = 120
PATIENCE    = 15

tr_dataset  = TensorDataset(
    torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
)
tr_loader   = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_va_t = torch.FloatTensor(X_va).to(DEVICE)
y_va_t = torch.FloatTensor(y_va).to(DEVICE)

# %% Cell 7: Training loop
print("\n[NB08] Training …")
train_losses, val_losses, val_aucs = [], [], []
best_val_loss = float("inf")
no_improve    = 0
best_state    = None

def raw_logits(model, x):
    """Forward pass returning raw logits for BCEWithLogitsLoss."""
    attn_weights = torch.softmax(model.feature_attn, dim=0)
    x_weighted   = x * attn_weights
    return model.net(x_weighted).squeeze(1)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = raw_logits(model, xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(X_tr)

    model.eval()
    with torch.no_grad():
        val_logits = raw_logits(model, X_va_t)
        val_loss   = criterion(val_logits, y_va_t).item()
        val_probs  = torch.sigmoid(val_logits).cpu().numpy()
    val_auc = roc_auc_score(y_va, val_probs)
    scheduler.step(val_loss)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)

    if epoch % 10 == 0 or epoch <= 5:
        attn = torch.softmax(model.feature_attn, dim=0).detach().numpy()
        print(f"  Epoch {epoch:>3}  train={epoch_loss:.4f}  val={val_loss:.4f}  "
              f"AUC={val_auc:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(best_state)
model.eval()
print(f"\n[NB08] Best val loss: {best_val_loss:.4f}")

# %% Cell 8: Jaya threshold optimization (same pattern as NB05)
# Minimise cost = 2.0 × FP_rate + 1.0 × FN_rate on validation set.
print("\n[NB08] Jaya threshold optimisation …")

with torch.no_grad():
    val_probs_np = torch.sigmoid(raw_logits(model, X_va_t)).cpu().numpy()

def fusion_cost(threshold, probs, labels):
    preds = (probs >= threshold).astype(int)
    fp    = int(((preds == 1) & (labels == 0)).sum())
    fn    = int(((preds == 0) & (labels == 1)).sum())
    tp    = int(((preds == 1) & (labels == 1)).sum())
    tn    = int(((preds == 0) & (labels == 0)).sum())
    fpr   = fp / (fp + tn + 1e-9)
    fnr   = fn / (fn + tp + 1e-9)
    return 2.0 * fpr + 1.0 * fnr

# Jaya: population of threshold candidates, iteratively pushed toward
# the best member and away from the worst.
POP_SIZE = 30
MAX_ITER = 100
pop      = np.random.uniform(0.1, 0.9, POP_SIZE)
costs    = np.array([fusion_cost(t, val_probs_np, y_va) for t in pop])

for iteration in range(MAX_ITER):
    best_idx  = np.argmin(costs)
    worst_idx = np.argmax(costs)
    best_t    = pop[best_idx]
    worst_t   = pop[worst_idx]
    r1, r2    = np.random.rand(POP_SIZE), np.random.rand(POP_SIZE)
    new_pop   = pop + r1*(best_t - np.abs(pop)) - r2*(worst_t - np.abs(pop))
    new_pop   = np.clip(new_pop, 0.05, 0.95)
    new_costs = np.array([fusion_cost(t, val_probs_np, y_va) for t in new_pop])
    improved  = new_costs < costs
    pop       = np.where(improved, new_pop, pop)
    costs     = np.where(improved, new_costs, costs)

OPTIMAL_THRESHOLD = float(pop[np.argmin(costs)])
print(f"  Optimal threshold : {OPTIMAL_THRESHOLD:.4f}")
print(f"  Val cost @ optimal: {fusion_cost(OPTIMAL_THRESHOLD, val_probs_np, y_va):.4f}")

HIGH_THRESHOLD     = min(OPTIMAL_THRESHOLD + 0.15, 0.90)
ELEVATED_THRESHOLD = max(OPTIMAL_THRESHOLD - 0.10, 0.25)
print(f"  HIGH threshold    : {HIGH_THRESHOLD:.4f}")
print(f"  ELEVATED threshold: {ELEVATED_THRESHOLD:.4f}")

# %% Cell 9: Test set evaluation
print("\n[NB08] Test set evaluation …")

X_te_t = torch.FloatTensor(X_te).to(DEVICE)

with torch.no_grad():
    test_logits = raw_logits(model, X_te_t)
    test_probs  = torch.sigmoid(test_logits).cpu().numpy()
    attn_final  = torch.softmax(model.feature_attn, dim=0).cpu().numpy()

test_preds = (test_probs >= OPTIMAL_THRESHOLD).astype(int)

auc  = roc_auc_score(y_te, test_probs)
mcc  = matthews_corrcoef(y_te, test_preds)
prec = precision_score(y_te, test_preds, zero_division=0)
rec  = recall_score(y_te, test_preds, zero_division=0)
f1   = f1_score(y_te, test_preds, zero_division=0)
cm   = confusion_matrix(y_te, test_preds)
fp_val = int(cm[0, 1]) if cm.shape == (2,2) else 0
fn_val = int(cm[1, 0]) if cm.shape == (2,2) else 0
tp_val = int(cm[1, 1]) if cm.shape == (2,2) else 0
tn_val = int(cm[0, 0]) if cm.shape == (2,2) else 0

print(f"  AUC-ROC   : {auc:.4f}")
print(f"  MCC       : {mcc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  TP={tp_val}  FP={fp_val}  TN={tn_val}  FN={fn_val}")

print(f"\n  Learned attention weights (feature importance):")
for i, name in enumerate(FEATURE_NAMES):
    bar = "█" * int(attn_final[i] * 60)
    print(f"    {name:<25}  {attn_final[i]:.4f}  {bar}")

# %% Cell 10: ROC curve plot
fpr_arr, tpr_arr, _ = roc_curve(y_te, test_probs)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(fpr_arr, tpr_arr, color="#0099cc", lw=2, label=f"FusionNet AUC={auc:.4f}")
axes[0].plot([0,1], [0,1], "k--", lw=0.8)
axes[0].axvline(x=0.15, color="gray", ls=":", alpha=0.5, label="15% FPR target")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("FusionNet ROC Curve"); axes[0].legend()

axes[1].bar(FEATURE_NAMES, attn_final, color="#00d4ff", edgecolor="#0099cc")
axes[1].set_title("Learned Feature Attention Weights")
axes[1].set_ylabel("Weight"); axes[1].tick_params(axis="x", rotation=45)
axes[1].set_ylim(0, None)

plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "fusion_roc.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[NB08] ROC plot saved → {roc_path}")

# Training curves
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(train_losses, label="train loss", color="#0099cc")
ax2.plot(val_losses,   label="val loss",   color="#ff5e57")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("BCE Loss")
ax2.set_title("FusionNet Training Curves"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fusion_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# %% Cell 11: Save model
model_path = os.path.join(MODEL_DIR, "fusion_net.pt")
torch.save(model.state_dict(), model_path)
print(f"[NB08] fusion_net.pt saved → {model_path}")

# %% Cell 12: Save fusion_config.json
config = {
    "input_dim":            INPUT_DIM,
    "fusion_threshold":     round(OPTIMAL_THRESHOLD, 6),
    "high_threshold":       round(HIGH_THRESHOLD, 6),
    "elevated_threshold":   round(ELEVATED_THRESHOLD, 6),
    "fusion_auc":           round(float(auc), 6),
    "fusion_mcc":           round(float(mcc), 6),
    "fusion_precision":     round(float(prec), 6),
    "fusion_recall":        round(float(rec), 6),
    "fusion_f1":            round(float(f1), 6),
    "true_positives":       tp_val,
    "false_positives":      fp_val,
    "true_negatives":       tn_val,
    "false_negatives":      fn_val,
    "feature_names":        FEATURE_NAMES,
    "optimal_attn_weights": [round(float(w), 6) for w in attn_final],
    "pos_weight_used":      round(float(pos_weight.item()), 4),
    "training_epochs":      len(train_losses),
    "best_val_loss":        round(float(best_val_loss), 6),
}

config_path = os.path.join(RESULTS_DIR, "fusion_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[NB08] fusion_config.json saved → {config_path}")

print("\n[NB08] ✓ Phase 2 model training complete!")
print(f"  AUC={auc:.4f}  MCC={mcc:.4f}  threshold={OPTIMAL_THRESHOLD:.4f}")
print(f"\n[NB08] NEXT STEP → add Phase 2 endpoints to app.py")