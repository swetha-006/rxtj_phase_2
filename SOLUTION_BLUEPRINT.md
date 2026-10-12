# RXT-J+ v4 "Glass Box"
## Real-Time Digital Fraud Detection — Hackathon Solution Blueprint

**Problem statement #6:** *Develop a system that generates fraud-risk scores using transaction amount, frequency, location, device, timing, and transaction history, while explaining alerts and minimizing false positives.*

**Document status:** Every number marked **[MEASURED]** was produced by running code against this repository's own data on 2026-09-18. Every number marked **[PROJECTED]** is an estimate that has **not** been verified and must be measured before it appears on a slide. This separation is deliberate and is itself part of the pitch — see §5.1.

---

## 0. The Pitch (30 seconds)

> Most fraud systems are graded on AUC. Banks don't pay AUC — they pay **fraud losses plus analyst hours plus annoyed customers**. RXT-J+ v4 optimizes those three directly.
>
> We know the difference matters because **our own v3 model lied to us**: it reported 0.96 AUC while flagging 99.96% of real traffic as fraud. We caught it, we can show you exactly how, and the audit harness that caught it now ships as a build gate.
>
> v4 scores every transaction in under a millisecond, routes it by **expected rupee cost** rather than a flat threshold, explains every alert in regulator-ready reason codes, and surfaces **fraud rings** — the accounts and devices moving together that per-transaction scoring structurally cannot see.

**The three-line differentiator for judges:**
1. **Cost-aware routing** — a ₹200 transaction and a ₹80,000 transaction do not deserve the same threshold. Nobody else in the room will have this.
2. **Glass-box honesty** — we ship the harness that failed our own previous model. Intellectual honesty is rare and memorable.
3. **Ring detection** — fraud is a network. Per-transaction models are blind to it by construction.

---

## 1. Strict Evaluator's Scorecard — The Build As It Stands Today

Grading the current repository against the six named requirements, with no charity.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Risk score from **amount** | Partial | `TransactionAmt`, `amt_cumsum`, `amt_cumstd` present and causal |
| 2 | Risk score from **frequency** | Partial | `velocity_1h`, `velocity_24h`, `txn_seq` present |
| 3 | Risk score from **location** | Partial | `addr1` only (332 distinct), **11.1% null** |
| 4 | Risk score from **device** | **Weak** | `DeviceInfo` is **79.9% null** — device signal absent on 4 of 5 transactions |
| 5 | Risk score from **timing** | Partial | `hour`, `TransactionDT` present |
| 6 | Risk score from **transaction history** | **Good** | `ProfileStore` EMA profiles — genuinely solid, see §2 |
| 7 | **Explaining alerts** | **Broken** | Ranks features by raw magnitude, not contribution — see §3.3 |
| 8 | **Minimizing false positives** | **Failing** | 11,112 false positives; precision 11.9% |

### 1.1 The finding that reframes everything

`results/deployment_config.json` reports **AUC 0.9624, MCC 0.8426**. That is measured on a set of 82,484 rows containing 41,242 positives — **exactly 50.0% fraud**, the SMOTE-balanced output of NB02. Production fraud rate is 3.499%.

When that same Phase 1 chain is run against the real distribution, NB08's own stdout records what it actually does:

```
P1: min=0.2887  mean=0.7121  max=0.7449  std=0.0114
Fraud rate @ threshold: 99.96%
```

**It flags 99.96% of all transactions as fraud.** The 0.96 is not a model result; it is an artifact of grading on a balanced set. If a judge asks "what is your AUC at production class balance," the honest answer today is **0.7406**, with **precision 11.9%** and **FPR 13.0%**.

### 1.2 Root cause, confirmed

NB08's Stage 1 guard deletes a stale P1 score file when `std <= 0.05`, regenerates it — and produces `std = 0.0114` again. It never re-validates after regenerating. Consequence: `p1_risk_score`, which should be the single strongest feature, carries a label correlation of **+0.0102** — second-weakest of the 13.

The Jaya-optimized attention weights confirm the model found nothing to work with: 12 of 13 features land between **0.0665 and 0.0818**. Near-uniform weights are what "no feature carries signal" looks like numerically.

**Verdict:** the failure is confined to the scorer. The product around it is sound. **Optimize — do not rebuild.**

---

## 2. KEEP — Assets That Already Work

Roughly 70% of what a judge actually touches is built and functioning. Rebuilding this would be self-sabotage.

| Asset | Why it survives | Evidence |
|---|---|---|
| **FastAPI service** (`app.py`) | 16 working routes, Pydantic-validated, real-time | Routes verified at `app.py:592–1080` |
| **`ProfileStore`** (`profile_store.py`) | EMA behavioral profiles (α=0.1), WAL-mode SQLite, transaction log, alert query, coverage stats. **This is requirement #6, already delivered.** | `profile_store.py:9,12,52–231` |
| **Causal feature engineering** | `amt_cumsum` is the expanding mean of *strictly prior* transactions; `txn_seq` increments by exactly 1 per card. **Leak-free.** | Verified 2026-09-18, §6.1 |
| **Dashboards** (4 HTML) | `panel.html` alone is 42KB of working UI — the thing judges look at | Present in root |
| **Decision tiering** (`decide_action`) | Right *shape* (APPROVE / MONITOR / STEP_UP / FREEZE), wrong *logic* — upgrade, don't discard | `app.py:584–590` |
| **Labeled dataset** | 590,540 rows, 13,553 accounts, 20,663 frauds, temporal column for honest splits | `data/tx_snapshot.parquet` |
| **Clean swap seam** | Entire scorer sits behind `score_raw_features()` / `score_earn_features()` with **4 call sites** | `app.py:388,435,673,695,764,821` |

> The swap seam is the whole argument for optimizing. The broken model is one function behind a stable interface, not a tangle through 16 endpoints.

---

## 3. KILL — What Cannot Be Fixed By Tuning

### 3.1 The Phase 1 deep chain — **delete entirely**

`Autoencoder → ResNet extractor → Nyström → IncrementalPCA → AttentionRXTJ → Jaya-weighted blend with IsolationForest`

Six transforms that compress 224 features into a scalar with **std 0.0114**. This is not a hyperparameter problem — the chain destroys signal by construction. No learning rate fixes a constant.

**Deleting this also disposes of:** NB03's re-runnability defect (`Z_train_concat` and `kpca` are used but never assigned anywhere in the notebook), plus 5 of 11 model artifacts. Technical debt removed for free.

### 3.2 Balanced-set evaluation — **delete**

NB02's SMOTE output must never again be used for reporting metrics. Resampling is a *training* technique; evaluating on it manufactures the 0.96 illusion. All reported metrics come from a temporal split at production class balance, permanently.

### 3.3 The current explanation logic — **delete**

```python
sorted_feats = sorted(ctx.items(), key=lambda x: abs(float(x[1])), reverse=True)[:3]
```
*(`app.py:1115`)*

This ranks the **largest normalized feature value**, not the feature that drove the decision. A feature sitting at 0.9 that the model ignores outranks one at 0.3 that caused the block. It produces fluent sentences with no causal relationship to the score — the worst kind of explanation, because it is confidently wrong. Replace with TreeSHAP (§5.4).

### 3.4 Artifacts retired

`autoencoder.pt`, `resnet_extractor.pt`, `attention_rxtj.pt`, `nystroem.pkl`, `incremental_pca.pkl`, `kernel_pca.pkl`, `jaya_optimal_weights.pkl`, `fusion_net.pt`, `fusion_scaler.pkl` — superseded. Keep `imputer.pkl` / `scaler.pkl` only if the raw-feature path is retained.

---

## 4. CHANGE — Fixes Required Regardless of Ambition

| # | Issue | Fix | Effort |
|---|---|---|---|
| C1 | Repo is `root:root` — blocks every write | `sudo chown -R vishu:vishu .` | 2 min |
| C2 | `profile_consumer.py` is a Kafka stub never wired to `ProfileStore` | Wire `handle_event` → `update_profile` + `log_transaction`, or cut it from the demo | 1 h |
| C3 | Hardcoded Windows paths (`F:\rxtj_project\...`, `f:\rxtj_phase_2\...` in `profile_stats.json`) | `pathlib` throughout | 30 min |
| C4 | `imbalanced-learn` used by NB01/NB02, absent from `requirements.txt` | Add, or remove with §3.2 | 5 min |
| C5 | **Zero tests, zero CI** | pytest on scoring + audit gate in GitHub Actions | 2 h |
| C6 | `README.md` describes v2/8-feature architecture; reality is v3/13-feature. References non-existent `tests/`, `Dockerfile`, `.env.example`, `fix_p1_scores.py` | Rewrite against reality | 1 h |
| C7 | No container | `Dockerfile` + `docker-compose.yml` | 1 h |
| C8 | Inconsistent thresholds between `deployment_config.json` (0.50) and `fusion_config.json` (0.357/0.507/0.257) | Single source of truth, §5.3 | 30 min |

---

## 5. THE SOLUTION — RXT-J+ v4 Architecture

```
                         ┌──────────────────────────────────────────┐
   Transaction  ────────▶│  1. FEATURE ASSEMBLY                     │
   (JSON / Kafka)        │     amount · frequency · location        │
                         │     device · timing · history            │
                         │     ProfileStore EMA lookup  (<1ms)      │
                         └────────────────┬─────────────────────────┘
                                          ▼
                         ┌──────────────────────────────────────────┐
                         │  2. RING GRAPH ENRICHMENT      [NOVEL]   │
                         │     card ↔ device ↔ address components   │
                         │     → ring_size, ring_velocity           │
                         └────────────────┬─────────────────────────┘
                                          ▼
                         ┌──────────────────────────────────────────┐
                         │  3. GRADIENT-BOOSTED SCORER              │
                         │     LightGBM · temporal split            │
                         │     → calibrated P(fraud)                │
                         └────────────────┬─────────────────────────┘
                                          ▼
                         ┌──────────────────────────────────────────┐
                         │  4. COST-SENSITIVE ROUTER      [NOVEL]   │
                         │     argmin of expected rupee cost        │
                         │     ALLOW / STEP-UP / BLOCK              │
                         └────────────────┬─────────────────────────┘
                                          ▼
                         ┌──────────────────────────────────────────┐
                         │  5. EXPLANATION LAYER                    │
                         │     TreeSHAP → reason codes              │
                         │     + counterfactual ("approved if ≤ ₹X")│
                         └────────────────┬─────────────────────────┘
                                          ▼
                         ┌──────────────────────────────────────────┐
                         │  6. GLASS-BOX AUDIT GATE       [NOVEL]   │
                         │     blocks deploy on leak / imbalance /  │
                         │     degenerate-feature / flag-rate fail  │
                         └──────────────────────────────────────────┘
```

### 5.1 Differentiator #1 — The Glass-Box Audit Gate

**What.** A `validate_model.py` that a model must pass before it can be deployed. It fails the build on:

| Check | Trigger | Why it exists |
|---|---|---|
| Class-balance integrity | eval set fraud rate deviates >20% from production rate | Catches the 0.9624-on-balanced-data illusion |
| Degenerate feature | any feature `std < 0.05` | Catches the exact `p1_risk_score` bug |
| Label leakage | any feature `\|corr\| > 0.95` with label | Standard leak guard |
| Temporal integrity | split is not chronological | Prevents future→past leakage |
| Flag-rate sanity | model flags >20% of traffic at deploy threshold | Catches the 99.96% failure |
| Causality audit | per-entity derived features must use only prior rows | Protects the feature engineering we verified clean |

**Why it wins.** Every other team presents a number. You present a number **plus the adversarial harness that tried to break it**. Then you run the harness against v3 live on stage and watch it fail with `FLAG RATE 99.96% — REJECTED`. That is the moment judges remember.

**Implementation.** ~150 lines, pure numpy/pandas. Wire into CI so it gates every commit.
**Effort:** 3–4 h.

### 5.2 Differentiator #2 — Cost-Sensitive Decision Router

**The insight.** A flat threshold treats a ₹200 coffee and an ₹80,000 transfer identically. That is the single largest avoidable source of false positives — you block thousands of small transactions to catch a few large frauds.

**The math.** For each transaction, compute expected cost of each action and take the minimum:

```
cost(ALLOW)    = p · amount
cost(STEP_UP)  = C_step + p · (1 − γ) · amount
cost(BLOCK)    = C_block · (1 − p)

where  p       = calibrated fraud probability
       amount  = transaction amount
       γ       = fraction of fraud that step-up authentication defeats
       C_step  = friction cost of a step-up challenge
       C_block = cost of blocking a legitimate customer
```

The `(1 − p)` on the block term is deliberate: successfully blocking fraud has near-zero net cost, so that term prices only the legitimate-customer case — the false positive you are trying to avoid.

**Worked example** (illustrative parameters `C_step=₹20`, `C_block=₹400`, `γ=0.85`):

| Transaction | p | ALLOW | STEP_UP | BLOCK | **Decision** |
|---|---|---|---|---|---|
| ₹200 | 0.30 | ₹60 | ₹29 | ₹280 | **STEP_UP** |
| ₹200 | 0.70 | ₹140 | ₹41 | ₹120 | **STEP_UP** |
| ₹80,000 | 0.30 | ₹24,000 | ₹3,620 | ₹280 | **BLOCK** |
| ₹80,000 | 0.08 | ₹6,400 | ₹980 | ₹368 | **BLOCK** |

Note the fourth row: an 8% fraud probability **blocks** when the amount is large, while a 70% probability **does not block** when the amount is small. No single threshold can express that. This is a direct, mathematically-grounded attack on false positives — and it is ~30 lines of code.

**Calibration matters here.** `p` must be a real probability, not a raw score, or the expected-cost arithmetic is meaningless. Use isotonic regression / `CalibratedClassifierCV` on the validation split. State this on the slide — it shows rigor.

**Effort:** 3–4 h including calibration.

### 5.3 Differentiator #3 — Alert-Budget Optimizer

**The reframe.** "Minimizing false positives" is not a threshold question; it is a capacity question. A bank has N analysts who can review K alerts per day. The real question: *given that budget, how much fraud do we catch?*

**[MEASURED]** — from the baseline in §6, on the held-out temporal test set:

| Alert budget | Alerts issued | Precision | Recall |
|---|---|---|---|
| top 0.1% | 88 | **81.8%** | 2.3% |
| top 0.5% | 442 | **52.5%** | 7.5% |
| top 1.0% | 885 | **40.9%** | 11.7% |
| top 2.0% | 1,771 | 31.0% | 17.8% |
| top 5.0% | 4,429 | 22.8% | 32.7% |

**The demo.** A slider in the dashboard: drag "analyst capacity," watch precision, recall, expected fraud caught (₹) and expected loss prevented (₹) update live. Judges immediately understand the tradeoff, and it speaks the operator's language rather than the modeler's.

**Effort:** 2 h (the numbers already exist; this is presentation).

### 5.4 Differentiator #4 — Reason Codes + Counterfactuals

**Replaces the broken §3.3 logic.**

**Layer 1 — TreeSHAP.** Per-transaction, exact, and fast enough for real time on a tree model. Gives *actual contribution*, signed, in score units.

**Layer 2 — Reason codes.** Map the top contributors to fixed, regulator-style codes (the adverse-action pattern used in credit decisioning):

```
R07  AMOUNT_ANOMALY       Amount is 8.4× this account's 30-day average
R12  DEVICE_UNRECOGNIZED  First transaction from this device
R03  VELOCITY_SPIKE       12 transactions in 1h vs. account norm of 0.4
R21  RING_ASSOCIATION     Device shared with 9 other accounts in 2h
```

**Layer 3 — Counterfactual.** *"This transaction would have been APPROVED if the amount were ≤ ₹4,200."* Computed by binary-searching the feature against the model until the routing decision flips. This is the line that makes judges sit up — it converts a score into something a human can act on and a customer can be told.

**Effort:** 4 h (SHAP 1h, codes 1h, counterfactual 2h).

### 5.5 Differentiator #5 — Fraud Ring Detection

**The insight.** Per-transaction scoring is structurally blind to coordinated fraud. Three accounts each making one unremarkable transaction from one shared device is invisible per-transaction and obvious as a graph.

**Implementation.** Build a bipartite graph over `card1 ↔ DeviceInfo` and `card1 ↔ addr1` within a rolling window; take connected components; emit:
- `ring_size` — accounts in this component
- `ring_new_account_rate` — fraction of component accounts under N days old
- `ring_velocity` — component transactions per hour

`networkx.connected_components` over a pandas groupby. No GPU, no deep learning.

**Honest constraint, state it on the slide:** `DeviceInfo` is **79.9% null**, so device-edge ring detection covers ~20% of transactions. Address edges cover ~89%. Present it as "ring detection where device telemetry exists" rather than implying universal coverage — and note that in a real deployment device fingerprinting is a solved ingestion problem, not a modeling one.

**[PROJECTED]** — ring features should add meaningful lift given `card1` already dominates importance at 37.8%, but **this is an untested hypothesis. Measure it before it goes on a slide.**

**Effort:** 3–4 h.

### 5.6 Differentiator #6 — Live Adaptive Profiles

**Already built** (`ProfileStore`, EMA α=0.1) — this just needs to be *shown*.

**The demo.** Submit the same ₹15,000 transaction twice: once against a fresh account (flagged — no baseline, high anomaly) and once against an account whose EMA profile shows a ₹14,000 average (approved). Same input, different decision, because the system learned the account's normal in between. Thirty seconds, zero slides, completely convincing.

**Effort:** 1 h (scripting the demo).

---

## 6. Measured Baseline — Proof The Swap Works

**[MEASURED 2026-09-18]** LightGBM trained on `data/tx_snapshot.parquet` — data already on disk. 70/15/15 temporal split by `TransactionDT`, ground-truth `isFraud` label, frequency encodings fit on train only. Test set n=88,581 at 3.48% fraud.

```
ROC-AUC : 0.7892
PR-AUC  : 0.2072
best_iteration: 577
```

Against the current system's own reported figures (**AUC 0.7406, PR-AUC 0.1028**), PR-AUC roughly **doubles**.

> **Comparability caveat — state this honestly.** FusionNet v3 is trained on an `account_compromise_window_24h` label (23,400 positives, 3.96%); the baseline above uses raw `isFraud` (3.48%). These are different prediction targets, so treat them as two standalone results, not a controlled head-to-head. The direction is not in doubt — a 2× PR-AUC gap does not reverse from a 13% prevalence difference — but do not present it as a clean A/B.

**What matters:** a ~30-line script beat eight notebooks and eleven model artifacts. That is the proof that the scorer is a swappable component rather than a reason to rebuild.

### 6.1 Leak audit (run before trusting any of the above)

| Check | Result |
|---|---|
| `txn_seq` increments by exactly 1 within `card1` | **100.00%** |
| `amt_cumsum` = expanding mean of **strictly prior** transactions | Confirmed on traced sample |
| First transaction per card: `amt_cumstd == 0` | **100.00%** |
| Frequency encodings fit on train split only | Yes, by construction |

The derived features are causally correct. This is a genuine strength of the existing work and should be said out loud.

### 6.2 Cold-start — the question you *will* be asked

| Cohort | n | Fraud rate | AUC |
|---|---|---|---|
| Overall | 88,581 | 3.48% | 0.7892 |
| `card1` **seen** in training | 87,371 | 3.48% | 0.7891 |
| `card1` **unseen** (new account) | 1,210 | 3.80% (46 frauds) | **0.8182** |

98.63% of test transactions come from a previously-seen card. On genuinely new accounts the model does **not** degrade — it scores slightly better. The lift is not account memorization. *(Small subsample — 46 frauds — so the interval is wide; present as "does not degrade," not as "performs better.")*

### 6.3 Where the signal actually lives

| Feature | Gain |
|---|---|
| `card1` | 37.82% |
| `DeviceInfo` | 9.27% |
| `addr1` | 9.25% |
| `TransactionAmt` | 6.60% |
| `txn_seq` | 6.21% |
| `card1_freq` | 5.56% |
| `velocity_24h` | 2.78% |
| `hour` | 2.35% |
| `velocity_1h` | 0.71% |
| `device_novel` | **0.22%** |

Note the last row: `device_novel`, a hand-engineered feature the current pipeline relies on, contributes **0.22%**. Several of the 13 hand-built contextual features are close to worthless. Let the tree model find its own interactions.

### 6.4 The largest single available win — recover the raw features

`data/tx_snapshot.parquet` contains **15 columns**. NB08 reads `data/IEEE CIS/train_transaction.csv`, which **is not in the repository**. That CSV holds 224 features — `V1–V339`, `C1–C14`, `D1–D15`, `card2–card6`, email domains, `dist1/2` — and on this dataset that is where the signal lives.

**Action:** re-download from the Kaggle *IEEE-CIS Fraud Detection* competition (requires Kaggle auth, ~650MB) and retrain on the full feature set.

**[PROJECTED]** This is the highest-leverage remaining action by a wide margin. Do **not** put a predicted AUC on a slide — train it, then report what you measured.

---

## 7. Build Plan

### Full path (~22–28 h)

| Phase | Work | Effort | Unlocks |
|---|---|---|---|
| **P0** | `chown` repo; rebuild venv; pin deps | 0.5 h | Everything |
| **P1** | Glass-box audit harness (§5.1) | 3–4 h | The narrative |
| **P2** | LightGBM + calibration; artifacts to `models/` | 3 h | Real scores |
| **P3** | Swap scorer behind `score_raw_features()`; delete dead chain | 2–3 h | Working API |
| **P4** | Cost-sensitive router (§5.2) | 3–4 h | FP reduction |
| **P5** | SHAP + reason codes + counterfactual (§5.4) | 4 h | Explainability |
| **P6** | Ring graph (§5.5) | 3–4 h | Wow factor |
| **P7** | Dashboard: budget slider, reason codes, ring view | 4 h | The demo |
| **P8** | Tests, CI, Dockerfile, README rewrite | 3 h | Polish |

### Minimum viable path (~8–10 h) — if time collapses

**P0 → P2 → P3 → P4 → P5(SHAP + reason codes only) → demo script.**

This alone delivers: honest metrics, a working real-time scorer, cost-aware FP reduction, and real explanations. Ring detection and the audit harness are the differentiators you drop last, not first — but a working honest system beats an ambitious broken one.

### Sequencing rule

**P1 before P2.** Build the audit harness *before* the new model, so the new model's first act is passing a gate it could have failed. If you build the model first you will be tempted to tune the gate around it, and the entire honesty narrative collapses.

---

## 8. Demo Script (5 minutes)

| Time | Beat | What judges see |
|---|---|---|
| 0:00 | **The confession** | "Our previous model reported 0.96 AUC. Here is it flagging 99.96% of transactions as fraud." Show the NB08 output. |
| 0:45 | **The gate** | Run `validate_model.py` against v3. `REJECTED: FLAG RATE 99.96%`. Run against v4. `PASSED`. |
| 1:30 | **Live scoring** | POST a transaction. Sub-millisecond score, reason codes, decision. |
| 2:15 | **Cost routing** | Same fraud probability, two amounts — ₹200 steps up, ₹80,000 blocks. Explain the expected-cost math. |
| 3:00 | **Counterfactual** | *"Approved if amount ≤ ₹4,200."* |
| 3:30 | **Adaptive profile** | Same transaction, fresh account vs. established account → different decisions. |
| 4:00 | **Ring** | Graph view: 9 accounts, 1 device, 2 hours. |
| 4:30 | **Budget slider** | "You have 4 analysts reviewing 450 alerts/day → 52.5% precision, ₹X prevented." |

Open with the confession. It is counterintuitive and it is the strongest card in the deck — it buys credibility for every number that follows.

---

## 9. Honest Limitations — Put These On A Slide

1. **AUC 0.7892 is modest in absolute terms.** Cause is known and stated: 15 columns instead of 224 (§6.4). We are not hiding it behind a balanced-set number.
2. **Device coverage is 79.9% null.** Ring detection via device edges covers ~20% of traffic. This is a data-ingestion gap, not a modeling failure.
3. **Cold-start evidence is thin** — 46 frauds in the unseen-card cohort. Directionally reassuring, not conclusive.
4. **Ring-feature lift is unmeasured.** Hypothesis, flagged as such, to be measured before claiming.
5. **Cost parameters (`C_step`, `C_block`, `γ`) are assumptions.** In production they come from the bank's own chargeback and churn data. Present the framework as parameterized, not the constants as truth.
6. **Single dataset, single time window.** No cross-institution validation.

---

## 10. Requirement Traceability

| Requirement | v3 (today) | v4 (target) | Mechanism |
|---|---|---|---|
| Amount | Partial | **Full** | Raw + expanding mean/std + `amt_z_hist` |
| Frequency | Partial | **Full** | `velocity_1h/24h`, `txn_seq`, ring velocity |
| Location | Partial | **Full** | `addr1` + frequency encoding + address ring edges |
| Device | Weak | **Partial (honest)** | `DeviceInfo` + device ring edges; 79.9% null disclosed |
| Timing | Partial | **Full** | `hour`, inter-transaction gap, ring time windows |
| Transaction history | Good | **Full** | `ProfileStore` EMA + causal expanding features |
| **Explaining alerts** | **Broken** | **Full** | TreeSHAP → reason codes → counterfactual |
| **Minimizing false positives** | **Failing** | **Full** | Cost-sensitive router + calibration + alert-budget optimizer |

---

## Appendix A — Evidence Log

Every claim above traces to one of these. All verified 2026-09-18.

| Claim | Source |
|---|---|
| AUC 0.9624 on 50.0%-positive set | `results/deployment_config.json` — TP 37,489 + FN 3,753 = 41,242 of 82,484 |
| P1 std 0.0114, flags 99.96% | NB08 cell 1 stdout |
| AUC 0.7406, precision 0.1193, FPR 0.1300, FP 11,112 | `results/fusion_config.json` |
| `p1_risk_score` corr +0.0102 | NB08 Stage 2 stdout |
| Jaya weights 0.0665–0.1366 | `results/fusion_config.json` |
| 590,540 rows · 13,553 accounts · 20,663 frauds (3.499%) | `data/profile_stats.json` |
| Parquet has 15 columns | Parquet footer schema |
| `DeviceInfo` 79.9% null · `addr1` 11.1% null | Direct computation |
| LightGBM AUC 0.7892 / PR-AUC 0.2072 | `scratchpad/lgbm_results.json` |
| Alert-budget table | `scratchpad/train_gbdt.py` output |
| Cold-start 98.63% seen · unseen AUC 0.8182 | `scratchpad/coldstart.py` output |
| Causality of `amt_cumsum` / `txn_seq` | `scratchpad/leak2.py` output |
| Explanation ranks by magnitude | `app.py:1115` |
| Scorer seam, 4 call sites | `app.py:388, 435, 673, 695, 764, 821` |
| No tests / CI / Dockerfile / `.env.example` | Filesystem check |
| IEEE-CIS CSV absent | `find data -iname "*.csv"` → empty |

## Appendix B — Reproducing The Baseline

```bash
python -m venv venv && ./venv/bin/pip install pandas pyarrow lightgbm scikit-learn
./venv/bin/python train_gbdt.py    # temporal split, isFraud, train-only encodings
./venv/bin/python leak2.py         # causality audit
./venv/bin/python coldstart.py     # unseen-account generalization
```

Scripts are in the session scratchpad and should be committed to `scripts/` once the repository is writable.
