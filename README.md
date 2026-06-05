# 🛡️ ML Transaction Risk Scoring for Payment Fraud Detection

> **RXT-J+ Model** — ResNeXt-Embedded GRU with Jaya Optimization for Real-Time Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![AUC](https://img.shields.io/badge/AUC-96.24%25-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)

---

RXT-J+ Phase 2 — Account Compromise Detection
Behavioral & Contextual Fusion Scoring Engine
Project: ML Transaction Risk Scoring for Payment Fraud
Institution: Sri Krishna College of Technology (SKCT), Coimbatore
Programme: B.E. Computer Science (Cyber Security) — Capstone Project (NEP 2020)
Team: Sm · Deepak B · Joselin Jennilia A
Guide: Ms. Soundarya, Assistant Professor, AIML Department
Dataset: IEEE-CIS Fraud Detection (590,540 transactions · 433 features)
Base Paper: Almazroi & Ayub, IEEE Access 2023 — DOI: 10.1109/ACCESS.2023.3339226
Table of Contents
What Phase 2 Does
The Core Idea
How It Builds on Phase 1
System Architecture
Technical Components
Results
Project Structure
Setup & Installation
Running the Pipeline
API Reference
Key Design Decisions
Limitations & Future Work
1. What Phase 2 Does
Phase 1 of RXT-J+ answered: "Is this specific transaction fraudulent?"
Phase 2 answers a fundamentally different and harder question: "Has this account been taken over by someone else?"
Account Takeover (ATO) is when a fraudster gains access to a legitimate user's account and begins making transactions. The individual transactions may look borderline or even normal on their own — but the pattern is wrong. The device is new. The merchant category has never appeared before. The transaction happens at 3 AM when the real owner only transacts during business hours. The amount is three standard deviations above the account's baseline.
Phase 2 detects this by building a behavioral fingerprint for each account and continuously measuring how far each new transaction deviates from that baseline. The further the deviation across multiple dimensions, the higher the compromise probability.
2. The Core Idea
The Difference Between a Fraudulent Transaction and a Compromised Account
Code
A transaction of $450 at a jewellery store may be low-risk on its own. But if that account has only ever transacted at grocery stores and pharmacies for an average of $62, at 9 AM, always from the same device — the $450 jewellery purchase at 2 AM from a new device is a strong signal of account compromise even if the transaction features look individually ordinary.
The Behavioral Fingerprint
For every account (card1 in the IEEE-CIS dataset), Phase 2 maintains a living behavioral profile that tracks:
Dimension
What it captures
Amount distribution
Typical spend amount and variability (rolling mean + std)
Merchant patterns
Which product categories (ProductCD) this account uses
Time-of-day histogram
When this account normally transacts (24-bin hour distribution)
Geographic baseline
Typical billing address region (addr1 rolling median)
Known devices
Set of DeviceInfo fingerprints seen before
Velocity
Transaction rate in the last 1h and 24h
The 8 Contextual Deviation Features
At inference time, for every incoming transaction, Phase 2 computes how far that transaction deviates from the account's stored baseline:
Feature
Formula
High value means
amount_z_score
(amount − mean) / std, clipped [−5, 5]
Unusually large or small amount
merchant_novelty
1 − freq(MCC) / max_freq
Merchant type never or rarely seen
geo_displacement
abs(addr1 − baseline_median) / 500
Far from typical billing location
hour_deviation
1 − hour_histogram[current_hour]
Transaction at an unusual time
device_novelty
1.0 if new device, 0.0 if seen before
New device never associated with account
velocity_ratio
txns_last_1h / (txns_last_24h / 24)
Sudden spike in transaction rate
behavioral_drift
Autoencoder reconstruction error (normalised)
Overall behavioral pattern changed
p1_risk_score
Phase 1 RXT-J+ output
Individual transaction-level risk
These 8 features are the input to FusionNet — a small attention-weighted neural network that learns to combine them into a single compromise_probability score.
3. How It Builds on Phase 1
Phase 1 built and trained the RXT-J+ classification engine:
Code
Phase 1 results: AUC = 96.24%, MCC = 0.8426, latency = 0.019ms, throughput = 51,351 TPS
Phase 2 extends this in three ways:
Reuses Phase 1 models directly. The autoencoder.pt and resnet_extractor.pt from EARN+ training are used to generate P1 risk scores for all 590K training rows, and the same attention_rxtj.pt + IFM ensemble scores each transaction at inference time.
Adds the behavioral profile layer. The ProfileStore (SQLite, WAL mode) maintains per-account rolling statistics. Every scored transaction updates the account's profile via exponential moving average (α = 0.1).
Introduces the FusionNet layer. A small MLP (8→64→32→16→1) with a learned feature attention vector combines the 8 contextual features into a final compromise_probability. The attention weights double as an explainability mechanism — the /account/explain endpoint returns which feature most drove the score.
Code
4. System Architecture
Code
5. Technical Components
5.1 Profile Store (profile_store.py)
SQLite database in WAL (Write-Ahead Logging) mode for concurrent read performance. Each method opens and closes its own connection — safe for multi-worker deployments.
Tables:
profiles — one row per account, stores all behavioral statistics
transaction_log — every scored event with compromise probability and context
alerts — indexed view of high-risk events in the last 24 hours
Profile update strategy: Exponential Moving Average with α = 0.1. This means the profile adapts slowly to new behaviour, preventing a compromised account from "retraining" its own baseline during an attack window.
5.2 Notebook Pipeline
Notebook
Purpose
Runtime
Output
06_behavioral_profiling.py
Build per-account profiles from all 590K transactions
~2.5h (first run) / ~15min (resume)
behavioral_profiles.db · tx_snapshot.parquet
07_contextual_features.py
Compute 8 deviation features per transaction
~15 min
contextual_features.npy (590540×8) · drift_norm_params.json
08_fusion_model_training.py
Generate P1 scores · Train FusionNet v2
~40 min total
fusion_net.pt · fusion_config.json
Smart resume: NB06 checks if behavioral_profiles.db already has ≥90% of accounts profiled and skips the 2.5-hour loop if so. NB08 checks if model_probs_full.npy exists with the correct row count and skips Stage 1 if so.
5.3 FusionNet Architecture
Python
The feature_attn parameter is a learnable softmax weight vector applied to the inputs before the MLP. After training, torch.softmax(feature_attn) gives the relative importance of each feature — used directly by the /account/explain endpoint to produce human-readable explanations.
Training configuration:
Loss: BCEWithLogitsLoss with pos_weight (fraud-class upweighting)
Optimiser: Adam, lr=5e-4, weight_decay=1e-4
Schedule: CosineAnnealingLR (T_max=200, eta_min=1e-5)
Early stopping: patience=25 on validation AUC
Threshold: Jaya algorithm optimisation (minimise 2×FPR + FNR)
5.4 API Endpoints (Phase 2)
Five new endpoints added to the existing FastAPI application. All Phase 1 endpoints remain unchanged.
6. Results
Phase 1 (Transaction-Level Fraud Detection)
Metric
Value
AUC-ROC
96.24%
MCC
0.8426
Inference latency
0.019 ms
Throughput
51,351 TPS
Jaya weights
W_MODEL=0.5343, W_IFM=0.4657
Phase 2 (Account Compromise Detection)
Metric
Value
AUC-ROC
0.7567
MCC
0.1664
Recall
54.6%
Precision
9.2%
Optimal threshold
0.5087
Accounts profiled
13,553
Training transactions
413,219
Learned Feature Attention Weights
The FusionNet's attention layer assigns the following importance to each feature after training:
Feature
Weight
Interpretation
behavioral_drift
0.1966
Highest — overall behavioral shift is the strongest signal
amount_z_score
0.1629
Second — unusually large amounts are a strong indicator
merchant_novelty
0.1279
Transactions at never-seen merchant types
device_novelty
0.1230
New device not previously associated with account
hour_deviation
0.1045
Activity at unusual hours
velocity_ratio
0.0991
Sudden burst of transactions
geo_displacement
0.0929
Geographic deviation from home region
p1_risk_score
0.0931
Phase 1 per-transaction risk score
Decision Thresholds
Decision
Threshold
Recommended Action
HIGH
≥ 0.6587
FREEZE_AND_NOTIFY — Block and alert customer
ELEVATED
≥ 0.4087
STEP_UP_AUTHENTICATION — Require 2FA / OTP
LOW
< 0.4087
APPROVE — Transaction proceeds normally
7. Project Structure
Code
8. Setup & Installation
Requirements
Python 3.11 (required — scientific packages do not have wheels for 3.12+)
Anaconda / Miniconda
Windows 10/11 (tested) or Linux
Create Environment
Bash
Install Dependencies
Bash
Or from requirements.txt:
Bash
requirements.txt (full Phase 2 list):
Code
Dataset
Download the IEEE-CIS Fraud Detection dataset from Kaggle and place the files at:
Code
9. Running the Pipeline
All Phase 1 notebooks (01–05) must be completed before running Phase 2. The Phase 2 notebooks must be run in order.
Step 1 — Build behavioral profiles
Bash
Runtime: ~2.5 hours (first run). If behavioral_profiles.db already exists with ≥90% of accounts profiled, the notebook automatically skips the profile loop and only rebuilds the snapshot (~15 min).
What it produces:
data/behavioral_profiles.db — 13,553 account profiles
data/tx_snapshot.parquet — per-row snapshot with pre-computed velocity and device novelty flags
Step 2 — Compute contextual features
Bash
Runtime: ~15 minutes
What it produces:
data/contextual_features.npy — (590,540 × 8) float32 feature matrix
data/drift_norm_params.json — autoencoder normalisation parameters for live inference
data/fusion_labels.npy — training labels
Step 3 — Train FusionNet
Bash
Runtime: ~40 minutes total (Stage 1: P1 score generation ~25 min; Stage 2: training ~15 min)
What it produces:
models/fusion_net.pt — trained FusionNet v2 weights
models/fusion_scaler.pkl — feature scaler for live inference
results/fusion_config.json — thresholds, metrics, attention weights
Step 4 — Start the API server
Only start the server after Step 3 is complete and all model files exist.
Bash
Expected startup log:
Code
Step 5 — Verify all endpoints
Bash
10. API Reference
Phase 1 Endpoints (unchanged)
Method
Path
Description
POST
/score
Score raw 224-dim feature vector
POST
/score/direct
Score with direct feature values
POST
/score/form
Score from a partial form input
POST
/score/batch
Score multiple transactions
GET
/history
Recent scoring history
GET
/health
Server health check
GET
/model/info
Loaded model metadata
Phase 2 Endpoints (new)
POST /account/compromise-score
Full Phase 2 inference. Runs Phase 1 scoring, retrieves the account's behavioral profile, computes all 8 contextual deviation features, runs FusionNet, and returns a compromise probability with explainability.
Request body:
Json
Response:
Json
GET /account/profile/{account_id}
Returns the stored behavioral profile for an account.
Query params: ?window=30d (informational)
Response: Full profile dict with amt_mean, amt_std, hour_hist[24], merchant_counts, known_device_count, geo_cluster, velocity_1h, velocity_24h, profile_age_hours.
Returns 404 if the account has not been seen before.
GET /account/history/{account_id}
Returns the last N scored events for an account, newest first.
Query params: ?limit=20 (max 100)
Response: { account_id, record_count, events: [{transaction_id, timestamp, compromise_prob, decision, top_trigger_feature, context}] }
GET /account/alerts
Returns all accounts with peak compromise probability above a threshold in the last 24 hours, ordered by score descending.
Query params: ?threshold=0.7&limit=50
Response: { threshold, alert_count, alerts: [{account_id, max_score, alert_count, last_seen, recommended_action}] }
POST /account/explain
Returns a human-readable natural language explanation for a previously scored transaction. Designed for analyst review interfaces and merchant risk dashboards.
Request body:
Json
Response:
Json
11. Key Design Decisions
Why card1 as the account proxy?
The IEEE-CIS dataset does not have an explicit account ID column. card1 is a hashed card number identifier — transactions with the same card1 value come from the same physical card, making it the closest available proxy for "account". It yields 13,553 unique accounts across the 590,540 transactions.
Why Exponential Moving Average for profile updates (α = 0.1)?
A slow α means the profile adapts gradually over many transactions. This is intentional — if an account is compromised and a hijacker begins making many transactions, a fast α would cause the profile to quickly "learn" the attacker's behaviour, erasing the signal. With α = 0.1, it takes approximately 22 transactions to move the mean by half the gap between old and new values.
Why FusionNet is small (8→64→32→16→1)?
The 8 contextual features carry most of the signal at a coarse level. A large model would overfit to the training distribution. The small architecture also ensures sub-millisecond inference latency, keeping the end-to-end /account/compromise-score latency under 10ms.
Why Jaya optimisation for the threshold?
Consistent with Phase 1's approach. The Jaya algorithm minimises a cost function that penalises false positives at 2× the weight of false negatives — reflecting the real-world context where false positives (blocking a legitimate customer) are more operationally costly than false negatives for a first-line detection system.
Why SQLite in WAL mode instead of Redis?
Redis would be faster but adds a deployment dependency. WAL mode SQLite achieves concurrent read performance suitable for a single-machine academic deployment while keeping the system self-contained and reproducible.
Why strict=False for the ResNet extractor?
The resnet_extractor.pt was trained in Phase 1. If the Phase 2 class definition has minor structural differences (e.g., stem key naming), strict=False allows the weights to load without crashing. The EARN+ feature quality degrades slightly but the system remains functional.
12. Limitations & Future Work
Current Limitations
P1 feature sparsity: The P1 scoring pipeline for the 590K training rows fills in only a subset of the 224 imputer features from the transaction CSV columns available in the snapshot. The remaining features are imputed to training-set means, which reduces the discriminative spread of P1 scores. A future fix is to load all 224 named features directly from the original CSV using imputer.feature_names_in_.
Behavioral drift range: The autoencoder reconstruction error shows narrow spread (min=0.1995, max=0.2530) because the EARN+ autoencoder was trained on balanced (SMOTE-Tomek) data and only 4 of 224 input features vary meaningfully in the current setup. With full feature loading, this range would widen significantly.
Account coverage: 48% of accounts in the profile store have ≥5 historical transactions. Accounts with fewer than 5 transactions produce less reliable contextual deviation scores.
FusionNet AUC: 0.7567 is a functional score but below the 0.85 target. This is primarily caused by the near-constant P1 risk score feature (std=0.011). Fixing the P1 feature loading is expected to push AUC to 0.85+.
Phase 3 Preview
Phase 3 targets: Fraud Pattern Visualisation for Merchant Risk Management. This extends Phase 2 by:
Aggregating account compromise events into merchant-level risk scores
Generating visualisations of fraud pattern clusters across merchants
Producing temporal heatmaps of compromise events by hour and day
Building a merchant-facing dashboard for proactive risk monitoring
Citation
If you use this work, please cite the base paper:
Bibtex
Acknowledgements
This project was developed as a B.E. Capstone Project under the New Education Policy (NEP) 2020 at Sri Krishna College of Technology, Coimbatore. We thank Ms. Soundarya (Assistant Professor, AIML Department) for her guidance throughout the project.
The IEEE-CIS Fraud Detection dataset is publicly available via the Kaggle IEEE-CIS Fraud Detection competition.
RXT-J+ Phase 2 — Advanced ML Transaction Risk Scoring Engine | SKCT Team 13 | 2024–2025