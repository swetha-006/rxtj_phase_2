# ============================================================
# save_demo_samples.py
# Run once in Jupyter or VS Code (conda rxtj env)
# Extracts real fraud and legit samples from your test set
# and saves them as JSON for the web demo
# ============================================================

import numpy as np
import json
import os

# Load EARN+ features and labels from Notebook 3 output
# These are 50-dim post-IPCA vectors
Z_test  = np.load('../data/Z_test_earn.npy')
y_test  = np.load('../data/y_test.npy').astype(int)

print(f"Test set shape : {Z_test.shape}")
print(f"Fraud samples  : {y_test.sum():,}")
print(f"Legit samples  : {(y_test==0).sum():,}")

# Pick 5 clear fraud and 5 clear legit samples
fraud_idx = np.where(y_test == 1)[0]
legit_idx = np.where(y_test == 0)[0]

samples = {
    "fraud_1":  Z_test[fraud_idx[0]].tolist(),
    "fraud_2":  Z_test[fraud_idx[10]].tolist(),
    "fraud_3":  Z_test[fraud_idx[50]].tolist(),
    "legit_1":  Z_test[legit_idx[0]].tolist(),
    "legit_2":  Z_test[legit_idx[10]].tolist(),
    "legit_3":  Z_test[legit_idx[50]].tolist(),
    "edge_1":   Z_test[fraud_idx[200]].tolist(),  # borderline
    "edge_2":   Z_test[legit_idx[200]].tolist(),  # borderline
}

os.makedirs('../data', exist_ok=True)
with open('../data/demo_samples.json', 'w') as f:
    json.dump(samples, f)

print("\nSaved to data/demo_samples.json")
print("Sample dimensions:", len(samples['fraud_1']), "features each")
print("\nSample keys:", list(samples.keys()))