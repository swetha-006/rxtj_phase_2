# ============================================================
# preprocessing.py — shared imputer + scaler transforms
# Used by Phase 1 scoring API (app.py) and Phase 2 profile builder
# so both stages apply the *same* transforms used at training time.
# ============================================================

import os
import joblib
import numpy as np


def load_preprocessors(base_dir):
    """Load the training-time SimpleImputer + StandardScaler from disk."""
    imputer = joblib.load(os.path.join(base_dir, 'models', 'imputer.pkl'))
    scaler  = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
    return imputer, scaler


def raw_feature_dim(imputer):
    """Expected raw feature dimension (matches what imputer was fit on)."""
    return int(imputer.statistics_.shape[0])


def transform_raw(features, imputer, scaler):
    """Apply imputer.transform + scaler.transform to raw features.

    Accepts a 1-D vector or a 2-D matrix. Always returns a float32 2-D array
    shaped (n_samples, n_features) ready for Nystroem.transform.
    """
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X.astype(np.float32)


def pad_form_features(values, target_dim):
    """Pad a partial form input (e.g. amount, is_foreign, age) to the full
    raw feature dimension expected by the imputer/scaler. Missing positions
    are filled with NaN so the imputer fills them with training-set means."""
    out = np.full(target_dim, np.nan, dtype=np.float32)
    n   = min(len(values), target_dim)
    out[:n] = np.asarray(values[:n], dtype=np.float32)
    return out