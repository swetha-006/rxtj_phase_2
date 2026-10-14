# ============================================================
# preprocessing.py — shared imputer + scaler transforms
# Used by Phase 1 scoring API (app.py) and Phase 2 profile builder
# so both stages apply the *same* transforms used at training time.
# ============================================================

import os
import hashlib
import joblib
import numpy as np

PREPROCESSOR_HASHES = {
    "imputer.pkl": "ac9b0886ce63a173778b33db0e78410b71d682ee1997f6f9ffbe36d536ed8003",
    "scaler.pkl": "d985ef87722ea7f091eb15f2c5d4a2637346765105858cc38e595961e7f4d291",
}


def verify_file_integrity(filepath: str, expected_hash: str) -> None:
    """Verify SHA-256 checksum of an artifact before deserializing (Issue 12 fix)."""
    if os.environ.get("GUARDIAN_VERIFY_MODEL_HASH", "1").lower() in ("0", "false", "no"):
        return
    if not os.path.exists(filepath):
        return
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_hash:
        raise RuntimeError(
            f"Preprocess artifact integrity check failed for '{os.path.basename(filepath)}'! "
            f"Expected SHA-256: {expected_hash}, got: {actual}. Possible artifact tampering detected."
        )


def load_preprocessors(base_dir):
    """Load the training-time SimpleImputer + StandardScaler from disk with integrity check."""
    imp_path = os.path.join(base_dir, 'models', 'imputer.pkl')
    scl_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    verify_file_integrity(imp_path, PREPROCESSOR_HASHES["imputer.pkl"])
    verify_file_integrity(scl_path, PREPROCESSOR_HASHES["scaler.pkl"])

    imputer = joblib.load(imp_path)
    if not hasattr(imputer, '_fill_dtype'):
        imputer._fill_dtype = np.float32
    scaler  = joblib.load(scl_path)
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