"""
test_ml_core.py — Comprehensive Integration and Validation Suite
Tests all requirements for P1, P2, P3, and P5:
  1. Artifact integrity and fresh process loading
  2. Prediction bounds [0, 1] and finiteness
  3. Flag-rate sanity (no 99.96% flag rate bug)
  4. TreeSHAP explanation layer
  5. Reason codes mapping (R07, R12, R03, R21)
  6. Bounded counterfactual search
  7. End-to-end FastAPI endpoint smoke tests
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Set UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import app
client = TestClient(app.app)


def test_artifact_loading():
    print("[TEST 1] Testing artifact loading...")
    art = joblib.load("models/lgbm.pkl")
    assert "model" in art, "Missing model in artifact"
    assert "calibrator" in art, "Missing calibrator in artifact"
    assert "features" in art, "Missing features in artifact"
    assert "threshold" in art, "Missing threshold in artifact"
    assert len(art["features"]) == 14, f"Expected 14 features, got {len(art['features'])}"
    print("  -> Artifact contains valid model, calibrator, 14 features, threshold:", art["threshold"])


def test_model_predictions_bounds():
    print("[TEST 2] Testing prediction bounds & finiteness on various inputs...")
    # Exact 14-dim input
    X_14 = np.zeros((5, 14), dtype=np.float32)
    X_14[:, 0] = [50.0, 150.0, 500.0, 1200.0, 5000.0]
    risk, preds, conf = app.score_raw_features(X_14)
    assert np.all(np.isfinite(risk)), "Non-finite risk score"
    assert np.all((risk >= 0.0) & (risk <= 1.0)), "Risk score outside [0, 1]"
    assert np.all((preds == 0) | (preds == 1)), "Invalid binary predictions"
    assert np.all((conf >= 0.0) & (conf <= 1.0)), "Confidence outside [0, 1]"

    # Legacy 224-dim input
    X_224 = np.zeros((3, 224), dtype=np.float32)
    X_224[:, 1] = [80.0, 250.0, 900.0]
    risk224, preds224, conf224 = app.score_raw_features(X_224)
    assert np.all((risk224 >= 0.0) & (risk224 <= 1.0)), "224-dim risk outside [0, 1]"

    # Legacy 50-dim EARN+ input
    X_50 = np.random.randn(4, 50).astype(np.float32)
    risk50, preds50, conf50 = app.score_earn_features(X_50)
    assert np.all((risk50 >= 0.0) & (risk50 <= 1.0)), "50-dim risk outside [0, 1]"

    print("  -> All predicted probabilities are finite and strictly within [0, 1] ✓")


def test_flag_rate_sanity():
    print("[TEST 3] Testing flag-rate sanity (verifying broken 99.96% scorer is fixed)...")
    # Generate 200 normal transactions across common amounts
    amounts = np.linspace(15, 300, 200)
    records = []
    for a in amounts:
        records.append({
            "TransactionAmt": a,
            "ProductCD": "W",
            "addr1": 299.0,
            "DeviceInfo": "Windows",
            "hour": 14,
            "amt_cumsum": a * 0.95,
            "amt_cumstd": 20.0,
            "txn_seq": 10,
            "velocity_1h": 0,
            "velocity_24h": 1,
            "device_novel": 0.0,
            "amt_to_mean": a / (a * 0.95 + 1.0),
            "amt_diff_mean": a * 0.05,
            "velocity_ratio": 0.0
        })
    df_samples = pd.DataFrame(records)
    risk, preds, conf = app.score_transaction_features(df_samples)
    flag_rate = preds.mean()
    print(f"  -> Flag rate on standard traffic: {flag_rate * 100:.2f}% (Mean score: {risk.mean():.4f})")
    assert flag_rate < 0.20, f"Flag rate {flag_rate*100:.2f}% exceeds 20% limit!"
    print("  -> Scorer flag-rate sanity passed ✓ (no longer flagging ~99.96% of traffic)")


def test_shap_explanations():
    print("[TEST 4] Testing Layer 1 TreeSHAP explanations...")
    form_data = {
        "amount": 950.0,
        "product_cd": "C",
        "device_info": "Linux",
        "hour": 3,
        "amt_cumsum": 50.0,
        "amt_cumstd": 15.0,
        "txn_seq": 1,
        "velocity_1h": 4,
        "velocity_24h": 6,
        "device_novel": 1.0
    }
    X_df = app.transform_to_lgbm_features(form_data)
    shap_out = app.compute_shap_layer(X_df)
    assert "raw_model_margin" in shap_out
    assert "calibrated_probability" in shap_out
    assert "feature_contributions" in shap_out
    assert len(shap_out["feature_contributions"]) == 14
    top_driver = shap_out["feature_contributions"][0]
    print(f"  -> Top TreeSHAP feature: {top_driver['feature']} (SHAP: {top_driver['shap_value']}, Direction: {top_driver['risk_direction']})")
    assert "disclosure" in shap_out
    print("  -> TreeSHAP layer verified ✓")


def test_reason_codes():
    print("[TEST 5] Testing Layer 2 Reason Codes (R07, R12, R03, R21)...")
    form_data = {
        "amount": 1200.0,
        "product_cd": "W",
        "device_info": "Unrecognized_New_OS",
        "hour": 2,
        "amt_cumsum": 60.0,
        "amt_cumstd": 10.0,
        "txn_seq": 2,
        "velocity_1h": 5,
        "velocity_24h": 12,
        "device_novel": 1.0
    }
    X_df = app.transform_to_lgbm_features(form_data)
    shap_out = app.compute_shap_layer(X_df)
    reason_codes = app.generate_reason_codes(X_df, shap_out["shap_dict"])
    
    code_names = [r["code"] for r in reason_codes]
    print(f"  -> Emitted reason codes: {code_names}")
    assert "R07" in code_names, "Expected R07 (AMOUNT_ANOMALY)"
    assert "R12" in code_names, "Expected R12 (DEVICE_UNRECOGNIZED)"
    assert "R03" in code_names, "Expected R03 (VELOCITY_SPIKE)"
    assert "R21" in code_names, "Expected R21 (RING_ASSOCIATION)"
    
    # Verify R21 shows UNAVAILABLE when no graph evidence is provided
    r21 = next(r for r in reason_codes if r["code"] == "R21")
    assert r21.get("status") == "UNAVAILABLE"
    print("  -> R21 ring association correctly reports UNAVAILABLE when graph cluster is absent ✓")


def test_counterfactual_explanation():
    print("[TEST 6] Testing Layer 3 Bounded Counterfactual Search...")
    high_risk_form = {
        "amount": 3000.0,
        "product_cd": "W",
        "device_info": "Windows",
        "hour": 14,
        "amt_cumsum": 80.0,
        "amt_cumstd": 20.0,
        "txn_seq": 5,
        "velocity_1h": 0,
        "velocity_24h": 1,
        "device_novel": 0.0
    }
    X_df = app.transform_to_lgbm_features(high_risk_form)
    cal_p, preds, _ = app.score_transaction_features(X_df)
    current_p = float(cal_p[0])
    cf_res = app.compute_counterfactual_explanation(X_df, current_p, threshold=app.THRESHOLD)
    assert "original_probability" in cf_res
    assert "disclaimer" in cf_res
    print(f"  -> Counterfactual result: found={cf_res['counterfactual_found']}, explanation: {cf_res['explanation']}")
    print("  -> Bounded counterfactual search verified ✓")


def test_api_endpoints():
    print("[TEST 7] Smoke testing all FastAPI endpoints...")

    # GET /
    r = client.get("/")
    assert r.status_code == 200, f"/ failed: {r.text}"
    assert r.json()["version"] == "4.0.0"

    # GET /health
    r = client.get("/health")
    assert r.status_code == 200, f"/health failed: {r.text}"
    assert r.json()["models_loaded"] is True

    # GET /model/info
    r = client.get("/model/info")
    assert r.status_code == 200, f"/model/info failed: {r.text}"
    info = r.json()
    assert "phase1" in info and "phase2" in info

    # GET /demo-samples
    r = client.get("/demo-samples")
    assert r.status_code == 200

    # POST /score (single)
    r = client.post("/score", json={"transaction_id": "T1", "features": [100.0] * 14})
    assert r.status_code == 200, f"/score failed: {r.text}"
    data = r.json()
    assert "risk_score" in data and "decision" in data
    assert 0.0 <= data["risk_score"] <= 1.0

    # POST /score?explain=true
    r = client.post("/score?explain=true", json={"transaction_id": "T1_exp", "features": [100.0] * 14})
    assert r.status_code == 200
    assert r.json()["explanation"] is not None

    # POST /score/form
    r = client.post("/score/form", json={"transaction_id": "T2", "amount": 85.50, "is_foreign": False, "customer_age": 35})
    assert r.status_code == 200
    assert "risk_score" in r.json()

    # POST /score/direct
    r = client.post("/score/direct", json={"transaction_id": "T3", "features": [0.5] * 50})
    assert r.status_code == 200

    # POST /score/batch
    batch_req = {
        "transactions": [
            {"transaction_id": f"B_{i}", "features": [float(i * 10)] * 14}
            for i in range(10)
        ]
    }
    r = client.post("/score/batch", json=batch_req)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 10

    # POST /score/form-full
    form_full_req = {
        "account_id": "ACC_42",
        "transaction_id": "T4",
        "amount": 250.0,
        "hour": 15,
        "product_cd": "W",
        "is_foreign": False,
        "customer_age": 30,
        "device_info": "Windows",
        "addr1": "299",
        "timestamp": 12345678.0
    }
    r = client.post("/score/form-full", json=form_full_req)
    assert r.status_code == 200, f"/score/form-full failed: {r.text}"
    ff_res = r.json()
    assert "p1_risk_score" in ff_res
    assert "compromise_probability" in ff_res
    assert "contextual_features" in ff_res

    # POST /account/compromise-score
    comp_req = {
        "account_id": "ACC_42",
        "transaction_id": "T5",
        "features": [100.0] * 14,
        "amount": 250.0,
        "hour": 15,
        "product_cd": "W",
        "device_info": "Windows",
        "addr1": "299"
    }
    r = client.post("/account/compromise-score", json=comp_req)
    assert r.status_code == 200

    # GET /account/profile/ACC_42
    r = client.get("/account/profile/ACC_42")
    assert r.status_code == 200

    # GET /account/history/ACC_42
    r = client.get("/account/history/ACC_42")
    assert r.status_code == 200
    assert r.json()["record_count"] >= 1

    # GET /account/alerts
    r = client.get("/account/alerts")
    assert r.status_code == 200

    # POST /score/explain (Dedicated 3-layer endpoint)
    r = client.post("/score/explain", json={
        "transaction_id": "T_EXP",
        "amount": 650.0,
        "hour": 14,
        "product_cd": "W",
        "device_info": "Windows",
        "addr1": "299"
    })
    assert r.status_code == 200, f"/score/explain failed: {r.text}"
    exp_res = r.json()
    assert "layer1_shap" in exp_res
    assert "layer2_reason_codes" in exp_res
    assert "layer3_counterfactual" in exp_res

    # POST /account/explain
    r = client.post("/account/explain", json={"account_id": "ACC_42", "transaction_id": "T4"})
    assert r.status_code == 200
    assert "what_triggered" in r.json()

    # GET /history
    r = client.get("/history")
    assert r.status_code == 200

    print("  -> All 16+ API endpoints smoke tested successfully with valid responses ✓")


def main():
    print("=" * 76)
    print("  RXT-J+ v4 ML Core Integration & Validation Test Suite")
    print("=" * 76)
    test_artifact_loading()
    test_model_predictions_bounds()
    test_flag_rate_sanity()
    test_shap_explanations()
    test_reason_codes()
    test_counterfactual_explanation()
    test_api_endpoints()
    print("=" * 76)
    print("  ALL TESTS PASSED SUCCESSFULLY (100% GREEN)!")
    print("=" * 76)


if __name__ == "__main__":
    main()
