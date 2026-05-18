"""
verify_phase1_fixes.py
Run from F:\\rxtj_project\\ with:  python verify_phase1_fixes.py
Checks all 6 Phase 1 fixes without starting the server.
"""

import os, sys, json, importlib, inspect, ast
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))

PASS  = "\033[92m  [PASS]\033[0m"
FAIL  = "\033[91m  [FAIL]\033[0m"
WARN  = "\033[93m  [WARN]\033[0m"
SKIP  = "\033[90m  [SKIP]\033[0m"
HEAD  = "\033[1;36m"
RESET = "\033[0m"

results = []

def check(label, passed, detail="", warn=False):
    symbol = WARN if warn else (PASS if passed else FAIL)
    status = "WARN" if warn else ("PASS" if passed else "FAIL")
    print(f"{symbol} {label}")
    if detail:
        print(f"        → {detail}")
    results.append((status, label))

def section(title):
    print(f"\n{HEAD}{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}{RESET}")

# ─────────────────────────────────────────────────────────────
section("FIX 1  ·  imputer & scaler loaded in load_models()")
# ─────────────────────────────────────────────────────────────

app_path = os.path.join(BASE, "app.py")

try:
    with open(app_path) as f:
        app_src = f.read()

    has_imputer_load  = "imputer.pkl" in app_src and \
                        ("joblib.load" in app_src or "load(" in app_src)
    has_scaler_load   = "scaler.pkl"  in app_src and \
                        ("joblib.load" in app_src or "load(" in app_src)

    # check they appear inside load_models function body
    tree = ast.parse(app_src)
    load_models_src = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_models":
            load_models_src = ast.get_source_segment(app_src, node) or ""

    imputer_in_fn = "imputer.pkl" in load_models_src
    scaler_in_fn  = "scaler.pkl"  in load_models_src

    check("imputer.pkl loaded inside load_models()", imputer_in_fn,
          "Not found inside load_models() body" if not imputer_in_fn else "")
    check("scaler.pkl loaded inside load_models()",  scaler_in_fn,
          "Not found inside load_models() body"  if not scaler_in_fn else "")

    # check they're applied in score_raw_features
    score_raw_src = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "score_raw_features":
            score_raw_src = ast.get_source_segment(app_src, node) or ""

    imputer_applied = "imputer.transform" in score_raw_src
    scaler_applied  = "scaler.transform"  in score_raw_src

    check("imputer.transform() called in score_raw_features()", imputer_applied,
          "Raw features go to nystroem without imputation" if not imputer_applied else "")
    check("scaler.transform() called in score_raw_features()",  scaler_applied,
          "Raw features go to nystroem without scaling" if not scaler_applied else "")

    # Verify order: imputer → scaler → nystroem
    if imputer_applied and scaler_applied:
        imp_pos = score_raw_src.find("imputer.transform")
        scl_pos = score_raw_src.find("scaler.transform")
        nys_pos = score_raw_src.find("nystroem.transform")
        order_ok = imp_pos < scl_pos < nys_pos
        check("Transform order is imputer → scaler → nystroem", order_ok,
              f"Positions: imputer={imp_pos}, scaler={scl_pos}, nystroem={nys_pos}" if not order_ok else "")

except Exception as e:
    check("app.py parse", False, str(e))

# ─────────────────────────────────────────────────────────────
section("FIX 2  ·  preprocessing.py is populated")
# ─────────────────────────────────────────────────────────────

preproc_path = os.path.join(BASE, "preprocessing.py")

try:
    size = os.path.getsize(preproc_path)
    check("preprocessing.py exists and is not empty", size > 0,
          f"File is {size} bytes" if size == 0 else f"{size} bytes")

    with open(preproc_path) as f:
        pp_src = f.read()

    has_preprocess_raw    = "preprocess_raw" in pp_src
    has_load_transformers = "load_transformers" in pp_src

    check("preprocess_raw() function defined", has_preprocess_raw,
          "Add: def preprocess_raw(X_raw) -> np.ndarray")
    check("load_transformers() function defined", has_load_transformers,
          "Add: def load_transformers() -> (imputer, scaler, nystroem, ipca)")

except FileNotFoundError:
    check("preprocessing.py exists", False, "File not found")

# ─────────────────────────────────────────────────────────────
section("FIX 3  ·  AutoencoderNet defined & autoencoder.pt loaded")
# ─────────────────────────────────────────────────────────────

try:
    ae_class_defined = "AutoencoderNet" in app_src or "Autoencoder" in app_src
    ae_pt_loaded     = "autoencoder.pt" in app_src
    ae_in_load_fn    = "autoencoder.pt" in load_models_src or \
                       "autoencoder" in load_models_src

    check("AutoencoderNet class defined in app.py", ae_class_defined,
          "Add AutoencoderNet(nn.Module) class above load_models()")
    check("autoencoder.pt referenced in app.py", ae_pt_loaded)
    check("autoencoder loaded inside load_models()", ae_in_load_fn,
          "autoencoder.pt must be loaded at startup, not inside endpoints")

    ae_path = os.path.join(BASE, "models", "autoencoder.pt")
    check("models/autoencoder.pt file exists on disk", os.path.exists(ae_path),
          f"Expected at: {ae_path}")

except Exception as e:
    check("Autoencoder check", False, str(e))

# ─────────────────────────────────────────────────────────────
section("FIX 4  ·  backend.py merged / deleted")
# ─────────────────────────────────────────────────────────────

backend_path = os.path.join(BASE, "backend.py")
backend_exists = os.path.exists(backend_path)

if backend_exists:
    with open(backend_path) as f:
        be_src = f.read()
    be_size = os.path.getsize(backend_path)
    check("backend.py removed or emptied", be_size == 0,
          f"backend.py still has {be_size} bytes — merge endpoints into app.py and delete", warn=(be_size > 0))
else:
    check("backend.py removed", True, "File deleted — good")

# Check that /score/form and /history exist in app.py
has_score_form = '"/score/form"' in app_src or "'/score/form'" in app_src or "score/form" in app_src
has_history    = '"/history"'    in app_src or "'/history'"    in app_src

check("/score/form endpoint exists in app.py", has_score_form,
      "Migrate /score/form from backend.py into app.py")
check("/history endpoint exists in app.py", has_history,
      "Migrate /history from backend.py into app.py")

# ─────────────────────────────────────────────────────────────
section("FIX 5  ·  README endpoint table is accurate")
# ─────────────────────────────────────────────────────────────

readme_path = os.path.join(BASE, "README.md")
try:
    with open(readme_path) as f:
        readme = f.read()

    ghost_endpoints = ["/predict", "/predict_batch", "/demo", "/stats"]
    real_endpoints  = ["/score", "/score/batch", "/score/direct", "/health", "/model/info"]

    for ep in ghost_endpoints:
        present = ep in readme
        check(f"Ghost endpoint '{ep}' removed from README", not present,
              f"README still documents '{ep}' which doesn't exist in app.py", warn=present)

    for ep in real_endpoints:
        present = ep in readme
        check(f"Real endpoint '{ep}' documented in README", present,
              f"Add '{ep}' to the README endpoint table", warn=not present)

except FileNotFoundError:
    check("README.md exists", False, "File not found")

# ─────────────────────────────────────────────────────────────
section("FIX 6  ·  Kafka producer stub in /score endpoint")
# ─────────────────────────────────────────────────────────────

has_kafka_import  = "kafka" in app_src.lower() and ("KafkaProducer" in app_src or "producer" in app_src.lower())
has_try_except    = "KafkaProducer" in app_src and "except" in app_src

check("KafkaProducer imported or stubbed in app.py", has_kafka_import,
      "Add a KafkaProducer stub inside /score endpoint (non-blocking, wrapped in try/except)",
      warn=not has_kafka_import)
check("Kafka call wrapped in try/except (fail-silent)", has_try_except,
      "Kafka failures must NOT break the scoring response", warn=not has_try_except)

# ─────────────────────────────────────────────────────────────
section("LIVE LOAD TEST  ·  Actually import app.py and load models")
# ─────────────────────────────────────────────────────────────

print(f"\n  Attempting to import app.py and run load_models()...")
print(f"  (This will take 5–15 seconds as models load into memory)\n")

try:
    sys.path.insert(0, BASE)
    # Import without triggering uvicorn
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", app_path)
    app_mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(app_mod)

    # Check globals exist
    check("model loaded (AttentionRXTJ)", hasattr(app_mod, 'model') and app_mod.model is not None)
    check("nystroem loaded",  hasattr(app_mod, 'nystroem')  and app_mod.nystroem is not None)
    check("ipca loaded",      hasattr(app_mod, 'ipca')       and app_mod.ipca is not None)
    check("ifm loaded",       hasattr(app_mod, 'ifm')        and app_mod.ifm is not None)
    check("imputer loaded",   hasattr(app_mod, 'imputer')    and app_mod.imputer is not None,
          "FIX 1 not applied — imputer not a module-level variable")
    check("scaler loaded",    hasattr(app_mod, 'scaler')     and app_mod.scaler is not None,
          "FIX 1 not applied — scaler not a module-level variable")
    check("autoencoder loaded", hasattr(app_mod, 'autoencoder') and app_mod.autoencoder is not None,
          "FIX 3 not applied — autoencoder not loaded at startup")
    check("config loaded",    hasattr(app_mod, 'config')     and app_mod.config is not None)

    # Quick score test
    import numpy as np
    try:
        dummy = np.zeros((1, 128), dtype=np.float32)
        imputer_out = app_mod.imputer.transform(dummy)
        scaler_out  = app_mod.scaler.transform(imputer_out)
        nys_out     = app_mod.nystroem.transform(scaler_out)
        ipca_out    = app_mod.ipca.transform(nys_out)
        risk, preds, conf = app_mod.score_earn_features(ipca_out)
        check("Full pipeline: dummy → imputer → scaler → nystroem → ipca → model → risk score",
              True, f"risk={risk[0]:.4f}, pred={preds[0]}, conf={conf[0]:.4f}")
    except Exception as e:
        check("Full pipeline end-to-end run", False, str(e))

except Exception as e:
    check("app.py import / load_models() execution", False,
          f"{type(e).__name__}: {e}")
    print()
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
section("SUMMARY")
# ─────────────────────────────────────────────────────────────

total  = len(results)
passed = sum(1 for s, _ in results if s == "PASS")
warned = sum(1 for s, _ in results if s == "WARN")
failed = sum(1 for s, _ in results if s == "FAIL")

print(f"\n  Total checks : {total}")
print(f"\033[92m  Passed       : {passed}\033[0m")
print(f"\033[93m  Warnings     : {warned}\033[0m")
print(f"\033[91m  Failed       : {failed}\033[0m")

if failed == 0 and warned == 0:
    print(f"\n\033[92m  ✓ All Phase 1 fixes verified. Ready to build Phase 2.\033[0m\n")
elif failed == 0:
    print(f"\n\033[93m  ⚠  No failures but {warned} warnings. Review before Phase 2.\033[0m\n")
else:
    print(f"\n\033[91m  ✗  {failed} check(s) failed. Fix before starting Phase 2.\033[0m\n")
    print("  Failed checks:")
    for s, label in results:
        if s == "FAIL":
            print(f"    • {label}")
    print()