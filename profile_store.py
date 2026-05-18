# ============================================================
# profile_store.py — Phase 2 Per-Account Behavioral Profile Store
# Place in: F:\rxtj_project\profile_store.py
# ============================================================

import os, sqlite3, json, time
from typing import Optional

_EMA_ALPHA = 0.1   # slow, stable adaptation

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS profiles (
    account_id      TEXT    PRIMARY KEY,
    txn_count       INTEGER DEFAULT 0,
    amt_mean        REAL    DEFAULT 0.0,
    amt_std         REAL    DEFAULT 0.0,
    amt_mean_7d     REAL    DEFAULT 0.0,
    amt_std_7d      REAL    DEFAULT 0.0,
    amt_min         REAL    DEFAULT 0.0,
    amt_max         REAL    DEFAULT 0.0,
    hour_hist       TEXT    DEFAULT '[]',
    merchant_counts TEXT    DEFAULT '{}',
    known_devices   TEXT    DEFAULT '[]',
    geo_cluster     TEXT    DEFAULT '',
    velocity_1h     INTEGER DEFAULT 0,
    velocity_24h    INTEGER DEFAULT 0,
    last_txn_dt     REAL    DEFAULT 0.0,
    last_updated    REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS transaction_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id           TEXT,
    transaction_id       TEXT,
    timestamp            REAL,
    p1_risk_score        REAL,
    compromise_prob      REAL DEFAULT NULL,
    decision             TEXT,
    top_trigger_feature  TEXT DEFAULT '',
    context_json         TEXT DEFAULT '{}',
    logged_at            REAL
);

CREATE INDEX IF NOT EXISTS idx_txlog_account ON transaction_log(account_id);
CREATE INDEX IF NOT EXISTS idx_txlog_ts      ON transaction_log(timestamp);
"""


class ProfileStore:
    """SQLite-backed per-account behavioral profile store (WAL mode).

    Each method opens and closes its own connection — safe for
    concurrent uvicorn workers.
    """

    def __init__(self, db_path: str = "data/behavioral_profiles.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        with self._conn() as c:
            c.executescript(_DDL)

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(row) -> dict:
        d = dict(row)
        d["hour_hist"]       = json.loads(d.get("hour_hist")       or "[]")
        d["merchant_counts"] = json.loads(d.get("merchant_counts") or "{}")
        d["known_devices"]   = json.loads(d.get("known_devices")   or "[]")
        return d

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_profile(self, account_id: str) -> Optional[dict]:
        """Return stored profile dict or None if account is unknown."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM profiles WHERE account_id=?", (str(account_id),)
            ).fetchone()
        return self._parse(row) if row else None

    def get_or_empty(self, account_id: str) -> dict:
        """Return profile or a safe zero-profile for new accounts."""
        p = self.get_profile(account_id)
        if p:
            return p
        return {
            "account_id": account_id, "txn_count": 0,
            "amt_mean": 0.0, "amt_std": 1.0,
            "amt_mean_7d": 0.0, "amt_std_7d": 1.0,
            "amt_min": 0.0, "amt_max": 0.0,
            "hour_hist": [1/24]*24, "merchant_counts": {},
            "known_devices": [], "geo_cluster": "",
            "velocity_1h": 0, "velocity_24h": 0,
            "last_txn_dt": 0.0, "last_updated": 0.0,
        }

    def update_profile(self, account_id: str, txn: dict) -> None:
        """Incrementally update a profile with one new transaction (EMA blend).

        txn keys: amount (float), hour (int 0-23), product_cd (str),
                  device_info (str|None), addr1 (str|None), txn_dt (float).
        """
        aid    = str(account_id)
        now    = time.time()
        amount = float(txn.get("amount", 0.0))
        hour   = int(txn.get("hour", 0)) % 24
        pcd    = str(txn.get("product_cd", ""))
        dev    = str(txn.get("device_info", "")) if txn.get("device_info") else ""
        addr   = str(txn.get("addr1", ""))
        dt     = float(txn.get("txn_dt", now))
        α      = _EMA_ALPHA

        ex = self.get_profile(aid)

        if ex is None:
            hh = [0.0]*24; hh[hour] = 1.0
            with self._conn() as c:
                c.execute(
                    "INSERT INTO profiles (account_id,txn_count,amt_mean,amt_std,"
                    "amt_mean_7d,amt_std_7d,amt_min,amt_max,hour_hist,merchant_counts,"
                    "known_devices,geo_cluster,velocity_1h,velocity_24h,last_txn_dt,last_updated)"
                    " VALUES (?,1,?,0,?,0,?,?,?,?,?,?,0,0,?,?)",
                    (aid, amount, amount, amount, amount,
                     json.dumps(hh),
                     json.dumps({pcd: 1} if pcd else {}),
                     json.dumps([dev] if dev else []),
                     addr, dt, now))
            return

        new_mean = α*amount + (1-α)*ex["amt_mean"]
        new_std  = α*abs(amount - ex["amt_mean"]) + (1-α)*ex["amt_std"]
        hh = ex["hour_hist"] if len(ex["hour_hist"]) == 24 else [1/24]*24
        imp = [0.0]*24; imp[hour] = 1.0
        hh_new = [α*imp[i]+(1-α)*hh[i] for i in range(24)]
        s = sum(hh_new) or 1.0
        hh_new = [v/s for v in hh_new]
        mc = ex["merchant_counts"]
        if pcd: mc[pcd] = mc.get(pcd, 0) + 1
        kd = ex["known_devices"]
        if dev and dev not in kd: kd = (kd+[dev])[-50:]
        geo = addr if addr else ex["geo_cluster"]

        with self._conn() as c:
            c.execute(
                "UPDATE profiles SET txn_count=?,amt_mean=?,amt_std=?,"
                "amt_mean_7d=?,amt_std_7d=?,amt_min=?,amt_max=?,hour_hist=?,"
                "merchant_counts=?,known_devices=?,geo_cluster=?,"
                "last_txn_dt=?,last_updated=? WHERE account_id=?",
                (ex["txn_count"]+1, round(new_mean,6), round(new_std,6),
                 round(α*amount+(1-α)*ex["amt_mean_7d"],6),
                 round(α*abs(amount-ex["amt_mean_7d"])+(1-α)*ex["amt_std_7d"],6),
                 min(ex["amt_min"], amount), max(ex["amt_max"], amount),
                 json.dumps(hh_new), json.dumps(mc), json.dumps(kd),
                 geo, dt, now, aid))

    def log_transaction(self, account_id: str, transaction_id: str,
                        p1_risk_score: float, decision: str,
                        timestamp: Optional[float] = None,
                        compromise_prob: Optional[float] = None,
                        top_trigger: str = "",
                        context: Optional[dict] = None) -> None:
        """Persist a scored event to the audit log."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO transaction_log (account_id,transaction_id,timestamp,"
                "p1_risk_score,compromise_prob,decision,top_trigger_feature,"
                "context_json,logged_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (str(account_id), str(transaction_id), timestamp or time.time(),
                 float(p1_risk_score),
                 float(compromise_prob) if compromise_prob is not None else None,
                 str(decision), str(top_trigger),
                 json.dumps(context or {}), time.time()))

    def get_history(self, account_id: str, limit: int = 20) -> list:
        """Last N scored events for an account, newest first."""
        limit = max(1, min(int(limit), 200))
        with self._conn() as c:
            rows = c.execute(
                "SELECT transaction_id,timestamp,p1_risk_score,compromise_prob,"
                "decision,top_trigger_feature,context_json FROM transaction_log"
                " WHERE account_id=? ORDER BY id DESC LIMIT ?",
                (str(account_id), limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d.pop("context_json") or "{}")
            out.append(d)
        return out

    def get_alerts(self, threshold: float = 0.7, limit: int = 50) -> list:
        """Accounts with peak compromise_prob >= threshold in last 24 h."""
        cutoff = time.time() - 86400
        limit  = max(1, min(int(limit), 500))
        with self._conn() as c:
            rows = c.execute(
                "SELECT account_id, MAX(compromise_prob) AS max_score,"
                " COUNT(*) AS alert_count, MAX(timestamp) AS last_seen"
                " FROM transaction_log"
                " WHERE compromise_prob>=? AND timestamp>=?"
                " GROUP BY account_id ORDER BY max_score DESC LIMIT ?",
                (float(threshold), cutoff, limit)).fetchall()

        def _action(s):
            if s >= 0.85: return "FREEZE_AND_NOTIFY"
            if s >= 0.70: return "STEP_UP_AUTHENTICATION"
            return "MONITOR_AND_FLAG"

        return [{"account_id": r["account_id"],
                 "max_score": round(float(r["max_score"]), 4),
                 "alert_count": int(r["alert_count"]),
                 "last_seen": float(r["last_seen"]),
                 "recommended_action": _action(float(r["max_score"]))}
                for r in rows]

    def coverage_stats(self) -> dict:
        """Diagnostic: profile count and coverage quality."""
        with self._conn() as c:
            total  = c.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
            enough = c.execute("SELECT COUNT(*) FROM profiles WHERE txn_count>=5").fetchone()[0]
            avg    = c.execute("SELECT AVG(txn_count) FROM profiles").fetchone()[0] or 0.0
        return {"total_accounts": total, "accounts_5plus": enough,
                "coverage_pct": round(100*enough/total, 1) if total else 0.0,
                "avg_txn_count": round(float(avg), 1)}