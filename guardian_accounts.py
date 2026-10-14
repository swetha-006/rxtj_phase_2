"""Minimal first-party account/session store for the Guardian demo.

Uses SQLite + PBKDF2 and opaque HttpOnly sessions. For public production, use
TLS, rate limiting, email verification, MFA, secret rotation, and security review.
"""
import hashlib, hmac, os, secrets, sqlite3, time
from pathlib import Path

DB_PATH = os.environ.get('GUARDIAN_ACCOUNTS_DB', str(Path(__file__).resolve().parent / 'data' / 'guardian_accounts.db'))
ITERATIONS = 310_000

def connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    return c

def init_db():
    with connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, display_name TEXT NOT NULL,
          password_hash TEXT NOT NULL, salt TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          created_at REAL NOT NULL, expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_records (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          record_type TEXT NOT NULL, record_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(user_id, record_type, record_key)
        );
        CREATE INDEX IF NOT EXISTS idx_records_user_type ON user_records(user_id, record_type, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
        ''')
        # Seed default permanent demo account if not exists
        demo_email = os.environ.get('GUARDIAN_DEMO_EMAIL', 'demo@guardian.ai').strip().lower()
        row = c.execute('SELECT id FROM users WHERE email=?', (demo_email,)).fetchone()
        if row is None:
            demo_pass = os.environ.get('GUARDIAN_DEMO_PASSWORD', 'GuardianDemo2026!')
            demo_name = os.environ.get('GUARDIAN_DEMO_NAME', 'Demo Lead Analyst')
            salt = secrets.token_bytes(16).hex()
            uid = 'usr_' + secrets.token_urlsafe(14)
            c.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',
                      (uid, demo_email, demo_name, _hash_password(demo_pass, salt), salt, time.time()))

def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), ITERATIONS).hex()

def register(email, display_name, password):
    email = email.strip().lower()
    if len(email) > 254 or '@' not in email or '.' not in email.rsplit('@', 1)[-1]:
        raise ValueError('Enter a valid email address.')
    if not display_name.strip() or len(display_name.strip()) > 80:
        raise ValueError('Name must be between 1 and 80 characters.')
    if len(password) < 10 or len(password) > 128:
        raise ValueError('Use a password between 10 and 128 characters.')
    salt = secrets.token_bytes(16).hex()
    uid = secrets.token_urlsafe(18)
    try:
        with connect() as c:
            c.execute('INSERT INTO users VALUES(?,?,?,?,?,?)', (uid, email, display_name.strip(), _hash_password(password, salt), salt, time.time()))
    except sqlite3.IntegrityError:
        raise ValueError('An account with this email already exists. Sign in instead.')
    return {'id': uid, 'email': email, 'display_name': display_name.strip(), 'created_at': time.time()}

def login(email, password):
    with connect() as c:
        row = c.execute('SELECT * FROM users WHERE email=?', (email.strip().lower(),)).fetchone()
    # Constant-ish work for unknown emails to reduce account enumeration timing differences.
    if row is None:
        _hash_password(password, '00112233445566778899aabbccddeeff')
        return None
    if not hmac.compare_digest(_hash_password(password, row['salt']), row['password_hash']):
        return None
    return {'id': row['id'], 'email': row['email'], 'display_name': row['display_name'], 'created_at': row['created_at']}

def create_session(user_id):
    token = secrets.token_urlsafe(36)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with connect() as c:
        c.execute('DELETE FROM sessions WHERE expires_at < ?', (now,))
        c.execute('INSERT INTO sessions VALUES(?,?,?,?)', (token_hash, user_id, now, now + 60 * 60 * 24 * 14))
    return token

def get_user(token):
    if not token: return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connect() as c:
        row = c.execute('SELECT u.id,u.email,u.display_name,u.created_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?', (token_hash, time.time())).fetchone()
    return dict(row) if row else None

def delete_session(token):
    if token:
        with connect() as c: c.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))

def list_records(user_id, record_type):
    import json
    with connect() as c:
        rows = c.execute('SELECT payload FROM user_records WHERE user_id=? AND record_type=? ORDER BY updated_at DESC LIMIT 500', (user_id, record_type)).fetchall()
    return [json.loads(r['payload']) for r in rows]

def put_records(user_id, record_type, records):
    import json
    if len(records) > 500: raise ValueError('A maximum of 500 records can be synced at once.')
    now = time.time()
    with connect() as c:
        for record in records:
            key = str(record.get('transaction_id') or record.get('id') or record.get('time') or '')[:200]
            if not key: continue
            payload = json.dumps(record, separators=(',', ':'), ensure_ascii=False)
            c.execute('INSERT INTO user_records(user_id,record_type,record_key,payload,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(user_id,record_type,record_key) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, record_type, key, payload, now))
