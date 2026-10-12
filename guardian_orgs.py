"""Organization workspaces, memberships, invitations and shared investigation cases.

Demo-grade implementation. Invitation tokens are returned to the owner so local demos
work without email delivery; production deployments must deliver them securely by email.
"""
import hashlib, json, secrets, sqlite3, time
from pathlib import Path
import guardian_accounts as accounts

ROLES = {'owner', 'admin', 'reviewer', 'viewer'}

def _connect():
    c = accounts.connect()
    c.execute('PRAGMA foreign_keys=ON')
    return c

def init_db():
    with _connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS organizations (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, slug TEXT NOT NULL UNIQUE,
          created_by TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memberships (
          organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          role TEXT NOT NULL CHECK(role IN ('owner','admin','reviewer','viewer')),
          joined_at REAL NOT NULL, PRIMARY KEY(organization_id,user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships(user_id);
        CREATE TABLE IF NOT EXISTS invitations (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
          email TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','reviewer','viewer')),
          token_hash TEXT NOT NULL UNIQUE, invited_by TEXT NOT NULL REFERENCES users(id),
          created_at REAL NOT NULL, expires_at REAL NOT NULL, accepted_at REAL,
          UNIQUE(organization_id,email,accepted_at)
        );
        CREATE TABLE IF NOT EXISTS investigation_cases (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
          transaction_id TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
          priority TEXT NOT NULL DEFAULT 'medium', assigned_to TEXT REFERENCES users(id),
          created_by TEXT NOT NULL REFERENCES users(id), created_at REAL NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(organization_id,transaction_id)
        );
        CREATE TABLE IF NOT EXISTS case_notes (
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES investigation_cases(id) ON DELETE CASCADE,
          author_id TEXT NOT NULL REFERENCES users(id), body TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS organization_audit (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
          actor_id TEXT NOT NULL REFERENCES users(id), event_type TEXT NOT NULL, entity_id TEXT NOT NULL,
          metadata TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cases_org_updated ON investigation_cases(organization_id,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_case_notes_case ON case_notes(case_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_org_audit_org ON organization_audit(organization_id,created_at DESC);
        ''')

def _audit(c, org_id, actor, event, entity, metadata=None):
    c.execute('INSERT INTO organization_audit VALUES(?,?,?,?,?,?,?)', (secrets.token_urlsafe(12),org_id,actor,event,str(entity),json.dumps(metadata or {}),time.time()))

def _slug(name):
    import re
    s=re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')[:40]
    return s or 'workspace'

def create_org(user, name):
    name=name.strip()
    if not 2 <= len(name) <= 100: raise ValueError('Workspace name must be 2–100 characters.')
    now=time.time(); oid='org_'+secrets.token_urlsafe(12); slug=_slug(name)+'-'+secrets.token_hex(3)
    with _connect() as c:
        c.execute('INSERT INTO organizations VALUES(?,?,?,?,?)',(oid,name,slug,user['id'],now))
        c.execute('INSERT INTO memberships VALUES(?,?,?,?)',(oid,user['id'],'owner',now))
        _audit(c,oid,user['id'],'workspace.created',oid,{'name':name})
    return get_org(user,oid)

def list_orgs(user):
    with _connect() as c:
        rows=c.execute('SELECT o.*,m.role,(SELECT COUNT(*) FROM memberships mm WHERE mm.organization_id=o.id) member_count FROM organizations o JOIN memberships m ON m.organization_id=o.id WHERE m.user_id=? ORDER BY o.created_at DESC',(user['id'],)).fetchall()
    return [dict(r) for r in rows]

def membership(user, org_id):
    with _connect() as c:
        r=c.execute('SELECT m.role,o.id,o.name,o.slug FROM memberships m JOIN organizations o ON o.id=m.organization_id WHERE m.organization_id=? AND m.user_id=?',(org_id,user['id'])).fetchone()
    return dict(r) if r else None

def require_role(user, org_id, allowed=None):
    m=membership(user,org_id)
    if not m: raise PermissionError('You are not a member of this workspace.')
    if allowed and m['role'] not in allowed: raise PermissionError('Your workspace role does not allow this action.')
    return m

def get_org(user, org_id):
    m=membership(user,org_id)
    if not m: raise PermissionError('You are not a member of this workspace.')
    with _connect() as c:
        r=c.execute('SELECT o.*,m.role,(SELECT COUNT(*) FROM memberships mm WHERE mm.organization_id=o.id) member_count FROM organizations o JOIN memberships m ON m.organization_id=o.id WHERE o.id=? AND m.user_id=?',(org_id,user['id'])).fetchone()
    return dict(r)

def list_members(user, org_id):
    require_role(user,org_id)
    with _connect() as c:
        rows=c.execute('SELECT u.id,u.email,u.display_name,m.role,m.joined_at FROM memberships m JOIN users u ON u.id=m.user_id WHERE m.organization_id=? ORDER BY CASE m.role WHEN \'owner\' THEN 0 WHEN \'admin\' THEN 1 ELSE 2 END,u.display_name',(org_id,)).fetchall()
    return [dict(r) for r in rows]

def invite(user, org_id, email, role):
    email=email.strip().lower()
    if role not in {'admin','reviewer','viewer'}: raise ValueError('Role must be admin, reviewer, or viewer.')
    if '@' not in email or len(email)>254: raise ValueError('Enter a valid email address.')
    require_role(user,org_id,{'owner','admin'})
    now=time.time(); token=secrets.token_urlsafe(32); token_hash=hashlib.sha256(token.encode()).hexdigest(); iid='inv_'+secrets.token_urlsafe(10)
    with _connect() as c:
        if c.execute('SELECT 1 FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.organization_id=? AND u.email=?',(org_id,email)).fetchone(): raise ValueError('This user is already a workspace member.')
        c.execute('INSERT INTO invitations(id,organization_id,email,role,token_hash,invited_by,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)',(iid,org_id,email,role,token_hash,user['id'],now,now+7*86400))
        _audit(c,org_id,user['id'],'member.invited',email,{'role':role,'invitation_id':iid})
    return {'id':iid,'email':email,'role':role,'expires_at':now+7*86400,'invite_token':token}

def accept_invitation(user, token):
    token_hash=hashlib.sha256(token.encode()).hexdigest(); now=time.time()
    with _connect() as c:
        row=c.execute('SELECT * FROM invitations WHERE token_hash=? AND accepted_at IS NULL AND expires_at>?',(token_hash,now)).fetchone()
        if not row: raise ValueError('Invitation is invalid, expired, or already accepted.')
        if row['email'].lower()!=user['email'].lower(): raise PermissionError('Sign in with the email address this invitation was sent to.')
        c.execute('INSERT INTO memberships VALUES(?,?,?,?)',(row['organization_id'],user['id'],row['role'],now))
        c.execute('UPDATE invitations SET accepted_at=? WHERE id=?',(now,row['id']))
        _audit(c,row['organization_id'],user['id'],'member.joined',user['id'],{'role':row['role']})
        org_id=row['organization_id']
    return get_org(user,org_id)

def create_case(user, org_id, transaction_id, title, priority='medium', assigned_to=None):
    require_role(user,org_id,{'owner','admin','reviewer'})
    transaction_id=transaction_id.strip()[:200]; title=title.strip()[:180]
    if not transaction_id or not title: raise ValueError('Transaction ID and case title are required.')
    if priority not in {'low','medium','high','critical'}: raise ValueError('Invalid priority.')
    now=time.time(); cid='case_'+secrets.token_urlsafe(12)
    with _connect() as c:
        if assigned_to and not c.execute('SELECT 1 FROM memberships WHERE organization_id=? AND user_id=?',(org_id,assigned_to)).fetchone(): raise ValueError('Assignee must be a member of this workspace.')
        try: c.execute('INSERT INTO investigation_cases VALUES(?,?,?,?,?,?,?,?,?,?)',(cid,org_id,transaction_id,title,'open',priority,assigned_to,user['id'],now,now))
        except sqlite3.IntegrityError: raise ValueError('A case for this transaction already exists in this workspace.')
        _audit(c,org_id,user['id'],'case.created',cid,{'transaction_id':transaction_id,'priority':priority})
    return get_case(user,org_id,cid)

def list_cases(user, org_id, status=None):
    require_role(user,org_id)
    with _connect() as c:
        if status: rows=c.execute('SELECT c.*,u.display_name assigned_name FROM investigation_cases c LEFT JOIN users u ON u.id=c.assigned_to WHERE c.organization_id=? AND c.status=? ORDER BY c.updated_at DESC LIMIT 500',(org_id,status)).fetchall()
        else: rows=c.execute('SELECT c.*,u.display_name assigned_name FROM investigation_cases c LEFT JOIN users u ON u.id=c.assigned_to WHERE c.organization_id=? ORDER BY c.updated_at DESC LIMIT 500',(org_id,)).fetchall()
    return [dict(r) for r in rows]

def get_case(user, org_id, case_id):
    require_role(user,org_id)
    with _connect() as c:
        r=c.execute('SELECT * FROM investigation_cases WHERE id=? AND organization_id=?',(case_id,org_id)).fetchone()
        if not r: raise LookupError('Case not found.')
        notes=c.execute('SELECT n.id,n.body,n.created_at,u.display_name author_name FROM case_notes n JOIN users u ON u.id=n.author_id WHERE n.case_id=? ORDER BY n.created_at',(case_id,)).fetchall()
    d=dict(r); d['notes']=[dict(n) for n in notes]; return d

def update_case(user, org_id, case_id, status=None, assigned_to=None, priority=None):
    m=require_role(user,org_id,{'owner','admin','reviewer'})
    valid={'open','investigating','escalated','resolved','confirmed_fraud','legitimate'}
    with _connect() as c:
        row=c.execute('SELECT * FROM investigation_cases WHERE id=? AND organization_id=?',(case_id,org_id)).fetchone()
        if not row: raise LookupError('Case not found.')
        fields=[]; vals=[]; changes={}
        if status is not None:
            if status not in valid: raise ValueError('Invalid case status.')
            fields.append('status=?'); vals.append(status); changes['status']=status
        if priority is not None:
            if priority not in {'low','medium','high','critical'}: raise ValueError('Invalid priority.')
            fields.append('priority=?'); vals.append(priority); changes['priority']=priority
        if assigned_to is not None:
            if assigned_to and not c.execute('SELECT 1 FROM memberships WHERE organization_id=? AND user_id=?',(org_id,assigned_to)).fetchone(): raise ValueError('Assignee must be a member of this workspace.')
            fields.append('assigned_to=?'); vals.append(assigned_to or None); changes['assigned_to']=assigned_to or None
        if fields:
            fields.append('updated_at=?'); vals.append(time.time()); vals.extend([case_id,org_id])
            c.execute('UPDATE investigation_cases SET '+','.join(fields)+' WHERE id=? AND organization_id=?',vals)
            _audit(c,org_id,user['id'],'case.updated',case_id,changes)
    return get_case(user,org_id,case_id)

def add_note(user,org_id,case_id,body):
    require_role(user,org_id,{'owner','admin','reviewer'}); body=body.strip()
    if not body or len(body)>5000: raise ValueError('Note must contain 1–5000 characters.')
    now=time.time(); nid='note_'+secrets.token_urlsafe(10)
    with _connect() as c:
        if not c.execute('SELECT 1 FROM investigation_cases WHERE id=? AND organization_id=?',(case_id,org_id)).fetchone(): raise LookupError('Case not found.')
        c.execute('INSERT INTO case_notes VALUES(?,?,?,?,?)',(nid,case_id,user['id'],body,now))
        c.execute('UPDATE investigation_cases SET updated_at=? WHERE id=?',(now,case_id))
        _audit(c,org_id,user['id'],'case.note_added',case_id,{'note_id':nid})
    return {'id':nid,'body':body,'created_at':now,'author_name':user.get('display_name','')}

def audit_events(user,org_id):
    require_role(user,org_id,{'owner','admin','reviewer'})
    with _connect() as c:
        rows=c.execute('SELECT a.*,u.display_name actor_name FROM organization_audit a JOIN users u ON u.id=a.actor_id WHERE a.organization_id=? ORDER BY a.created_at DESC LIMIT 300',(org_id,)).fetchall()
    out=[]
    for r in rows:
        d=dict(r); d['metadata']=json.loads(d['metadata']); out.append(d)
    return out
