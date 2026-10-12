import secrets
from fastapi.testclient import TestClient
import app

client = TestClient(app.app)
PASSWORD = 'Guardian-Test-Password-123'

def account(label):
    email = f'{label.lower().replace(" ", "-")}-{secrets.token_hex(5)}@example.test'
    r = client.post('/auth/register', json={'email': email, 'display_name': label, 'password': PASSWORD})
    assert r.status_code == 201, r.text
    return email

def test_stage3_analytics_alerts_filter_and_csv_export():
    email = account('Stage Three Owner')
    assert client.post('/auth/login', json={'email': email, 'password': PASSWORD}).status_code == 200
    org_resp = client.post('/api/orgs', json={'name': 'Stage Three Operations'})
    assert org_resp.status_code == 201, org_resp.text
    oid = org_resp.json()['organization']['id']
    for txn, title, priority in [
        ('TX-S3-HIGH', 'Review unusual device', 'high'),
        ('TX-S3-CRIT', 'Critical velocity spike', 'critical'),
        ('TX-S3-LOW', 'Routine review', 'low'),
    ]:
        resp = client.post(f'/api/orgs/{oid}/cases', json={'transaction_id': txn, 'title': title, 'priority': priority})
        assert resp.status_code == 201, resp.text
    analytics = client.get(f'/api/orgs/{oid}/analytics')
    assert analytics.status_code == 200, analytics.text
    data = analytics.json()
    assert data['total_cases'] == 3
    assert data['priorities']['critical'] == 1
    assert data['unassigned_cases'] == 3
    alerts = client.get(f'/api/orgs/{oid}/alerts')
    assert alerts.status_code == 200
    assert {a['transaction_id'] for a in alerts.json()['alerts']} == {'TX-S3-HIGH', 'TX-S3-CRIT'}
    filtered = client.get(f'/api/orgs/{oid}/cases', params={'priority': 'critical', 'q': 'velocity'})
    assert filtered.status_code == 200
    assert [c['transaction_id'] for c in filtered.json()['cases']] == ['TX-S3-CRIT']
    export = client.get(f'/api/orgs/{oid}/cases-export.csv')
    assert export.status_code == 200
    assert 'text/csv' in export.headers['content-type']
    assert 'TX-S3-HIGH' in export.text and 'TX-S3-CRIT' in export.text

def test_stage3_endpoints_reject_nonmembers():
    owner = account('Stage Three Isolated Owner')
    client.post('/auth/login', json={'email': owner, 'password': PASSWORD})
    org = client.post('/api/orgs', json={'name': 'Private Stage Three Workspace'}).json()['organization']
    outsider = account('Stage Three Isolated Outsider')
    client.post('/auth/login', json={'email': outsider, 'password': PASSWORD})
    for path in [f"/api/orgs/{org['id']}/analytics", f"/api/orgs/{org['id']}/alerts", f"/api/orgs/{org['id']}/cases-export.csv"]:
        assert client.get(path).status_code == 403
