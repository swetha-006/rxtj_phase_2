import secrets
from fastapi.testclient import TestClient
import app

client = TestClient(app.app)

def signup(label):
    email=f'{label}-{secrets.token_hex(5)}@example.test'
    r=client.post('/auth/register',json={'email':email,'display_name':label,'password':'Guardian-Test-Password-123'})
    assert r.status_code==201, r.text
    return email

def test_org_invite_shared_case_and_tenant_isolation():
    owner_email=signup('Workspace Owner')
    org=client.post('/api/orgs',json={'name':'Guardian QA Workspace'})
    assert org.status_code==201,org.text
    oid=org.json()['organization']['id']
    assert org.json()['organization']['role']=='owner'
    invited_email=signup('Workspace Reviewer')
    client.post('/auth/login',json={'email':owner_email,'password':'Guardian-Test-Password-123'})
    invite=client.post(f'/api/orgs/{oid}/invitations',json={'email':invited_email,'role':'reviewer'})
    assert invite.status_code==201,invite.text
    token=invite.json()['invite_token']
    client.post('/auth/login',json={'email':invited_email,'password':'Guardian-Test-Password-123'})
    accepted=client.post('/api/invitations/accept',json={'token':token})
    assert accepted.status_code==200,accepted.text
    client.post('/auth/login',json={'email':owner_email,'password':'Guardian-Test-Password-123'})
    members=client.get(f'/api/orgs/{oid}/members')
    assert members.status_code==200 and len(members.json()['members'])==2
    case=client.post(f'/api/orgs/{oid}/cases',json={'transaction_id':'TX-STAGE2-001','title':'Review unusual payment','priority':'high'})
    assert case.status_code==201,case.text
    cid=case.json()['case']['id']
    note=client.post(f'/api/orgs/{oid}/cases/{cid}/notes',json={'body':'Review shared-device evidence.'})
    assert note.status_code==201,note.text
    updated=client.patch(f'/api/orgs/{oid}/cases/{cid}',json={'status':'investigating'})
    assert updated.status_code==200 and updated.json()['case']['status']=='investigating'
    # Third account is not a member and cannot read workspace data.
    outsider_email=signup('Outsider')
    outsider=client.get(f'/api/orgs/{oid}/cases')
    # Current session is outsider because signup just occurred.
    assert outsider.status_code==403
