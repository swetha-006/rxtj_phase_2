import pytest
import guardian_accounts as accounts

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, 'DB_PATH', str(tmp_path / 'guardian-test.db'))
    accounts.init_db()
    return accounts

def test_register_login_and_session(isolated_db):
    user = isolated_db.register('person@example.test', 'Test Person', 'A-very-long-password-123')
    assert user['email'] == 'person@example.test'
    assert isolated_db.login('person@example.test', 'A-very-long-password-123')['id'] == user['id']
    assert isolated_db.login('person@example.test', 'wrong-password') is None
    token = isolated_db.create_session(user['id'])
    assert isolated_db.get_user(token)['id'] == user['id']
    isolated_db.delete_session(token)
    assert isolated_db.get_user(token) is None

def test_records_are_isolated_per_user(isolated_db):
    a = isolated_db.register('a@example.test', 'Account A', 'Password-long-A-123')
    b = isolated_db.register('b@example.test', 'Account B', 'Password-long-B-123')
    isolated_db.put_records(a['id'], 'transactions', [{'transaction_id':'A-1','amount':100}])
    isolated_db.put_records(b['id'], 'transactions', [{'transaction_id':'B-1','amount':200}])
    assert [x['transaction_id'] for x in isolated_db.list_records(a['id'], 'transactions')] == ['A-1']
    assert [x['transaction_id'] for x in isolated_db.list_records(b['id'], 'transactions')] == ['B-1']

def test_registration_validation(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.register('not-an-email', 'Person', 'Password-long-123')
    with pytest.raises(ValueError):
        isolated_db.register('person@example.test', 'Person', 'short')
