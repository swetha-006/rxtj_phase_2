"""
Unit tests for RXT-J+ Fraud Ring Detector (P6).
"""
import pytest
from ring_detector import FraudRingDetector, NULL_SENTINELS


def test_isolated_account():
    det = FraudRingDetector()
    m = det.add_transaction(account_id="ACC_SOLO", device_info="UniquePhone1", addr1=123.0, timestamp=1000.0)
    assert m.ring_size == 1
    assert not m.is_ring_alert
    assert m.risk_level == "LOW"
    assert m.connected_accounts == ["ACC_SOLO"]


def test_null_and_sentinel_protection():
    """Verify missing devices and addresses do NOT link cards into fake mega-clusters."""
    det = FraudRingDetector()
    
    # Add multiple cards with null or sentinel devices
    det.add_transaction("ACC_A", device_info=None, addr1=None, timestamp=1000.0)
    det.add_transaction("ACC_B", device_info="nan", addr1="", timestamp=1001.0)
    m_c = det.add_transaction("ACC_C", device_info="unknown", addr1="missing", timestamp=1002.0)

    # ACC_C must remain isolated, size 1, no linkage
    assert m_c.ring_size == 1
    assert m_c.connected_accounts == ["ACC_C"]


def test_syndicate_ring_formation():
    """Verify multiple accounts sharing a device form a ring."""
    det = FraudRingDetector(min_ring_size_alert=3)

    # 3 accounts share a device
    det.add_transaction("CARD_1", device_info="TargetDeviceX", addr1=450.0, timestamp=1000.0, txn_seq=1)
    det.add_transaction("CARD_2", device_info="TargetDeviceX", addr1=450.0, timestamp=1200.0, txn_seq=1)
    m3 = det.add_transaction("CARD_3", device_info="TargetDeviceX", addr1=450.0, timestamp=1400.0, txn_seq=1)

    assert m3.ring_size == 3
    assert m3.is_ring_alert is True
    assert set(m3.connected_accounts) == {"CARD_1", "CARD_2", "CARD_3"}
    assert "TargetDeviceX" in m3.shared_devices
    assert "450" in m3.shared_addresses
    assert m3.ring_new_account_rate == 1.0


def test_subgraph_export_format():
    """Verify the JSON subgraph structure matches D3/Canvas UI expectations."""
    det = FraudRingDetector()
    det.add_transaction("CARD_V", device_info="DeviceVis", addr1=200.0, timestamp=1000.0)
    det.add_transaction("CARD_W", device_info="DeviceVis", addr1=200.0, timestamp=1010.0)

    payload = det.export_subgraph("CARD_V")
    assert "nodes" in payload
    assert "links" in payload
    assert "ring_metrics" in payload
    assert any(n["id"] == "card:CARD_V" and n["target"] is True for n in payload["nodes"])
    assert any(n["id"] == "dev:DeviceVis" for n in payload["nodes"])
    assert payload["summary"]["ring_size"] == 2
