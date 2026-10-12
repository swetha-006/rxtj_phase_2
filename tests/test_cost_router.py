"""
Unit tests for RXT-J+ Cost-Sensitive Router (P4).
"""
import pytest
from cost_router import (
    CostRouter,
    ACTION_ALLOW,
    ACTION_STEP_UP,
    ACTION_BLOCK,
    decide_action,
    decide_action_details,
    LEGACY_APPROVE,
    LEGACY_STEP_UP,
    LEGACY_FREEZE,
)


def test_blueprint_worked_examples():
    """Verify exact cases from SOLUTION_BLUEPRINT.md §5.2 table."""
    router = CostRouter(c_step=20.0, c_block=400.0, gamma=0.85)

    # 1. Micro-transaction moderate risk: ₹200, p=0.30 -> STEP_UP
    r1 = router.route(p=0.30, amount=200.0)
    assert r1.action == ACTION_STEP_UP
    assert r1.recommended_action == LEGACY_STEP_UP
    assert r1.cost_allow == 60.0
    assert r1.cost_step_up == 29.0
    assert r1.cost_block == 280.0

    # 2. Micro-transaction high risk: ₹200, p=0.70 -> STEP_UP
    r2 = router.route(p=0.70, amount=200.0)
    assert r2.action == ACTION_STEP_UP
    assert r2.cost_allow == 140.0
    assert r2.cost_step_up == 41.0
    assert r2.cost_block == 120.0

    # 3. High value moderate risk: ₹80,000, p=0.30 -> BLOCK
    r3 = router.route(p=0.30, amount=80000.0)
    assert r3.action == ACTION_BLOCK
    assert r3.recommended_action == LEGACY_FREEZE
    assert r3.cost_allow == 24000.0
    assert r3.cost_step_up == 3620.0
    assert r3.cost_block == 280.0

    # 4. High value low risk: ₹80,000, p=0.08 -> BLOCK
    r4 = router.route(p=0.08, amount=80000.0)
    assert r4.action == ACTION_BLOCK
    assert r4.cost_allow == 6400.0
    assert r4.cost_step_up == 980.0
    assert r4.cost_block == 368.0


def test_zero_amount_and_edge_cases():
    """Verify zero amount and probability boundary cases."""
    router = CostRouter()

    # p = 0.0 -> ALLOW
    r_zero_risk = router.route(p=0.0, amount=50000.0)
    assert r_zero_risk.action == ACTION_ALLOW

    # amount = 0 -> ALLOW
    r_zero_amt = router.route(p=0.99, amount=0.0)
    assert r_zero_amt.action == ACTION_ALLOW

    # Clamping tests
    r_overflow = router.route(p=1.5, amount=100.0)
    assert r_overflow.p == 1.0


def test_batch_vectorization():
    """Verify batch routing gives identical results to sequential calls."""
    router = CostRouter()
    probs = [0.01, 0.30, 0.70, 0.08]
    amts = [1500.0, 200.0, 200.0, 80000.0]

    batch = router.route_batch(probs, amts)
    assert len(batch) == 4
    for i in range(4):
        single = router.route(probs[i], amts[i])
        assert batch[i].action == single.action
        assert batch[i].optimal_cost == single.optimal_cost


def test_decide_action_drop_in_compatibility():
    """Verify decide_action helper interface matches legacy app.py actions."""
    # Small amount moderate risk
    act1 = decide_action(compromise_prob=0.3, p1_risk=0.2, amount=200.0)
    assert act1 == LEGACY_STEP_UP

    # Large amount moderate risk
    act2 = decide_action(compromise_prob=0.3, p1_risk=0.2, amount=80000.0)
    assert act2 == LEGACY_FREEZE
