"""
RXT-J+ v4 Cost-Sensitive Decision Router (P4)
==============================================
Replaces static flat-threshold decisioning with expected-rupee cost optimization.

Mathematical Specification (SOLUTION_BLUEPRINT.md §5.2):
  cost(ALLOW)   = p * amount
  cost(STEP_UP) = C_step + p * (1 - γ) * amount
  cost(BLOCK)   = C_block * (1 - p)

Where:
  p       = calibrated fraud probability [0.0, 1.0]
  amount  = transaction amount in ₹ (INR)
  γ       = fraction of fraud defeated by step-up authentication (default: 0.85)
  C_step  = friction cost of step-up challenge (default: ₹20)
  C_block = business cost of blocking a legitimate customer (default: ₹400)

Decision rule:
  action  = argmin(cost(ALLOW), cost(STEP_UP), cost(BLOCK))
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Standard action constants
ACTION_ALLOW = "ALLOW"
ACTION_STEP_UP = "STEP_UP"
ACTION_BLOCK = "BLOCK"

# Legacy API mapping constants for seamless backward compatibility with app.py
LEGACY_APPROVE = "APPROVE"
LEGACY_STEP_UP = "STEP_UP_AUTHENTICATION"
LEGACY_FREEZE = "FREEZE_AND_NOTIFY"
LEGACY_MONITOR = "MONITOR_AND_FLAG"

ACTION_TO_LEGACY = {
    ACTION_ALLOW: LEGACY_APPROVE,
    ACTION_STEP_UP: LEGACY_STEP_UP,
    ACTION_BLOCK: LEGACY_FREEZE,
}


@dataclass
class CostDecision:
    """Detailed output of a cost-sensitive decision."""
    action: str                   # 'ALLOW', 'STEP_UP', 'BLOCK'
    recommended_action: str       # Legacy-compatible: 'APPROVE', 'STEP_UP_AUTHENTICATION', 'FREEZE_AND_NOTIFY'
    p: float                      # Calibrated fraud probability
    amount: float                 # Transaction amount
    cost_allow: float             # Expected rupee cost of ALLOW
    cost_step_up: float           # Expected rupee cost of STEP_UP
    cost_block: float             # Expected rupee cost of BLOCK
    optimal_cost: float           # Minimum cost achieved
    flat_threshold_action: str    # Decision under static 0.15 threshold
    flat_threshold_cost: float    # Expected cost under static 0.15 threshold
    rupee_savings: float          # Savings of Cost Router vs flat threshold
    explanation: str              # Concise executive rationale

    def to_dict(self) -> Dict:
        return asdict(self)


class CostRouter:
    """
    Cost-Sensitive Decision Router engine with parameterized business constants.
    """

    def __init__(
        self,
        c_step: float = 20.0,
        c_block: float = 400.0,
        gamma: float = 0.85,
        flat_threshold_baseline: float = 0.15,
    ):
        """
        Initialize router parameters.
        :param c_step: Friction cost of OTP / step-up challenge in ₹.
        :param c_block: False positive penalty (customer churn + support) in ₹.
        :param gamma: Challenge efficacy (proportion of fraud stopped by step-up, 0..1).
        :param flat_threshold_baseline: Comparison threshold for baseline bench.
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if c_step < 0 or c_block < 0:
            raise ValueError("Costs must be non-negative")

        self.c_step = float(c_step)
        self.c_block = float(c_block)
        self.gamma = float(gamma)
        self.flat_threshold_baseline = float(flat_threshold_baseline)

    def route(self, p: float, amount: float) -> CostDecision:
        """
        Route a single transaction based on calibrated fraud probability and amount.
        """
        p_clamped = float(np.clip(p, 0.0, 1.0))
        amt = max(0.0, float(amount))

        # Expected cost arithmetic
        cost_allow = p_clamped * amt
        cost_step_up = self.c_step + p_clamped * (1.0 - self.gamma) * amt
        cost_block = self.c_block * (1.0 - p_clamped)

        costs = {
            ACTION_ALLOW: cost_allow,
            ACTION_STEP_UP: cost_step_up,
            ACTION_BLOCK: cost_block,
        }

        # Select action with minimum expected cost
        best_action = min(costs, key=costs.get)
        optimal_cost = costs[best_action]

        # Baseline comparison: what would a standard flat threshold do?
        # Standard flat rule: p >= baseline -> BLOCK (or STEP_UP if near borderline)
        flat_action = ACTION_BLOCK if p_clamped >= self.flat_threshold_baseline else ACTION_ALLOW
        flat_cost = costs[flat_action]
        rupee_savings = max(0.0, flat_cost - optimal_cost)

        # Generate human-understandable explanation
        explanation = self._build_explanation(best_action, p_clamped, amt, costs)

        return CostDecision(
            action=best_action,
            recommended_action=ACTION_TO_LEGACY[best_action],
            p=round(p_clamped, 4),
            amount=round(amt, 2),
            cost_allow=round(cost_allow, 2),
            cost_step_up=round(cost_step_up, 2),
            cost_block=round(cost_block, 2),
            optimal_cost=round(optimal_cost, 2),
            flat_threshold_action=flat_action,
            flat_threshold_cost=round(flat_cost, 2),
            rupee_savings=round(rupee_savings, 2),
            explanation=explanation,
        )

    def _build_explanation(
        self, action: str, p: float, amt: float, costs: Dict[str, float]
    ) -> str:
        if action == ACTION_ALLOW:
            return (
                f"Low fraud probability ({p*100:.1f}%) on ₹{amt:,.0f} makes direct approval "
                f"the lowest risk action (expected loss ₹{costs[ACTION_ALLOW]:,.2f})."
            )
        elif action == ACTION_STEP_UP:
            return (
                f"Moderate risk ({p*100:.1f}%) on ₹{amt:,.0f}. Step-up challenge costs only "
                f"₹{costs[ACTION_STEP_UP]:,.2f}, saving ₹{costs[ACTION_BLOCK] - costs[ACTION_STEP_UP]:,.2f} "
                f"vs an aggressive block and ₹{costs[ACTION_ALLOW] - costs[ACTION_STEP_UP]:,.2f} vs unchecked approval."
            )
        else:  # ACTION_BLOCK
            return (
                f"High-impact risk on ₹{amt:,.0f} with {p*100:.1f}% fraud probability. "
                f"Potential loss (₹{costs[ACTION_ALLOW]:,.2f}) outweighs false-positive customer insult cost "
                f"(₹{costs[ACTION_BLOCK]:,.2f}). Blocking is optimal."
            )

    def route_batch(
        self, probs: Union[np.ndarray, List[float]], amounts: Union[np.ndarray, List[float]]
    ) -> List[CostDecision]:
        """Vectorized execution over an array of transactions."""
        p_arr = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
        amt_arr = np.maximum(0.0, np.asarray(amounts, dtype=np.float64))

        if len(p_arr) != len(amt_arr):
            raise ValueError(f"Length mismatch: {len(p_arr)} probs vs {len(amt_arr)} amounts")

        # Vectorized cost matrix: shape (N, 3)
        c_allow = p_arr * amt_arr
        c_step = self.c_step + p_arr * (1.0 - self.gamma) * amt_arr
        c_block = self.c_block * (1.0 - p_arr)

        cost_matrix = np.column_stack([c_allow, c_step, c_block])
        best_indices = np.argmin(cost_matrix, axis=1)

        actions_map = [ACTION_ALLOW, ACTION_STEP_UP, ACTION_BLOCK]

        results = []
        for i in range(len(p_arr)):
            best_action = actions_map[best_indices[i]]
            costs_dict = {
                ACTION_ALLOW: c_allow[i],
                ACTION_STEP_UP: c_step[i],
                ACTION_BLOCK: c_block[i],
            }
            opt_cost = cost_matrix[i, best_indices[i]]
            flat_act = ACTION_BLOCK if p_arr[i] >= self.flat_threshold_baseline else ACTION_ALLOW
            flat_c = costs_dict[flat_act]
            savings = max(0.0, flat_c - opt_cost)

            results.append(
                CostDecision(
                    action=best_action,
                    recommended_action=ACTION_TO_LEGACY[best_action],
                    p=round(float(p_arr[i]), 4),
                    amount=round(float(amt_arr[i]), 2),
                    cost_allow=round(float(c_allow[i]), 2),
                    cost_step_up=round(float(c_step[i]), 2),
                    cost_block=round(float(c_block[i]), 2),
                    optimal_cost=round(float(opt_cost), 2),
                    flat_threshold_action=flat_act,
                    flat_threshold_cost=round(float(flat_c), 2),
                    rupee_savings=round(float(savings), 2),
                    explanation=self._build_explanation(best_action, float(p_arr[i]), float(amt_arr[i]), costs_dict),
                )
            )
        return results

    def compare_portfolio(
        self, probs: np.ndarray, amounts: np.ndarray, true_fraud: np.ndarray
    ) -> Dict[str, float]:
        """
        Simulate total financial outcome across a portfolio comparing:
        1. Pure unmanaged (ALLOW all)
        2. Static flat-threshold policy (e.g. 0.15)
        3. Cost-Sensitive Optimal Router
        """
        p_arr = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
        amt_arr = np.asarray(amounts, dtype=np.float64)
        y_true = np.asarray(true_fraud, dtype=np.int32)

        # Cost router decisions
        batch = self.route_batch(p_arr, amt_arr)

        cost_router_loss = 0.0
        flat_threshold_loss = 0.0
        unmanaged_loss = float(np.sum(amt_arr[y_true == 1]))

        for i, d in enumerate(batch):
            is_fraud = bool(y_true[i] == 1)
            amt = amt_arr[i]

            # Actual loss under Cost Router:
            if d.action == ACTION_ALLOW:
                cost_router_loss += amt if is_fraud else 0.0
            elif d.action == ACTION_STEP_UP:
                # Defeats gamma fraction of fraud, plus fixed challenge friction
                cost_router_loss += self.c_step
                if is_fraud:
                    cost_router_loss += (1.0 - self.gamma) * amt
            elif d.action == ACTION_BLOCK:
                # If legit, incur customer friction cost; if fraud, 0 financial fraud loss
                cost_router_loss += 0.0 if is_fraud else self.c_block

            # Actual loss under Flat Threshold (p >= 0.15 -> block, else allow):
            if d.flat_threshold_action == ACTION_ALLOW:
                flat_threshold_loss += amt if is_fraud else 0.0
            else:
                flat_threshold_loss += 0.0 if is_fraud else self.c_block

        return {
            "unmanaged_total_loss": round(unmanaged_loss, 2),
            "flat_threshold_loss": round(flat_threshold_loss, 2),
            "cost_router_loss": round(cost_router_loss, 2),
            "savings_vs_flat_threshold": round(flat_threshold_loss - cost_router_loss, 2),
            "savings_pct_vs_flat": round(
                100.0 * (flat_threshold_loss - cost_router_loss) / max(flat_threshold_loss, 1.0), 2
            ),
        }


# Module-level convenience singleton matching app.py usage pattern
default_router = CostRouter()


def decide_action(
    compromise_prob: float,
    p1_risk: float,
    amount: float = 100.0,
    router: Optional[CostRouter] = None,
) -> str:
    """
    Drop-in replacement for app.py decide_action().
    Combines calibrated fraud risk and transaction amount into optimal expected cost action.
    """
    active_router = router or default_router
    # Use max of Phase 1 risk and compromise prob as calibrated p
    effective_p = max(float(compromise_prob), float(p1_risk))
    decision = active_router.route(p=effective_p, amount=amount)
    return decision.recommended_action


def decide_action_details(
    p: float,
    amount: float,
    router: Optional[CostRouter] = None,
) -> CostDecision:
    """Detailed version returning complete cost breakdown and explanation."""
    active_router = router or default_router
    return active_router.route(p=p, amount=amount)


if __name__ == "__main__":
    print("=" * 78)
    print("  RXT-J+ v4 COST-SENSITIVE DECISION ROUTER VERIFICATION")
    print("=" * 78)

    test_router = CostRouter(c_step=20.0, c_block=400.0, gamma=0.85)

    # Verification cases from SOLUTION_BLUEPRINT.md §5.2
    cases = [
        {"desc": "Coffee / micro-payment", "amount": 200.0, "p": 0.30, "expected": "STEP_UP"},
        {"desc": "Micro-payment high risk", "amount": 200.0, "p": 0.70, "expected": "STEP_UP"},
        {"desc": "Large wire transfer high risk", "amount": 80000.0, "p": 0.30, "expected": "BLOCK"},
        {"desc": "Large wire transfer low risk", "amount": 80000.0, "p": 0.08, "expected": "BLOCK"},
        {"desc": "Legitimate everyday grocery", "amount": 1500.0, "p": 0.01, "expected": "ALLOW"},
    ]

    print("\n[1] Checking Benchmark Scenarios (Blueprint §5.2 Table):")
    print("-" * 78)
    print(f"{'Amount':>10} | {'p':>6} | {'ALLOW Cost':>11} | {'STEP_UP Cost':>13} | {'BLOCK Cost':>11} | {'Decision':>9} | {'Match?':>7}")
    print("-" * 78)

    all_matched = True
    for c in cases:
        res = test_router.route(p=c["p"], amount=c["amount"])
        matched = res.action == c["expected"]
        all_matched = all_matched and matched
        match_str = "PASS" if matched else "FAIL"
        print(
            f"₹{res.amount:>9,.0f} | {res.p:>6.2f} | ₹{res.cost_allow:>10,.2f} | "
            f"₹{res.cost_step_up:>12,.2f} | ₹{res.cost_block:>10,.2f} | "
            f"{res.action:>9} | {match_str:>7}"
        )

    print("-" * 78)
    print(f"Blueprint Table Verification: {'ALL PASSED' if all_matched else 'SOME FAILED'}\n")

    print("[2] Explanations Sample:")
    print("-" * 78)
    for c in cases[:3]:
        res = test_router.route(p=c["p"], amount=c["amount"])
        print(f"• [₹{c['amount']:,.0f} @ p={c['p']:.2f} -> {res.action}]")
        print(f"  {res.explanation}\n")
