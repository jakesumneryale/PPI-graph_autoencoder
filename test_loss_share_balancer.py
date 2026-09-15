"""Tests for the loss-share balancer used by ``train_gate.py``.

``train_gate`` imports torch and torch_geometric, which are only installed on the
cluster, so the class under test is extracted from the source file instead of
imported.  The balancer itself is pure Python arithmetic.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


def load_balancer_class():
    source = Path(__file__).with_name("train_gate.py").read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "LossShareBalancer"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train_gate.py", "exec"), namespace)
    return namespace["LossShareBalancer"]


LossShareBalancer = load_balancer_class()


class FakeLoss:
    """Stands in for a torch scalar: only ``detach`` and ``__float__`` are used."""

    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self):
        return self

    def __float__(self) -> float:
        return float(self.value)


# Magnitudes measured at epoch 50 of gat4_residual_03_plus_voronoi_contact_area.
OBSERVED = {
    "node_mse": 0.04653,
    "edge_attr_mse": 9.83770,
    "edge_presence_bce": 0.16078,
    "target_mse": 0.08481,
}
SHARES = {
    "node_mse": 0.15,
    "edge_attr_mse": 0.15,
    "edge_presence_bce": 0.10,
    "target_mse": 0.60,
}


class LossShareBalancerTests(unittest.TestCase):
    def _converged_balancer(self, shares=None, steps=2000):
        balancer = LossShareBalancer(shares or SHARES, momentum=0.99)
        losses = {name: FakeLoss(value) for name, value in OBSERVED.items()}
        for _ in range(steps):
            balancer.observe(losses)
        return balancer

    def test_shares_are_normalised_to_sum_to_one(self):
        balancer = LossShareBalancer({"a": 3.0, "b": 1.0}, momentum=0.9)
        self.assertAlmostEqual(sum(balancer.shares.values()), 1.0)
        self.assertAlmostEqual(balancer.shares["a"], 0.75)

    def test_weighted_terms_reproduce_the_requested_shares(self):
        balancer = self._converged_balancer()
        weights = balancer.weights()
        contributions = {name: weights[name] * OBSERVED[name] for name in OBSERVED}
        total = sum(contributions.values())
        for name, share in SHARES.items():
            self.assertAlmostEqual(contributions[name] / total, share, places=4)

    def test_dockq_term_dominates_after_balancing(self):
        """The whole point: target_mse must carry 60% of the loss, not 0.85%."""
        balancer = self._converged_balancer()
        weights = balancer.weights()
        contributions = {name: weights[name] * OBSERVED[name] for name in OBSERVED}
        total = sum(contributions.values())
        self.assertGreater(contributions["target_mse"] / total, 0.5)

        unbalanced = {"node_mse": 1.0, "edge_attr_mse": 1.0, "edge_presence_bce": 0.1, "target_mse": 1.0}
        raw = {name: unbalanced[name] * OBSERVED[name] for name in OBSERVED}
        self.assertLess(raw["target_mse"] / sum(raw.values()), 0.01)

    def test_bias_correction_makes_early_steps_usable(self):
        balancer = LossShareBalancer(SHARES, momentum=0.99)
        losses = {name: FakeLoss(value) for name, value in OBSERVED.items()}
        balancer.observe(losses)
        weights = balancer.weights()
        contributions = {name: weights[name] * OBSERVED[name] for name in OBSERVED}
        total = sum(contributions.values())
        for name, share in SHARES.items():
            self.assertAlmostEqual(contributions[name] / total, share, places=4)

    def test_weights_before_any_observation_fall_back_to_shares(self):
        balancer = LossShareBalancer(SHARES, momentum=0.99)
        self.assertEqual(balancer.weights(), balancer.shares)

    def test_zero_share_disables_a_term(self):
        shares = dict(SHARES, edge_attr_mse=0.0)
        balancer = self._converged_balancer(shares)
        self.assertEqual(balancer.weights()["edge_attr_mse"], 0.0)

    def test_zero_magnitude_term_does_not_divide_by_zero(self):
        balancer = LossShareBalancer(SHARES, momentum=0.99)
        losses = {name: FakeLoss(0.0) for name in SHARES}
        for _ in range(10):
            balancer.observe(losses)
        for weight in balancer.weights().values():
            self.assertTrue(weight == weight and weight != float("inf"))

    def test_rejects_degenerate_shares(self):
        with self.assertRaises(ValueError):
            LossShareBalancer({"a": 0.0, "b": 0.0})
        with self.assertRaises(ValueError):
            LossShareBalancer({"a": -1.0, "b": 2.0})


if __name__ == "__main__":
    unittest.main()
