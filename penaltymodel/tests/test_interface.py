import unittest

import networkx as nx

import penaltymodel as pm


class TestGet(unittest.TestCase):
    def test_typical(self):
        """smoke test"""
        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)},
                                {v: (-2, 2) for v in range(2)},
                                {(0, 1): (-1, 1)})
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        penalty_model = pm.PenaltyModel(spec, model, 2, -2)

        ret_model = pm.get_penalty_model_from_specification(spec)
