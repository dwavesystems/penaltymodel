import unittest

import networkx as nx

import penaltymodel as pm


class TestModel(unittest.TestCase):
    pass


class TestSpecification(unittest.TestCase):
    pass


class TestPenaltyModel(unittest.TestCase):
    def test_construction(self):
        """smoke test for construction. Make sure all the information got
        propogated properly."""

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.SPIN)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.BINARY)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

    def test__repr__(self):
        """check that repr works correctly."""
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.SPIN)

        # should recreate the model
        from penaltymodel import BinaryQuadraticModel
        m2 = eval(m.__repr__())

        self.assertEqual(m, m2)
