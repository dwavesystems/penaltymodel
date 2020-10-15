import unittest

import networkx as nx

import dimod

import penaltymodel.core as pm
from penaltymodel.cache.utils import isolated_cache

class TestInterfaceWithCache(unittest.TestCase):
    @isolated_cache
    def test_retrieval(self):
        # put some stuff in the database

        spec = pm.Specification(nx.path_graph(2), (0, 1), {(-1, -1), (1, 1)}, vartype=pm.SPIN)
        model = dimod.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        for cache in pm.iter_caches():
            cache(widget)

        # now try to get it back
        new_widget = pm.get_penalty_model(spec)

        self.assertEqual(widget, new_widget)

class TestInterfaceWithLP(unittest.TestCase):
    def assert_dict_almost_equal(self, dict0, dict1):
        self.assertEqual(len(dict0), len(dict1))
        for key in dict0:
            self.assertIn(key, dict1)
            self.assertAlmostEqual(dict0[key], dict1[key])

    @isolated_cache
    def test_retrieval(self):
        eq = {(-1, -1), (1, 1)}

        spec = pm.Specification(nx.path_graph(2), (0, 1), eq, vartype=pm.SPIN)
        widget = pm.get_penalty_model(spec)

        self.assert_dict_almost_equal(widget.model.linear, {0: 0, 1: 0})
        self.assert_dict_almost_equal(widget.model.quadratic, {(0, 1): -1})
