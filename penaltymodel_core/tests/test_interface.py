import unittest

import networkx as nx

import penaltymodel as pm

try:
    import penaltymodel_cache as pmc
    _cache = True
except ImportError:
    _cache = False


try:
    import penaltymodel_maxgap as maxgap
    _maxgap = True
except ImportError:
    _maxgap = False


@unittest.skipUnless(_cache, "penaltymodel_cache is not installed")
class TestInterfaceWithCache(unittest.TestCase):
    def test_retrieval(self):
        # put some stuff in the database

        spec = pm.Specification(nx.path_graph(2), (0, 1), {(-1, -1), (1, 1)}, vartype=pm.SPIN)
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        for cache in pm.iter_caches():
            cache(widget)

        # now try to get it back
        new_widget = pm.get_penalty_model(spec)

        self.assertEqual(widget, new_widget)


@unittest.skipUnless(_maxgap, "penaltymodel_maxgap is not installed")
class TestInterfaceWithMaxGap(unittest.TestCase):
    def test_retrieval(self):
        spec = pm.Specification(nx.path_graph(2), (0, 1), {(-1, -1), (1, 1)}, vartype=pm.SPIN)
        widget = pm.get_penalty_model(spec)

        self.assertEqual(widget.model.linear, {0: 0, 1: 0})
        self.assertEqual(widget.model.quadratic, {(0, 1): -1})
