import unittest

import networkx as nx

from penaltymodel import Specification

import penaltymodel_maxgap as maxgap


class TestInterface(unittest.TestCase):
    """We assume that the generation code works correctly.
    Test that the interface gives a penalty model corresponding to the specification"""
    def test_typical(self):
        graph = nx.complete_graph(3)
        spec = Specification(graph,
                             {(-1, -1): 0,
                              (+1, +1): 0},
                             [0, 1])

        pm = maxgap.get_penalty_model(spec)

        # some quick test to see that the penalty model propogated in
        for v in graph:
            self.assertIn(v, pm.model.linear)
        for (u, v) in graph.edges:
            self.assertIn(u, pm.model.adj[v])
