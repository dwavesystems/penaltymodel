import unittest

import networkx as nx

import dwave_maxgap as maxgap


class Test_maxgap_small_no_aux(unittest.TestCase):
    def test_basic(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1), (1, 1, 1, 1)}

        pm = maxgap.generate_small_no_aux(graph, configurations, list(graph))