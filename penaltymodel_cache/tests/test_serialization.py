import unittest
import random
import itertools

import networkx as nx


import penaltymodel_cache as pmc
from penaltymodel_cache.serialization import _serialize_linear_biases


class TestGraphSerialization(unittest.TestCase):
    def test_graph_encode_decode(self):
        graphs = [nx.Graph(),
                  nx.complete_graph(5),
                  nx.barbell_graph(6, 17)]

        for G in graphs:

            nodelist = list(G.nodes)
            edgelist = list(G.edges)

            num_nodes, num_edges, edges_str = pmc.serialize_graph(nodelist, edgelist)
            H_nodelist, H_edgelist = pmc.decode_graph(num_nodes, num_edges, edges_str)

            H = nx.Graph()
            H.add_nodes_from(H_nodelist)
            H.add_edges_from(H_edgelist)

            self.assertEqual(set(H.nodes), set(G.nodes))
            self.assertEqual(set(G.edges), set(H.edges))


class TestConfigurationsSerialization(unittest.TestCase):
    def test_configurations_encode_decode(self):
        configurations = {(-1, -1), (1, 1)}

        self.assertEqual(configurations,
                         pmc.decode_configurations(*pmc.serialize_configurations(configurations)))

        configurations = {(1, -1), (1, 1)}

        self.assertEqual(configurations,
                         pmc.decode_configurations(*pmc.serialize_configurations(configurations)))

        configurations = {(1, -1): 0, (1, 1): 0}

        # in this case should come out as a set
        self.assertEqual(set(configurations),
                         pmc.decode_configurations(*pmc.serialize_configurations(configurations)))

        configurations = {(1, -1): 0, (1, 1): .4}

        # in this case should come out as a set
        self.assertEqual(configurations,
                         pmc.decode_configurations(*pmc.serialize_configurations(configurations)))

    def test_docstrings(self):
        self.assertEqual(_serialize_linear_biases({1: -1, 2: 1, 3: 0}, [1, 2, 3]),
                         'AAAAAAAA8L8AAAAAAADwPwAAAAAAAAAA')
        self.assertEqual(_serialize_linear_biases({1: -1, 2: 1, 3: 0}, [3, 2, 1]),
                         'AAAAAAAAAAAAAAAAAADwPwAAAAAAAPC/')


class TestModelSerialization(unittest.TestCase):
    def test_model_encode_decode(self):
        h = {v: random.uniform(-2, 2) for v in range(100)}
        J = {(u, v): random.uniform(-1, 1) for u, v in itertools.combinations(range(100), 2)
             if random.random() > .96}
        offset = .5

        nodelist = list(range(100))
        edgelist = list(J.keys())

        linear_string, quadratic_string, off = pmc.serialize_biases(h, J, offset,
                                                                    nodelist, edgelist)
        hh, JJ, off = pmc.decode_biases(linear_string, quadratic_string, off,
                                        nodelist, edgelist)

        self.assertEqual(hh, h)
        self.assertEqual(JJ, J)
        self.assertEqual(off, offset)
