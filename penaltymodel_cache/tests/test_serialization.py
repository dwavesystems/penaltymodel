import unittest

import networkx as nx


import penaltymodel_cache as pmc


class TestGraphSerialization(unittest.TestCase):
    def test_graph_encode_decode(self):
        graphs = [nx.Graph(),
                  nx.complete_graph(5),
                  nx.barbell_graph(6, 17)]

        for G in graphs:
            num_nodes, num_edges, edges_str = pmc.serialize_graph(G)
            H = pmc.decode_graph(num_nodes, num_edges, edges_str)

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
