from collections import defaultdict

import pysmt.test

import networkx as nx
import dwave_networkx as dnx

from dwave_maxgap.smt import Theta, Table


class TestTheta(pysmt.test.TestCase):
    def test_thetaconstruction(self):
        """test that when we construct theta from a graph it has
        the correct structure."""

        graphs = [nx.complete_graph(5),
                  dnx.chimera_graph(1),
                  nx.path_graph(12)]
        linear_ranges = defaultdict(lambda: (-2., 2))
        quadratic_ranges = defaultdict(lambda: (-1., 1))

        for graph in graphs:
            theta = Theta()
            theta.build_from_graph(graph, linear_ranges, quadratic_ranges)

            # ok, let's check that the set of nodes are the same for graph and theta
            self.assertEqual(set(graph.nodes), set(theta.linear))

            # let's also check that they both have the same edges
            for u, v in graph.edges:
                self.assertIn(u, theta.adj)
                self.assertIn(v, theta.adj[u])
                self.assertIn(v, theta.adj)
                self.assertIn(u, theta.adj[v])
            for v, u in theta.quadratic:
                self.assertIn(u, theta.adj)
                self.assertIn(v, theta.adj[u])
                self.assertIn(v, theta.adj)
                self.assertIn(u, theta.adj[v])
                self.assertIn(u, graph)
                self.assertIn(v, graph[u])
                self.assertIn(v, graph)
                self.assertIn(u, graph[v])

            # check that the edges are unique
            for u, v in theta.quadratic:
                self.assertNotIn((v, u), theta.quadratic)

            # check that each bias is unique
            self.assertEqual(len(set(id(bias) for __, bias in theta.linear.items())),
                             len(theta.linear))
            self.assertEqual(len(set(id(bias) for __, bias in theta.quadratic.items())),
                             len(theta.quadratic))
