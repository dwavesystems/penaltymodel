from collections import defaultdict

import pysmt.test
from pysmt.shortcuts import GT, LT, Real, And, Equals

import networkx as nx
import dwave_networkx as dnx

from dwave_maxgap.smt import Theta, Table


class TestTheta(pysmt.test.TestCase):
    def test_thetaconstruction(self):
        """test that when we construct theta from a graph it has
        the correct structure."""

        disconnect1 = nx.Graph()
        disconnect1.add_nodes_from([0, 1])
        disconnect2 = nx.complete_graph(4)
        disconnect2.add_edge(4, 5)

        graphs = [nx.complete_graph(5),
                  dnx.chimera_graph(1),
                  nx.path_graph(12),
                  nx.Graph(),
                  nx.complete_graph(1),
                  disconnect1, disconnect2]
        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

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

                # make sure that everything points to the same bias
                self.assertEqual(id(theta.quadratic[(v, u)]), id(theta.adj[u][v]))
                self.assertEqual(id(theta.quadratic[(v, u)]), id(theta.adj[v][u]))

            # check that the edges are unique
            for u, v in theta.quadratic:
                self.assertNotIn((v, u), theta.quadratic)

            # check that each bias is unique
            self.assertEqual(len(set(id(bias) for __, bias in theta.linear.items())),
                             len(theta.linear))
            self.assertEqual(len(set(id(bias) for __, bias in theta.quadratic.items())),
                             len(theta.quadratic))

    def test_energy_ranges(self):

        disconnect1 = nx.Graph()
        disconnect1.add_nodes_from([0, 1])
        disconnect2 = nx.complete_graph(4)
        disconnect2.add_edge(4, 5)

        graphs = [nx.complete_graph(5),
                  dnx.chimera_graph(1),
                  nx.path_graph(12),
                  nx.Graph(),
                  nx.complete_graph(1),
                  disconnect1, disconnect2]
        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        for graph in graphs:
            theta = Theta()
            theta.build_from_graph(graph, linear_ranges, quadratic_ranges)

            for v, bias in theta.linear.items():
                min_, max_ = linear_ranges[v]
                self.assertUnsat(And(GT(bias, Real(max_)), And(theta.assertions)))
                self.assertUnsat(And(LT(bias, Real(min_)), And(theta.assertions)))

            for (u, v), bias in theta.quadratic.items():
                min_, max_ = quadratic_ranges[(u, v)]
                self.assertUnsat(And(GT(bias, Real(max_)), And(theta.assertions)))
                self.assertUnsat(And(LT(bias, Real(min_)), And(theta.assertions)))

    def test_energy(self):

        disconnect1 = nx.Graph()
        disconnect1.add_nodes_from([0, 1])
        disconnect2 = nx.complete_graph(4)
        disconnect2.add_edge(4, 5)

        graphs = [nx.complete_graph(5),
                  dnx.chimera_graph(1),
                  nx.path_graph(12),
                  nx.Graph(),
                  nx.complete_graph(1),
                  disconnect1, disconnect2]

        # set the values exactly
        linear_ranges = defaultdict(lambda: (1., 1.))
        quadratic_ranges = defaultdict(lambda: (-1., -1.))

        for graph in graphs:
            theta = Theta()
            theta.build_from_graph(graph, linear_ranges, quadratic_ranges)

            spins = {v: 1 for v in graph}

            # classical energy
            energy = 6.  # offset = 6
            for v in graph:
                energy += spins[v] * linear_ranges[v][0]
            for u, v in graph.edges:
                energy += spins[v] * spins[u] * quadratic_ranges[(u, v)][0]

            smt_energy = theta.energy(spins)

            self.assertSat(And([Equals(Real(energy), smt_energy),
                                And(theta.assertions),
                                Equals(theta.offset, Real(6.))]))
            self.assertUnsat(And([Equals(Real(energy), smt_energy),
                                  And(theta.assertions),
                                  Equals(theta.offset, Real(6.1))]))

            # let's also test the energy of subtheta
            # fixing all of the variable puts the whole value into offset
            subtheta = theta.fix_variables(spins)
            self.assertSat(And([Equals(Real(energy), subtheta.offset),
                                And(theta.assertions),
                                Equals(theta.offset, Real(6.))]))
            self.assertUnsat(And([Equals(Real(energy), subtheta.offset),
                                  And(theta.assertions),
                                  Equals(theta.offset, Real(6.1))]))

            # finally let's try fixing a subset of variables
            if len(graph) < 3:
                continue

            subspins = {0: 1, 1: 1}
            subtheta = theta.fix_variables(subspins)

            smt_energy = subtheta.energy(spins)

            self.assertSat(And([Equals(Real(energy), smt_energy),
                                And(theta.assertions),
                                Equals(theta.offset, Real(6.))]))
            self.assertUnsat(And([Equals(Real(energy), smt_energy),
                                  And(theta.assertions),
                                  Equals(theta.offset, Real(6.1))]))