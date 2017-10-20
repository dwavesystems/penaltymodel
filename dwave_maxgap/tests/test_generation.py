import unittest

import networkx as nx
import dwave_networkx as dnx

import dwave_maxgap as maxgap

from pysmt.environment import get_env, reset_env


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.env = reset_env()

    def test_trivial(self):
        # this should test things like empty graphs and empty configs
        pass

    def test_basic(self):
        graph = dnx.chimera_graph(1)
        graph = graph.subgraph([0, 1, 2, 3, 4, 5, 6, 7])

        configurations = {(-1, -1, -1),
                          (-1, +1, -1),
                          (+1, -1, -1),
                          (+1, +1, +1)}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations,
                                                  (0, 1, 2),
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges)

        print(h)
        print(J)
        print(offset)
        print(gap)

    def test_basic_no_aux(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1), (1, 1, 1, 1)}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, list(graph),
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges)

        # print(h)
        # print(J)
        # print(offset)
        # print(gap)

    def test_one_aux(self):
        graph = nx.complete_graph(3)

        configurations = {(-1, -1), (1, 1)}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, [0, 1],
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges)

        # print(h)
        # print(J)
        # print(offset)
        # print(gap)