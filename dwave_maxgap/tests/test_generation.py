import unittest

import networkx as nx
import dwave_networkx as dnx

import dwave_maxgap as maxgap


class TestNoAux(unittest.TestCase):
    def test_basic(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1), (1, 1, 1, 1)}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising_no_aux(graph, configurations, list(graph),
                                                         linear_energy_ranges,
                                                         quadratic_energy_ranges)

        # TODO, something interesting here


class TestGeneration(unittest.TestCase):
    def test_basic(self):
        graph = dnx.chimera_graph(1)

        configurations = {(-1, -1, -1),
                          (1, 1, 1),
                          (-1, 1, -1),
                          (1, -1, -1)}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations,
                                                  (0, 1, 2),
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges)