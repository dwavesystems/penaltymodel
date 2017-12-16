import unittest

from collections import defaultdict
import itertools

import networkx as nx
import dwave_networkx as dnx
import penaltymodel as pm

import penaltymodel_maxgap as maxgap

from pysmt.environment import get_env, reset_env


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.env = reset_env()

    def test_impossible_model(self):
        graph = nx.path_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        with self.assertRaises(pm.ImpossiblePenaltyModel):
            maxgap.generate_ising(graph, configurations, decision_variables,
                                  linear_energy_ranges,
                                  quadratic_energy_ranges,
                                  None)

    def check_linear_energy_ranges(self, linear, linear_energy_ranges):
        for v, bias in linear.items():
            min_, max_ = linear_energy_ranges[v]
            self.assertGreaterEqual(bias, min_)
            self.assertLessEqual(bias, max_)

    def check_quadratic_energy_ranges(self, quadratic, quadratic_energy_ranges):
        for edge, bias in quadratic.items():
            min_, max_ = quadratic_energy_ranges[edge]
            self.assertGreaterEqual(bias, min_)
            self.assertLessEqual(bias, max_)

    def check_generated_ising_model(self, feasible_configurations, decision_variables,
                                    linear, quadratic, ground_energy, infeasible_gap):
        """Check that the given Ising model has the correct energy levels"""
        if not feasible_configurations:
            return

        from dimod import ExactSolver

        samples = ExactSolver().sample_ising(linear, quadratic)

        # samples are returned in order of energy
        sample, ground = next(iter(samples.items()))
        gap = float('inf')

        self.assertIn(tuple(sample[v] for v in decision_variables), feasible_configurations)

        seen_configs = set()

        for sample, energy in samples.items():
            config = tuple(sample[v] for v in decision_variables)

            # we want the minimum energy for each config of the decisison variables,
            # so once we've seen it once we can skip
            if config in seen_configs:
                continue

            if config in feasible_configurations:
                self.assertAlmostEqual(energy, ground)
                seen_configs.add(config)
            else:
                gap = min(gap, energy - ground)

        self.assertAlmostEqual(ground_energy, ground)
        self.assertAlmostEqual(gap, infeasible_gap)

    def test_trivial(self):
        # this should test things like empty graphs and empty configs
        pass

    def test_basic(self):
        """A typical use case, an AND gate on a chimera tile."""
        graph = dnx.chimera_graph(1, 1, 4)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

    def test_restricted_energy_ranges(self):
        """Create asymmetric energy ranges and test against that."""
        graph = dnx.chimera_graph(1, 1, 3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)
        linear_energy_ranges = {v: (-1., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., .5) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

    def test_disjoint(self):
        graph = dnx.chimera_graph(1, 1, 3)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, -1): 0,
                          (+1, +1, -1): 0}
        decision_variables = (0, 1, 8)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

        graph = dnx.chimera_graph(1, 1, 3)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, +1, -1): 0,
                          (+1, +1, -1, -1): 0}
        decision_variables = (0, 1, 3, 8)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

    def test_basic_no_aux(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1): 0, (1, 1, 1, 1): 0}
        decision_variables = list(graph)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

    def test_one_aux(self):
        graph = nx.complete_graph(3)

        configurations = {(-1, -1): 0, (1, 1): 0}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        decision_variables = [0, 1]

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)

    def test_specify_msat(self):
        """Test a simple model specifying yices as the smt solver. Combined
        with the other test_specify_... tests, serves as a smoke test for
        the smt_solver_name parameter.
        """
        linear_energy_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_energy_ranges = defaultdict(lambda: (-1., 1.))

        graph = nx.complete_graph(3)
        configurations = {(-1, -1): 0, (1, 1): 0}
        decision_variables = [0, 1]

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  'msat')
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)

    def test_specify_z3(self):
        """Test a simple model specifying yices as the smt solver. Combined
        with the other test_specify_... tests, serves as a smoke test for
        the smt_solver_name parameter.
        """
        graph = nx.complete_graph(3)
        configurations = {(-1, -1): 0, (1, 1): 0}
        decision_variables = [0, 1]
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  'z3')
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)

    def test_multiplication(self):

        graph = nx.complete_graph(4)
        configurations = {(x, y, x * y): 0 for x, y in itertools.product((-1, 1), repeat=2)}
        decision_variables = [0, 1, 2]

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        h, J, offset, gap = maxgap.generate_ising(graph, configurations, decision_variables,
                                                  linear_energy_ranges,
                                                  quadratic_energy_ranges,
                                                  None)
        self.check_generated_ising_model(configurations, decision_variables, h, J, offset, gap)
        self.check_linear_energy_ranges(h, linear_energy_ranges)
        self.check_quadratic_energy_ranges(J, quadratic_energy_ranges)
