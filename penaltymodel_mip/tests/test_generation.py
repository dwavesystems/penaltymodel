import unittest
import itertools

# from collections import defaultdict

import dimod
import networkx as nx


import penaltymodel.core as pm
import penaltymodel.mip as mip


class TestGeneration(unittest.TestCase):

    #
    # Utilities
    #

    def check_bqm_table(self, bqm, gap, table, decision):
        """check that the bqm has ground states matching table"""
        response = dimod.ExactSolver().sample(bqm)

        highest_feasible_energy = max(table.values()) if isinstance(table, dict) else 0

        seen_gap = float('inf')
        seen_table = set()
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)  # sanity check

            config = tuple(sample[v] for v in decision)

            if energy < highest_feasible_energy + .001:
                self.assertIn(config, table)

                if isinstance(table, dict):
                    self.assertAlmostEqual(table[config], energy)

                seen_table.add(config)

            elif config not in table:
                seen_gap = min(seen_gap, energy - highest_feasible_energy)

        for config in table:
            self.assertIn(config, seen_table)

        self.assertEqual(seen_gap, gap)
        self.assertGreater(gap, 0)

    def check_bqm_graph(self, bqm, graph):
        """bqm and graph have the same structure"""
        self.assertEqual(len(bqm.linear), len(graph.nodes))
        self.assertEqual(len(bqm.quadratic), len(graph.edges))

        for v in bqm.linear:
            self.assertIn(v, graph)
        for u, v in bqm.quadratic:
            self.assertIn(u, graph.adj[v])

    #
    # Tests
    #

    def test_impossible_AND_3path(self):
        """AND gate cannot exist on a 3-path"""

        graph = nx.path_graph(3)

        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}

        decision_variables = (0, 1, 2)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}

        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        with self.assertRaises(pm.ImpossiblePenaltyModel):
            mip.generate_bqm(graph,
                             configurations,
                             decision_variables,
                             linear_energy_ranges=linear_energy_ranges,
                             quadratic_energy_ranges=quadratic_energy_ranges)

    def test_empty_no_aux(self):
        graph = nx.Graph()
        configurations = {}
        decision = []

        bqm, offset = mip.generate_bqm(graph, configurations, decision)

        self.check_bqm_graph(bqm, graph)
        self.assertEqual(offset, 0)

    def test_empty_some_aux(self):
        graph = nx.complete_graph(3)
        configurations = {}
        decision = []

        bqm, offset = mip.generate_bqm(graph, configurations, decision)

        self.check_bqm_graph(bqm, graph)
        self.assertEqual(offset, 0)

    def test_AND_K4(self):
        """A typical use case, an AND gate on a K4."""
        graph = nx.complete_graph(4)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        bqm, gap = mip.generate_bqm(graph, configurations, decision_variables)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_NAE3SAT_4cycle(self):
        """A typical use case, an AND gate on a K4."""
        graph = nx.cycle_graph(4)
        configurations = {config for config in itertools.product((-1, 1), repeat=3) if len(set(config)) > 1}
        decision_variables = (0, 1, 2)

        bqm, gap = mip.generate_bqm(graph, configurations, decision_variables)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_restricted_energy_ranges(self):
        """Create asymmetric energy ranges and test against that."""
        graph = nx.complete_graph(5)

        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}

        decision_variables = (0, 1, 2)

        linear_energy_ranges = {v: (-1., 2.) for v in graph}

        quadratic_energy_ranges = {(u, v): (-1., .5) for u, v in graph.edges}

        bqm, gap = mip.generate_bqm(graph,
                                    configurations,
                                    decision_variables,
                                    linear_energy_ranges=linear_energy_ranges,
                                    quadratic_energy_ranges=quadratic_energy_ranges)

        for u, bias in bqm.linear.items():
            low, high = linear_energy_ranges[u]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        for (u, v), bias in bqm.quadratic.items():
            if (u, v) in quadratic_energy_ranges:
                low, high = quadratic_energy_ranges[(u, v)]
            else:
                low, high = quadratic_energy_ranges[(v, u)]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_disjoint_two_sided(self):
        graph = nx.complete_graph(6)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, -1): 0,
                          (+1, +1, -1): 0}
        decision_variables = (0, 1, 8)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        bqm, gap = mip.generate_bqm(graph,
                                    configurations,
                                    decision_variables,
                                    linear_energy_ranges=linear_energy_ranges,
                                    quadratic_energy_ranges=quadratic_energy_ranges)

        for u, bias in bqm.linear.items():
            low, high = linear_energy_ranges[u]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        for (u, v), bias in bqm.quadratic.items():
            if (u, v) in quadratic_energy_ranges:
                low, high = quadratic_energy_ranges[(u, v)]
            else:
                low, high = quadratic_energy_ranges[(v, u)]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_disjoint_one_sided(self):
        graph = nx.complete_graph(6)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, +1): 0,
                          (+1, +1, -1): 0}
        decision_variables = (0, 1, 3)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        bqm, gap = mip.generate_bqm(graph,
                                    configurations,
                                    decision_variables,
                                    linear_energy_ranges=linear_energy_ranges,
                                    quadratic_energy_ranges=quadratic_energy_ranges)

        for u, bias in bqm.linear.items():
            low, high = linear_energy_ranges[u]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        for (u, v), bias in bqm.quadratic.items():
            if (u, v) in quadratic_energy_ranges:
                low, high = quadratic_energy_ranges[(u, v)]
            else:
                low, high = quadratic_energy_ranges[(v, u)]
            self.assertLessEqual(bias, high)
            self.assertGreaterEqual(bias, low)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_return_auxiliary_AND_K3(self):

        graph = nx.complete_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        bqm, gap, aux_configs = mip.generate_bqm(graph, configurations, decision_variables,
                                                 return_auxiliary=True)

        # no aux variables
        self.assertEqual(aux_configs,
                         {(-1, -1, -1): {},
                          (-1, +1, -1): {},
                          (+1, -1, -1): {},
                          (+1, +1, +1): {}})

    def test_return_auxiliary_AND_K5(self):

        graph = nx.complete_graph(5)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        bqm, gap, aux_configs = mip.generate_bqm(graph, configurations, decision_variables,
                                                 return_auxiliary=True)

        for config in configurations:
            sample = dict(zip(decision_variables, config))

            sample.update(aux_configs[config])

            self.assertAlmostEqual(bqm.energy(sample), 0.0)

    def test_min_gap_no_aux(self):
        # Verify min_classical_gap parameter works
        # Note: test will run the problem with a gap of 5, where the gap is too high. Lowering
        #   gap to 4 should produce a BQM.

        # Set up problem
        nodes = ['a', 'b']
        states = {(-1, -1): 0,
                  (-1, 1): 0,
                  (1, -1): 0}
        graph = nx.complete_graph(nodes)

        # Run problem with a min_classical_gap that is set too high
        with self.assertRaises(pm.ImpossiblePenaltyModel):
            large_min_gap = 5
            mip.generate_bqm(graph, states, nodes, min_classical_gap=large_min_gap)

        # Run same problem with the min_classical_gap set to a lower threshold
        smaller_min_gap = 4
        bqm, gap = mip.generate_bqm(graph, states, nodes, min_classical_gap=smaller_min_gap)
        self.assertEqual(smaller_min_gap, gap)
        self.check_bqm_table(bqm, gap, states, nodes)

    def test_min_gap_with_auxiliary(self):
        # Verify min_classical_gap parameter works.
        nodes = ['a']
        states = {(-1,): 0}
        graph = nx.complete_graph(nodes + ['aux0'])

        # min_classical_gap should be too large for this problem
        with self.assertRaises(pm.ImpossiblePenaltyModel):
            large_min_gap = 7
            mip.generate_bqm(graph, states, nodes, min_classical_gap=large_min_gap)

        # Lowering the min_classical_gap should result to a bqm being found
        smaller_min_gap = 6
        bqm, gap = mip.generate_bqm(graph, states, nodes, min_classical_gap=smaller_min_gap)
        self.assertEqual(smaller_min_gap, gap)
        self.check_bqm_table(bqm, gap, states, nodes)

    def test_nonzero_ground_state_no_aux(self):
        decision_variables = ('a',)
        graph = nx.complete_graph(decision_variables)
        configurations = {(-1,): -1}

        bqm, gap = mip.generate_bqm(graph, configurations, decision_variables, min_classical_gap=2)

        self.check_bqm_table(bqm, gap, configurations, decision_variables)
        self.check_bqm_graph(bqm, graph)

    def test_multiple_nonzero_feasible_states_no_aux(self):
        nodes = [0, 1, 2]
        graph = nx.complete_graph(nodes)
        configurations = {(+1, +1, -1): -1.5,
                          (+1, -1, +1): -2.5,
                          (+1, -1, -1): -4.5,
                          (-1, -1, +1): -1.5,
                          (-1, -1, -1): 0.5}

        bqm, gap = mip.generate_bqm(graph, configurations, nodes, min_classical_gap=0.75)

        self.check_bqm_table(bqm, gap, configurations, nodes)
        self.check_bqm_graph(bqm, graph)

    def test_multiple_nonzero_feasible_states_with_aux(self):
        """
        nodes = ['a', 'b', 'c']
        graph = nx.complete_graph(nodes)
        configurations = {(+1, +1, -1): -3,
                          (+1, -1, +1): -2,
                          (+1, -1, -1): -6,
                          (-1, +1, -1): 0,
                          (-1, -1, +1): -1,
                          (-1, -1, -1): -1}
        bqm, gap = mip.generate_bqm(graph, configurations, nodes)
        """

        nodes = ['a', 'b']
        graph = nx.complete_graph(nodes + ['aux0'])
        configurations = {(+1, +1): -3,
                          (+1, -1): -4}

        bqm, gap = mip.generate_bqm(graph, configurations, nodes, min_classical_gap=0)
        self.check_bqm_graph(bqm, graph)

    def test_all_possible_config(self):
        """Test when all possible configurations for the decision variable is defined
        """
        nodes = ['a']
        graph = nx.complete_graph(nodes)
        configurations = {(-1,): -2,
                          (+1,): 2}

        bqm, gap = mip.generate_bqm(graph, configurations, nodes)
        self.check_bqm_graph(bqm, graph)
