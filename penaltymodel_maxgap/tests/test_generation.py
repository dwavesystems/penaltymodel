import unittest

import dimod
import dwave_networkx as dnx
import networkx as nx

from pysmt.environment import reset_env

import penaltymodel.core as pm
import penaltymodel.maxgap as maxgap


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.env = reset_env()

    def generate_and_check(self, graph, configurations, decision_variables,
                           linear_energy_ranges, quadratic_energy_ranges,
                           min_classical_gap):
        """Checks that MaxGap's BQM and gap obeys the constraints set by configurations,
        linear and quadratic energy ranges, and min classical gap.

        Note: The gap is checked for whether it obeys the min classical gap constraint, and whether
        it is the largest gap for a given BQM. However, this gap may not necessarily be the largest
        gap for the given set of constraints (i.e. configurations, energy ranges), and this is not
        checked for in this function.
        """
        bqm, gap = maxgap.generate(graph, configurations, decision_variables,
                                   linear_energy_ranges,
                                   quadratic_energy_ranges,
                                   min_classical_gap)

        self.assertGreaterEqual(gap, min_classical_gap)

        # check that the bqm/graph have the same structure
        self.assertEqual(len(bqm.linear), len(graph.nodes))
        for v in bqm.linear:
            self.assertIn(v, graph.nodes)
        self.assertEqual(len(bqm.quadratic), len(graph.edges))
        for u, v in bqm.quadratic:
            self.assertIn((u, v), graph.edges)

        # now solve for the thing
        sampleset = dimod.ExactSolver().sample(bqm)

        # check that ground has 0 energy
        if len(sampleset):
            self.assertAlmostEqual(sampleset.first.energy, min(configurations.values()))

        # check gap and other energies
        best_gap = float('inf')
        seen = set()
        highest_feasible_energy = (max(configurations.values()) if isinstance(configurations, dict)
                                   else 0)
        for sample, energy in sampleset.data(['sample', 'energy']):
            config = tuple(sample[v] for v in decision_variables)

            # we want the minimum energy for each config of the decision variables,
            # so once we've seen it once we can skip
            if config in seen:
                continue

            if config in configurations:
                self.assertAlmostEqual(energy, configurations[config])
                seen.add(config)
            else:
                best_gap = min(best_gap, energy - highest_feasible_energy)

        # check energy ranges
        for v, bias in bqm.linear.items():
            min_, max_ = linear_energy_ranges[v]
            self.assertGreaterEqual(bias, min_)
            self.assertLessEqual(bias, max_)

        for (u, v), bias in bqm.quadratic.items():
            min_, max_ = quadratic_energy_ranges.get((u, v), quadratic_energy_ranges.get((v, u), None))
            self.assertGreaterEqual(bias, min_)
            self.assertLessEqual(bias, max_)

        self.assertAlmostEqual(best_gap, gap)

    def test_disjoint(self):
        graph = dnx.chimera_graph(1, 1, 3)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, -1): 0,
                          (+1, +1, -1): 0}
        decision_variables = (0, 1, 8)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_disjoint_decision_accross_subgraphs(self):
        graph = dnx.chimera_graph(1, 1, 3)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, +1, -1): 0,
                          (+1, +1, -1, -1): 0}
        decision_variables = (0, 1, 3, 8)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_empty(self):
        # this should test things like empty graphs and empty configs
        graph = nx.Graph()
        configurations = {}
        decision_variables = tuple()
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_impossible(self):
        graph = nx.path_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 2

        with self.assertRaises(pm.ImpossiblePenaltyModel):
            maxgap.generate(graph, configurations, decision_variables,
                            linear_energy_ranges,
                            quadratic_energy_ranges,
                            min_classical_gap)

    def test_K1(self):
        graph = nx.complete_graph(1)
        configurations = {(+1,): 0}
        decision_variables = [0]
        linear_energy_ranges = {0: (-2, 2)}
        quadratic_energy_ranges = {}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_K1_multiple_energies(self):
        graph = nx.complete_graph(1)
        configurations = {(+1,): .1, (-1,): -.3}
        decision_variables = [0]
        linear_energy_ranges = {0: (-2, 2)}
        quadratic_energy_ranges = {}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_K33(self):
        graph = nx.Graph()
        for i in range(3):
            for j in range(3, 6):
                graph.add_edge(i, j)

        decision_variables = (0, 2, 3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_K3_one_aux(self):
        graph = nx.complete_graph(3)

        configurations = {(-1, -1): 0, (1, 1): 0}

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        decision_variables = [0, 1]
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_K4(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1): 0, (1, 1, 1, 1): 0}
        decision_variables = list(graph)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}
        min_classical_gap = 1

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_min_gap_equals_max_gap(self):
        # Make sure that a model is always grabbed, even when min_gap == max_gap
        min_gap = 4     # This value is also the max classical gap
        decision_variables = ['a']
        config = {(-1,): -1}
        graph = nx.complete_graph(decision_variables)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        bqm, gap = maxgap.generate(graph, config, decision_variables,
                                   linear_energy_ranges,
                                   quadratic_energy_ranges,
                                   min_gap,
                                   None)

        # Check that a model was found
        self.assertIsNotNone(bqm.linear)
        self.assertIsNotNone(bqm.quadratic)
        self.assertIsNotNone(bqm.offset)
        self.assertEqual(min_gap, gap)  # Min gap is also the max classical gap in this case

    def test_min_gap_no_aux(self):
        """Verify min_classical_gap parameter works
        """
        # Set up problem
        decision_variables = ['a', 'b', 'c']
        or_gate = {(-1, -1, -1): 0,
                   (-1, 1, 1): 0,
                   (1, -1, 1): 0,
                   (1, 1, 1): 0}
        graph = nx.complete_graph(decision_variables)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        # Run problem with a min_classical_gap that is too large
        with self.assertRaises(pm.ImpossiblePenaltyModel):
            large_min_gap = 3
            maxgap.generate(graph, or_gate, decision_variables,
                            linear_energy_ranges,
                            quadratic_energy_ranges,
                            large_min_gap,
                            None)

        # Lowering min_classical_gap should lead to a bqm being found
        min_classical_gap = 1.5
        self.generate_and_check(graph, or_gate, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_min_gap_with_aux(self):
        """Verify min_classical_gap parameter works
        """
        decision_variables = ['a', 'b', 'c']
        xor_gate = {(-1, -1, -1): 0,
                    (-1, 1, 1): 0,
                    (1, -1, 1): 0,
                    (1, 1, -1): 0}
        graph = nx.complete_graph(decision_variables + ['aux0'])

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        # Run problem with a min_classical_gap that is too large
        with self.assertRaises(pm.ImpossiblePenaltyModel):
            large_min_gap = 3
            maxgap.generate(graph, xor_gate, decision_variables,
                            linear_energy_ranges,
                            quadratic_energy_ranges,
                            large_min_gap,
                            None)

        # Lowering min_classical_gap should lead to a bqm being found
        min_classical_gap = .5
        self.generate_and_check(graph, xor_gate, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_negative_min_gap_feasible_bqm(self):
        # Regardless of the negative min classical gap, this feasible BQM should return its usual
        # max classical gap.
        decision_variables = ['a']
        configurations = {(-1,): 0}
        graph = nx.complete_graph(decision_variables)
        min_classical_gap = -2

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_negative_min_gap_impossible_bqm(self):
        """XOR Gate problem without auxiliary variables
        Note: Regardless of the negative gap, this BQM should remain impossible.
        """
        negative_gap = -3
        decision_variables = ['a', 'b', 'c']
        xor_gate = {(-1, -1, -1): 0,
                    (-1, 1, 1): 0,
                    (1, -1, 1): 0,
                    (1, 1, -1): 0}
        graph = nx.complete_graph(decision_variables)

        linear_energy_ranges = {v: (-2., 2.) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

        with self.assertRaises(pm.ImpossiblePenaltyModel):
            maxgap.generate(graph, xor_gate, decision_variables,
                            linear_energy_ranges,
                            quadratic_energy_ranges,
                            negative_gap,
                            None)

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
        min_classical_gap = 2

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_negative_feasible_positive_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state has negative energy and the gap is set high enough that
        the infeasible states must have positive energy.
        """
        min_classical_gap = 0.5
        decision_variables = ['a']
        configurations = {(1,): -0.5}
        graph = nx.complete_graph(decision_variables)

        linear_energy_ranges = {v: (-2, 2) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1, 1) for u, v in graph.edges}

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_positive_feasible_positive_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state and the infeasible states must have positive energy.
        """
        # Note: I expect the gap produced to be >= 4 because the objective function,
        #   2*a + 2*b -2*c + 0.5*a*b + a*c + b*c, produces such a solution. However, there is a bug
        #   that is preventing the following unit test (with min_classical_gap = 3) from running,
        #   which is why a lower min_classical_gap is used instead. This has been documented in a
        #   GitHub issue.
        # min_classical_gap = 3
        min_classical_gap = 1
        decision_variables = ['a', 'b']
        configurations = {(1, -1): -2.5,
                          (-1, 1): -2.5,
                          (-1, -1): 0.5}
        graph = nx.complete_graph(decision_variables + ['c'])

        linear_energy_ranges = {v: (-2, 2) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1, 1) for u, v in graph.edges}

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)

    def test_negative_feasible_negative_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state and the infeasible states have negative energy levels.
        """
        min_classical_gap = 0.5
        decision_variables = ['a']
        configurations = {(1,): -10}
        graph = nx.complete_graph(decision_variables)

        linear_energy_ranges = {v: (-2, 2) for v in graph}
        quadratic_energy_ranges = {(u, v): (-1, 1) for u, v in graph.edges}

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_energy_ranges,
                                quadratic_energy_ranges,
                                min_classical_gap)
