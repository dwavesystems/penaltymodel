# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# developer note: this combines all of the tests from maxgap and mip
# before we merged. There is likely a lot of redundancy

import itertools
import unittest

import dimod
import networkx as nx

from penaltymodel.generation import generate, ImpossiblePenaltyModel
from penaltymodel.utils import table_to_sampleset

MAX_GAP_DELTA = 0.01


class TestGenerate(unittest.TestCase):
    def check_bqm_table(self, bqm, gap, table, decision):
        """check that the bqm has ground states matching table"""
        response = dimod.ExactSolver().sample(bqm)

        highest_feasible_energy = max(table.values()) if isinstance(table, dict) else 0

        # Looping though ExactSolver results and comparing with BQM
        seen_gap = float('inf')
        seen_table = set()
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)  # sanity check

            config = tuple(sample[v] for v in decision)

            # Configurations with energies < highest feasible energy
            if energy < highest_feasible_energy + .001:
                self.assertIn(config, table)

                # If there's no specific target energy to match, we've already
                # satisfied the configuration by being less than the highest
                # feasible energy
                if not isinstance(table, dict):
                    seen_table.add(config)
                    continue

                # Check configuration against specific target energy
                # Note: For a given valid decision configuration, there could
                #   be different sets of decision + auxiliary configurations. We
                #   only need one of those sets to match the specific target
                #   energy, while the remaining sets can be above that target.
                self.assertGreaterEqual(round(energy - table[config], ndigits=7), 0)

                # If configuration matches target energy, the configuration
                # should be added to the seen_table
                if round(table[config]-energy, ndigits=7) == 0:
                    seen_table.add(config)

            # Get smallest gap among non-table configurations
            elif config not in table:
                seen_gap = min(seen_gap, energy - highest_feasible_energy)

        # Verify that all table configurations have been accounted for
        for config in table:
            self.assertIn(config, seen_table)

        self.assertAlmostEqual(seen_gap, gap)
        self.assertGreater(round(gap, ndigits=7), 0)

    def check_bqm_graph(self, bqm, graph):
        """bqm and graph have the same structure"""
        self.assertEqual(len(bqm.linear), len(graph.nodes))
        self.assertEqual(len(bqm.quadratic), len(graph.edges))

        for v in bqm.linear:
            self.assertIn(v, graph)
        for u, v in bqm.quadratic:
            self.assertIn(u, graph.adj[v])

    def generate_and_check(self, graph, configurations, decision_variables,
                           *, known_classical_gap=0, **kwargs):

        bqm, gap, aux = generate(graph, table_to_sampleset(configurations, decision_variables), **kwargs)

        min_classical_gap = kwargs.get('min_classical_gap', 2)

        # Check gap
        # Note: Due to the way MaxGap searches for the maximum gap, if
        #   known_classical_gap == "maximum possible gap", then `gap` can be
        #   slightly smaller than known_classical_gap.
        self.assertGreaterEqual(round(gap, 9), min_classical_gap)
        self.assertGreaterEqual(gap, known_classical_gap - MAX_GAP_DELTA)

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

        # Get highest feasible energy
        if isinstance(configurations, dict) and configurations:
            highest_feasible_energy = max(configurations.values())
        else:
            highest_feasible_energy = 0

        # check gap and other energies
        best_gap = float('inf')
        seen = set()
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

        min_lin, max_lin = kwargs.get('linear_bound', (-2, 2))
        min_quad, max_quad = kwargs.get('quadratic_bound', (-1, 1))

        # check energy ranges
        for v, bias in bqm.linear.items():
            self.assertGreaterEqual(round(bias, 9), min_lin)
            self.assertLessEqual(round(bias, 9), max_lin)

        for (u, v), bias in bqm.quadratic.items():
            self.assertGreaterEqual(round(bias, 9), min_quad)
            self.assertLessEqual(round(bias, 9), max_quad)

        self.assertAlmostEqual(best_gap, gap)

    def test_disjoint(self):
        graph = nx.Graph()
        for u, v in itertools.product([0, 1, 2], [3, 4, 5]):
            graph.add_edge(u, v)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, -1): 0,
                          (+1, +1, -1): 0}
        decision_variables = (0, 1, 8)

        self.generate_and_check(graph, configurations, decision_variables)

    def test_disjoint_decision_accross_subgraphs(self):
        graph = nx.Graph()
        for u, v in itertools.product([0, 1, 2], [3, 4, 5]):
            graph.add_edge(u, v)
        graph.add_edge(8, 9)

        configurations = {(-1, -1, +1, -1): 0,
                          (+1, +1, -1, -1): 0}
        decision_variables = (0, 1, 3, 8)

        self.generate_and_check(graph, configurations, decision_variables)

    def test_empty(self):
        # this should test things like empty graphs and empty configs
        graph = nx.Graph()
        configurations = {}
        decision_variables = tuple()

        self.generate_and_check(graph, configurations, decision_variables)

    def test_impossible(self):
        graph = nx.path_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        with self.assertRaises(ImpossiblePenaltyModel):
            generate(graph, table_to_sampleset(configurations, decision_variables))

    def test_K1(self):
        graph = nx.complete_graph(1)
        configurations = {(+1,): 0}
        decision_variables = [0]
        self.generate_and_check(graph, configurations, decision_variables)

    def test_K1_multiple_energies(self):
        graph = nx.complete_graph(1)
        configurations = {(+1,): .1, (-1,): -.3}
        decision_variables = [0]

        self.generate_and_check(graph, configurations, decision_variables)

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

        self.generate_and_check(graph, configurations, decision_variables)

    def test_K3_one_aux(self):
        graph = nx.complete_graph(3)
        configurations = {(-1, -1): 0, (1, 1): 0}
        decision_variables = [0, 1]

        self.generate_and_check(graph, configurations, decision_variables)

    def test_K4(self):
        graph = nx.complete_graph(4)

        configurations = {(-1, -1, -1, -1): 0, (1, 1, 1, 1): 0}
        decision_variables = list(graph)

        self.generate_and_check(graph, configurations, decision_variables)

    def test_min_gap_equals_max_gap(self):
        # Make sure that a model is always grabbed, even when min_gap == max_gap
        min_gap = 4     # This value is also the max classical gap
        decision_variables = ['a']
        config = {(-1,): -1}
        graph = nx.complete_graph(decision_variables)

        bqm, gap, aux = generate(graph, table_to_sampleset(config, decision_variables),
                                 min_classical_gap=min_gap)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel({'a': 2}, {}, 1, 'SPIN'))
        self.assertEqual(min_gap, gap)

    def test_min_gap_no_aux(self):
        # Set up problem
        decision_variables = ['a', 'b', 'c']
        or_gate = {(-1, -1, -1): 0,
                   (-1, 1, 1): 0,
                   (1, -1, 1): 0,
                   (1, 1, 1): 0}
        graph = nx.complete_graph(decision_variables)

        # Run problem with a min_classical_gap that is too large
        with self.assertRaises(ImpossiblePenaltyModel):
            large_min_gap = 3
            generate(graph, table_to_sampleset(or_gate, decision_variables),
                     min_classical_gap=large_min_gap)

        # Lowering min_classical_gap should lead to a bqm being found
        min_classical_gap = 1.5
        self.generate_and_check(graph, or_gate, decision_variables,
                                min_classical_gap=min_classical_gap)

    def test_min_gap_with_aux(self):
        decision_variables = ['a', 'b', 'c']
        xor_gate = {(-1, -1, -1): 0,
                    (-1, 1, 1): 0,
                    (1, -1, 1): 0,
                    (1, 1, -1): 0}
        graph = nx.complete_graph(decision_variables + ['aux0'])

        # Run problem with a min_classical_gap that is too large
        with self.assertRaises(ImpossiblePenaltyModel):
            large_min_gap = 3
            generate(graph, table_to_sampleset(xor_gate, decision_variables),
                     min_classical_gap=large_min_gap)

        # Lowering min_classical_gap should lead to a bqm being found
        min_classical_gap = .5
        self.generate_and_check(graph, xor_gate, decision_variables,
                                min_classical_gap=min_classical_gap)

    def test_negative_min_gap_feasible_bqm(self):
        # Regardless of the negative min classical gap, this feasible BQM should return its usual
        # max classical gap.
        decision_variables = ['a']
        configurations = {(-1,): 0}
        graph = nx.complete_graph(decision_variables)
        min_classical_gap = -2

        self.generate_and_check(graph, configurations, decision_variables,
                                min_classical_gap=min_classical_gap)

    # def test_negative_min_gap_impossible_bqm(self):
    #     """XOR Gate problem without auxiliary variables
    #     Note: Regardless of the negative gap, this BQM should remain impossible.
    #     """
    #     negative_gap = -3
    #     decision_variables = ['a', 'b', 'c']
    #     xor_gate = {(-1, -1, -1): 0,
    #                 (-1, 1, 1): 0,
    #                 (1, -1, 1): 0,
    #                 (1, 1, -1): 0}
    #     graph = nx.complete_graph(decision_variables)

    #     with self.assertRaises(ImpossiblePenaltyModel):
    #         generate(graph, xor_gate, decision_variables,
    #                  min_classical_gap=negative_gap)

    def test_restricted_energy_ranges(self):
        """Create asymmetric energy ranges and test against that."""
        graph = nx.Graph()
        for u, v in itertools.product([0, 1, 2], [3, 4, 5]):
            graph.add_edge(u, v)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)
        linear_bound = (-1., 2.)
        quadratic_bound = (-1, .5)

        self.generate_and_check(graph, configurations, decision_variables,
                                linear_bound=linear_bound,
                                quadratic_bound=quadratic_bound,
                                )

    def test_negative_feasible_positive_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state has negative energy and the gap is set high enough that
        the infeasible states must have positive energy.
        """
        min_classical_gap = 0.5
        decision_variables = ['a']
        configurations = {(1,): -0.5}
        graph = nx.complete_graph(decision_variables)

        # Known solution: -2*a + 1.5
        known_classical_gap = 4

        self.generate_and_check(graph, configurations, decision_variables,
                                known_classical_gap=known_classical_gap)

    def test_positive_feasible_positive_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state and the infeasible states must have positive energy.
        """
        min_classical_gap = 1
        decision_variables = ['a', 'b']
        configurations = {(1, -1): 4,
                          (-1, 1): 4,
                          (-1, -1): 0}
        graph = nx.complete_graph(decision_variables + ['c'])

        # Known solution: 2*a + 2*b -2*c + a*b + a*c + b*c + 7
        known_classical_gap = 8

        self.generate_and_check(graph, configurations, decision_variables,
                                min_classical_gap=min_classical_gap,
                                known_classical_gap=known_classical_gap)

    def test_negative_feasible_negative_infeasible(self):
        """Testing that gap is wrt the energy of the highest feasible state, rather than wrt zero.

        Case where highest feasible state and the infeasible states have negative energy levels.
        """
        min_classical_gap = 0.5
        decision_variables = ['a']
        configurations = {(1,): -10}
        graph = nx.complete_graph(decision_variables + ['b', 'c'])


        # Known solution: -2*a - 2*b - 2*c - a*b - a*c - b*c - 1
        known_classical_gap = 8

        self.generate_and_check(graph, configurations, decision_variables,
                                min_classical_gap=min_classical_gap,
                                known_classical_gap=known_classical_gap)

    def test_impossible_AND_3path(self):
        """AND gate cannot exist on a 3-path"""
        graph = nx.path_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        with self.assertRaises(ImpossiblePenaltyModel):
            generate(graph, table_to_sampleset(configurations, decision_variables))

    def test_empty_no_aux(self):
        graph = nx.Graph()
        sample = {}

        bqm, gap, aux = generate(graph, sample)

        self.check_bqm_graph(bqm, graph)

    def test_empty_some_aux(self):
        graph = nx.complete_graph(3)
        bqm, gap, aux = generate(graph, {})

        self.check_bqm_graph(bqm, graph)

    def test_AND_K4(self):
        """A typical use case, an AND gate on a K4."""
        graph = nx.complete_graph(4)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        bqm, gap, aux = generate(graph, table_to_sampleset(configurations, decision_variables))

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_NAE3SAT_4cycle(self):
        """A typical use case, an AND gate on a K4."""
        graph = nx.cycle_graph(4)
        configurations = {config: 0 for config in itertools.product((-1, 1), repeat=3) if len(set(config)) > 1}
        decision_variables = (0, 1, 2)

        bqm, gap, aux = generate(graph, table_to_sampleset(configurations, decision_variables))

        self.check_bqm_table(bqm, gap, configurations, decision_variables)

    def test_return_auxiliary_AND_K3(self):

        graph = nx.complete_graph(3)
        configurations = {(-1, -1, -1): 0,
                          (-1, +1, -1): 0,
                          (+1, -1, -1): 0,
                          (+1, +1, +1): 0}
        decision_variables = (0, 1, 2)

        bqm, gap, aux_configs = generate(graph, table_to_sampleset(configurations, decision_variables))

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

        bqm, gap, aux_configs = generate(graph, table_to_sampleset(configurations, decision_variables))

        for config in configurations:
            sample = dict(zip(decision_variables, config))

            sample.update(aux_configs[config])

            self.assertAlmostEqual(bqm.energy(sample), 0.0)
