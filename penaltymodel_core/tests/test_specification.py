# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import itertools

import networkx as nx

import penaltymodel.core as pm

import dimod


class TestSpecification(unittest.TestCase):
    def test_construction_empty(self):
        spec = pm.Specification(nx.Graph(), [], {}, dimod.SPIN)
        self.assertEqual(len(spec), 0)

    def test_construction_typical(self):
        graph = nx.complete_graph(10)
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        self.assertEqual(spec.graph, graph)  # literally the same object
        self.assertEqual(spec.decision_variables, decision_variables)
        self.assertEqual(spec.feasible_configurations, feasible_configurations)
        self.assertIs(spec.vartype, dimod.SPIN)

    def test_construction_from_edgelist(self):
        graph = nx.barbell_graph(10, 7)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(-1, -1, -1): 0.}

        # specification from edges
        spec0 = pm.Specification(graph.edges, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        # specification from graph
        spec1 = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        self.assertEqual(spec0, spec1)

    def test_construction_bad_graph(self):
        graph = 1
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}

        with self.assertRaises(TypeError):
            pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

    def test_ranges_default(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 0): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)

        for v in graph:
            self.assertEqual(spec.ising_linear_ranges[v], [-2, 2])

        for u, v in graph.edges:
            self.assertEqual(spec.ising_quadratic_ranges[u][v], [-1, 1])
            self.assertEqual(spec.ising_quadratic_ranges[v][u], [-1, 1])

    def test_linear_ranges_specified(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations,
                                ising_linear_ranges={v: [-v, 2] for v in graph},
                                vartype=dimod.BINARY)

        # check default energy ranges
        for v in graph:
            self.assertEqual(spec.ising_linear_ranges[v], [-v, 2])

        spec = pm.Specification(graph, decision_variables, feasible_configurations,
                                ising_linear_ranges={v: (-v, 2) for v in graph},
                                vartype=dimod.BINARY)

        # check default energy ranges
        for v in graph:
            self.assertEqual(spec.ising_linear_ranges[v], [-v, 2])

    def test_quadratic_ranges_partially_specified(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations,
                                ising_quadratic_ranges={0: {1: [0, 1], 2: [-1, 0]}, 2: {0: [-1, 0]}},
                                vartype=dimod.BINARY)

        ising_quadratic_ranges = spec.ising_quadratic_ranges
        for u in ising_quadratic_ranges:
            for v in ising_quadratic_ranges[u]:
                self.assertIs(ising_quadratic_ranges[u][v], ising_quadratic_ranges[v][u])
        for u, v in graph.edges:
            self.assertIn(v, ising_quadratic_ranges[u])
            self.assertIn(u, ising_quadratic_ranges[v])

        self.assertEqual(ising_quadratic_ranges[0][1], [0, 1])

    def test_linear_ranges_bad(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 1): 0.}

        with self.assertRaises(ValueError):
            pm.Specification(graph, decision_variables, feasible_configurations,
                             ising_linear_ranges={v: [-v, 'a'] for v in graph},
                             vartype=dimod.BINARY)

        with self.assertRaises(TypeError):
            pm.Specification(graph, decision_variables, feasible_configurations,
                             ising_linear_ranges={v: [-v, 1, 1] for v in graph},
                             vartype=dimod.BINARY)

    def test_vartype_specified(self):
        graph = nx.complete_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        self.assertIs(spec.vartype, dimod.SPIN)

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)
        self.assertIs(spec.vartype, dimod.BINARY)

        # now set up a spec that can only have one vartype
        graph = nx.complete_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, -1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        self.assertIs(spec.vartype, dimod.SPIN)

        # the feasible_configurations are spin
        with self.assertRaises(ValueError):
            spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)

    def test_relabel_typical(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = dict(enumerate('abcdefghijklmnopqrstuvwxyz'))

        new_spec = spec.relabel_variables(mapping, inplace=False)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        self.assertEqual(new_spec, test_spec)
        self.assertEqual(new_spec.ising_linear_ranges, test_spec.ising_linear_ranges)
        self.assertEqual(new_spec.ising_quadratic_ranges, test_spec.ising_quadratic_ranges)

    def test_relabel_copy(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = dict(enumerate('abcdefghijklmnopqrstuvwxyz'))

        new_spec = spec.relabel_variables(mapping, inplace=False)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        self.assertEqual(new_spec, test_spec)
        self.assertEqual(new_spec.ising_linear_ranges, test_spec.ising_linear_ranges)
        self.assertEqual(new_spec.ising_quadratic_ranges, test_spec.ising_quadratic_ranges)

    def test_relabel_inplace(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = {i: v for i, v in enumerate('abcdefghijklmnopqrstuvwxyz') if i in graph}

        new_spec = spec.relabel_variables(mapping, inplace=True)

        self.assertIs(new_spec, spec)  # should be the same object
        self.assertIs(new_spec.graph, spec.graph)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        self.assertEqual(new_spec, test_spec)
        self.assertEqual(new_spec.ising_linear_ranges, test_spec.ising_linear_ranges)
        self.assertEqual(new_spec.ising_quadratic_ranges, test_spec.ising_quadratic_ranges)

    def test_relabel_inplace_identity(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = {v: v for v in graph}

        new_spec = spec.relabel_variables(mapping, inplace=True)

    def test_relabel_inplace_overlap(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = {v: v + 5 for v in graph}

        new_spec = spec.relabel_variables(mapping, inplace=True)

    def test_relabel_forwards_and_backwards(self):
        graph = nx.path_graph(4)
        graph.add_edge(0, 2)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        # make another one
        graph = nx.path_graph(4)
        graph.add_edge(0, 2)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        original_spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        identity = {v: v for v in graph}

        new_label_sets = [(10, 1),
                          ('a', 'b'),
                          (1, 'b'),
                          ('1', '2', '3', '4'),
                          ('a', 'b', 'c', 'd')]
        new_label_sets.extend(itertools.permutations(graph))

        for new in new_label_sets:
            mapping = dict(enumerate(new))
            inv_mapping = {u: v for v, u in mapping.items()}

            # apply then invert with inplace=False
            copy_spec = spec.relabel_variables(mapping, inplace=False)
            inv_copy = copy_spec.relabel_variables(inv_mapping, inplace=False)
            self.assertEqual(inv_copy, original_spec)
            self.assertEqual(inv_copy.ising_linear_ranges, original_spec.ising_linear_ranges)
            self.assertEqual(inv_copy.ising_quadratic_ranges, original_spec.ising_quadratic_ranges)

            # apply then invert with inplace=True
            spec.relabel_variables(mapping, inplace=True)
            if mapping == identity:
                self.assertEqual(spec, original_spec)
            else:
                self.assertNotEqual(spec, original_spec)
            spec.relabel_variables(inv_mapping, inplace=True)
            self.assertEqual(spec, original_spec)
            self.assertEqual(spec.ising_linear_ranges, original_spec.ising_linear_ranges)
            self.assertEqual(spec.ising_quadratic_ranges, original_spec.ising_quadratic_ranges)

    def test_bad_relabel(self):
        graph = nx.path_graph(4)
        graph.add_edge(0, 2)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        mapping = {0: 2, 1: 1}

        with self.assertRaises(ValueError):
            spec.relabel_variables(mapping, inplace=False)

        with self.assertRaises(ValueError):
            spec.relabel_variables(mapping, inplace=True)
