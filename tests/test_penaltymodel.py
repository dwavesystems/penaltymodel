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
import random
import itertools

import networkx as nx

import penaltymodel.core as pm

import dimod


class TestPenaltyModel(unittest.TestCase):
    def test_construction(self):

        # build a specification
        graph = nx.complete_graph(10)
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        # build a model
        model = dimod.BinaryQuadraticModel({v: 0 for v in graph},
                                           {edge: 0 for edge in graph.edges},
                                           0.0,
                                           vartype=dimod.SPIN)

        # build a PenaltyModel explicitly
        pm0 = pm.PenaltyModel(graph, decision_variables, feasible_configurations, dimod.SPIN, model, .1, 0)

        # build from spec
        pm1 = pm.PenaltyModel.from_specification(spec, model, .1, 0)

        # should result in equality
        self.assertEqual(pm0, pm1)

    def test_relabel(self):
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # now set up the same widget with 0 relabelled to 'a'
        graph = nx.path_graph(3)
        graph = nx.relabel_nodes(graph, {0: 'a'})
        decision_variables = ('a', 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        test_widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # without copy
        new_widget = widget.relabel_variables({0: 'a'}, inplace=False)
        self.assertEqual(test_widget, new_widget)
        self.assertEqual(new_widget.decision_variables, ('a', 2))

        widget.relabel_variables({0: 'a'}, inplace=True)
        self.assertEqual(widget, test_widget)
        self.assertEqual(widget.decision_variables, ('a', 2))

    def test_bad_energy_range(self):
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        linear = {v: -3 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        with self.assertRaises(ValueError):
            widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 5 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        with self.assertRaises(ValueError):
            widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

    def test_relabel_forwards_and_backwards(self):
        graph = nx.path_graph(4)
        graph.add_edge(0, 2)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # make another one
        graph = nx.path_graph(4)
        graph.add_edge(0, 2)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        original_widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        new_label_sets = [(10, 1),
                          ('a', 'b'),
                          (1, 'b'),
                          ('1', '2', '3', '4'),
                          ('a', 'b', 'c', 'd')]
        new_label_sets.extend(itertools.permutations(graph))

        for new in new_label_sets:
            mapping = dict(enumerate(new))
            inv_mapping = {u: v for v, u in mapping.items()}

            # apply then invert with copy=False
            widget.relabel_variables(mapping, inplace=True)
            widget.relabel_variables(inv_mapping, inplace=True)
            self.assertEqual(widget, original_widget)

            # apply then invert with copy=True
            copy_widget = widget.relabel_variables(mapping, inplace=False)
            inv_copy = copy_widget.relabel_variables(inv_mapping, inplace=False)
            self.assertEqual(inv_copy, original_widget)

    def test_bqm_and_graph_label_matching(self):
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel({'a': -1, 'b': 0}, {('c', 'a'): 0}, 0,
                                         vartype)
        g1 = nx.complete_graph(['a', 1, 2])
        g2 = nx.complete_graph(['a', 'b', 'c'])

        with self.assertRaises(ValueError):
            pm.PenaltyModel(g1, ['a'], {(0, )}, vartype, bqm, 2, 0)

        pm.PenaltyModel(g2, ['a'], {(0, )}, vartype, bqm, 2, 0)
