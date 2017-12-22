import unittest
import random
import itertools

import networkx as nx

import penaltymodel as pm


class TestPenaltyModel(unittest.TestCase):
    def test_construction(self):

        # build a specification
        graph = nx.complete_graph(10)
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)

        # build a model
        model = pm.BinaryQuadraticModel({v: 0 for v in graph},
                                        {edge: 0 for edge in graph.edges},
                                        0.0,
                                        vartype=pm.Vartype.SPIN)

        # build a PenaltyModel explicitly
        pm0 = pm.PenaltyModel(graph, decision_variables, feasible_configurations, pm.SPIN, model, .1, 0)

        # build from spec
        pm1 = pm.PenaltyModel.from_specification(spec, model, .1, 0)

        # should result in equality
        self.assertEqual(pm0, pm1)

    def test_relabel(self):
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # now set up the same widget with 0 relabelled to 'a'
        graph = nx.path_graph(3)
        graph = nx.relabel_nodes(graph, {0: 'a'})
        decision_variables = ('a', 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        test_widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # without copy
        new_widget = widget.relabel_variables({0: 'a'}, copy=True)
        self.assertEqual(test_widget, new_widget)

        widget.relabel_variables({0: 'a'}, copy=False)
        self.assertEqual(widget, test_widget)

    def test_bad_energy_range(self):
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)

        linear = {v: -3 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        with self.assertRaises(ValueError):
            widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 5 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        with self.assertRaises(ValueError):
            widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)
