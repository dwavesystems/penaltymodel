import unittest
import random
import itertools

import networkx as nx

import penaltymodel as pm


class TestSpecification(unittest.TestCase):
    def test_construction_empty(self):
        spec = pm.Specification(nx.Graph(), [], {})
        self.assertEqual(len(spec), 0)

    def test_construction_typical(self):
        # in this case should identify as being binary
        graph = nx.complete_graph(10)
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        self.assertEqual(spec.graph, graph)  # literally the same object
        self.assertEqual(spec.decision_variables, decision_variables)
        self.assertEqual(spec.feasible_configurations, feasible_configurations)
        self.assertIs(spec.vartype, pm.SPIN)

    def test_construction_from_edgelist(self):
        graph = nx.barbell_graph(10, 7)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(-1, -1, -1): 0.}

        # specification from edges
        spec0 = pm.Specification(graph.edges, decision_variables, feasible_configurations)

        # specification from graph
        spec1 = pm.Specification(graph, decision_variables, feasible_configurations)

        self.assertEqual(spec0, spec1)

    def test_energy_ranges_default(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 0): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.BINARY)

        # check default energy ranges
        for v in graph:
            self.assertEqual(spec.linear_energy_ranges[v], (-2, 2))
        for edge in graph.edges:
            self.assertEqual(spec.quadratic_energy_ranges[edge], (-1, 1))

    def test_energy_ranges_specified(self):
        graph = nx.barbell_graph(4, 16)
        decision_variables = (0, 4, 3)
        feasible_configurations = {(0, 0, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations,
                                linear_energy_ranges={v: (-2, 2) for v in graph},
                                quadratic_energy_ranges={edge: (-1, 1) for edge in graph.edges},
                                vartype=pm.BINARY)

        # check default energy ranges
        for v in graph:
            self.assertEqual(spec.linear_energy_ranges[v], (-2, 2))
        for edge in graph.edges:
            self.assertEqual(spec.quadratic_energy_ranges[edge], (-1, 1))

    def test_vartype_specified(self):
        graph = nx.complete_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)
        self.assertIs(spec.vartype, pm.SPIN)

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.BINARY)
        self.assertIs(spec.vartype, pm.BINARY)

        # now set up a spec that can only have one vartype
        graph = nx.complete_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, -1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)
        self.assertIs(spec.vartype, pm.SPIN)

        # the feasible_configurations are spin
        with self.assertRaises(ValueError):
            spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.BINARY)

    def test_vartype_default(self):
        graph = nx.complete_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}

        spec = pm.Specification(graph, decision_variables, feasible_configurations)
        self.assertIs(spec.vartype, pm.SPIN)

    def test_relabel_typical(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        mapping = dict(enumerate('abcdefghijklmnopqrstuvwxyz'))

        new_spec = spec.relabel_variables(mapping)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations)

        self.assertEqual(new_spec, test_spec)

    def test_relabel_copy(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        mapping = dict(enumerate('abcdefghijklmnopqrstuvwxyz'))

        new_spec = spec.relabel_variables(mapping, copy=True)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations)

        self.assertEqual(new_spec, test_spec)

    def test_relabel_inplace(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        mapping = {i: v for i, v in enumerate('abcdefghijklmnopqrstuvwxyz') if i in graph}

        new_spec = spec.relabel_variables(mapping, copy=False)

        self.assertIs(new_spec, spec)  # should be the same object
        self.assertIs(new_spec.graph, spec.graph)

        # create a test spec
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = (mapping[v] for v in decision_variables)
        test_spec = pm.Specification(graph, decision_variables, feasible_configurations)

        self.assertEqual(new_spec, test_spec)

    def test_relabel_inplace_identity(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        mapping = {v: v for v in graph}

        new_spec = spec.relabel_variables(mapping, copy=False)

    def test_relabel_inplace_overlap(self):
        graph = nx.circular_ladder_graph(12)
        decision_variables = (0, 2, 5)
        feasible_configurations = {(1, 1, 1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        mapping = {v: v + 5 for v in graph}

        new_spec = spec.relabel_variables(mapping, copy=False)


class TestPenaltyModel(unittest.TestCase):
    def test_construction(self):

        # build a specification
        graph = nx.complete_graph(10)
        decision_variables = (0, 4, 5)
        feasible_configurations = {(-1, -1, -1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)

        # build a model
        model = pm.BinaryQuadraticModel({v: 0 for v in graph},
                                        {edge: 0 for edge in graph.edges},
                                        0.0,
                                        vartype=pm.Vartype.SPIN)

        # build a PenaltyModel explicitly
        pm0 = pm.PenaltyModel(graph, decision_variables, feasible_configurations, model, .1, 0)

        # build from spec
        pm1 = pm.PenaltyModel.from_specification(spec, model, .1, 0)

        # should result in equality
        self.assertEqual(pm0, pm1)

    def test_relabel(self):
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # now set up the same widget with 0 relabelled to 'a'
        graph = nx.path_graph(3)
        graph = nx.relabel_nodes(graph, {0: 'a'})
        decision_variables = ('a', 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        test_widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # without copy
        new_widget = widget.relabel_variables({0: 'a'}, copy=True)
        self.assertEqual(test_widget, new_widget)

        widget.relabel_variables({0: 'a'}, copy=False)
        self.assertEqual(widget, test_widget)
