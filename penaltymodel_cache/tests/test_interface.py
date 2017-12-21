import unittest
import os
import time
import multiprocessing
import itertools

import networkx as nx
import penaltymodel as pm

import penaltymodel_cache as pmc

tmp_database_name = 'tmp_test_database_manager_{}.db'.format(time.time())


class TestInterfaceFunctions(unittest.TestCase):
    def setUp(self):
        self.database = pmc.cache_file(filename=tmp_database_name)

    def test_typical(self):
        dbfile = self.database

        # insert a penalty model
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, pm.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # cache the penaltymodel
        pmc.cache_penalty_model(widget, database=dbfile)

        # retrieve it
        widget_ = pmc.get_penalty_model(spec, database=dbfile)

        self.assertEqual(widget_, widget)

    def test_arbitrary_labels(self):
        dbfile = self.database

        # set up a specification and a corresponding penaltymodel
        graph = nx.Graph()
        for i in 'abcd':
            for j in 'efgh':
                graph.add_edge(i, j)

        decision_variables = ('a', 'e')
        feasible_configurations = ((-1, -1), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.SPIN)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 0 for edge in graph.edges}
        if decision_variables in quadratic:
            quadratic[decision_variables] = -1
        else:
            u, v = decision_variables
            assert (v, u) in quadratic
            quadratic[(v, u)] = -1
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # now cache the pmodel to make sure there is something to find
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # now try to retrieve it
        retreived_pmodel = pmc.get_penalty_model(spec, database=dbfile)

        self.assertIs(retreived_pmodel.model.vartype, pm.SPIN)

        # check that the specification is equal to the retreived_pmodel
        self.assertTrue(spec.__eq__(retreived_pmodel))

    def test_binary_specification(self):
        dbfile = self.database

        # set up a specification and a corresponding penaltymodel
        graph = nx.Graph()
        for i in 'abcd':
            for j in 'efgh':
                graph.add_edge(i, j)

        decision_variables = ('a', 'e')
        feasible_configurations = ((0, 0), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.BINARY)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 0 for edge in graph.edges}
        if decision_variables in quadratic:
            quadratic[decision_variables] = -1
        else:
            u, v = decision_variables
            assert (v, u) in quadratic
            quadratic[(v, u)] = -1
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # now cache the pmodel to make sure there is something to find
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # now try to retrieve it
        retreived_pmodel = pmc.get_penalty_model(spec, database=dbfile)

        self.assertIs(retreived_pmodel.model.vartype, pm.BINARY)

        # check that the specification is equal to the retreived_pmodel
        self.assertTrue(spec.__eq__(retreived_pmodel))
