import unittest
import os
import time
import multiprocessing

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
        spec = pm.Specification(graph, decision_variables, feasible_configurations)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # cache the penaltymodel
        pmc.cache_penalty_model(widget, database=dbfile)

        # retrieve it
        widget_ = pmc.get_penalty_model(spec, database=dbfile)

        self.assertEqual(widget_, widget)
