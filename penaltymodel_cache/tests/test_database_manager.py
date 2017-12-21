import unittest
import time
import sqlite3
import os

from collections import defaultdict

import networkx as nx
import penaltymodel as pm

import penaltymodel_cache as pmc


class TestConnectionAndConfiguration(unittest.TestCase):
    """Test the creation of the database and tables"""
    def test_connection(self):
        """Connect to the default database. We will not be using the default
        for many tests."""
        conn = pmc.cache_connect()
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()


class TestDatabaseManager(unittest.TestCase):
    """These tests assume that the database has been created or already
    exists correctly"""
    def setUp(self):
        # new connection for just this test
        self.clean_connection = pmc.cache_connect(':memory:')

    def tearDown(self):
        # close the memory connection
        self.clean_connection.close()

    def test_graph_insert_retrieve(self):
        conn = self.clean_connection

        graph = nx.barbell_graph(8, 8)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        with conn as cur:
            pmc.insert_graph(cur, nodelist, edgelist)

            # should only be one graph
            graphs = list(pmc.iter_graph(cur))
            self.assertEqual(len(graphs), 1)
            (nodelist_, edgelist_), = graphs
            self.assertEqual(nodelist, nodelist_)
            self.assertEqual(edgelist, edgelist_)

        # trying to reinsert should still result in only one graph
        with conn as cur:
            pmc.insert_graph(cur, nodelist, edgelist)
            graphs = list(pmc.iter_graph(cur))
            self.assertEqual(len(graphs), 1)

        # inserting with an empty dict as encoded_data should populate it
        encoded_data = {}
        with conn as cur:
            pmc.insert_graph(cur, nodelist, edgelist, encoded_data)
        self.assertIn('num_nodes', encoded_data)
        self.assertIn('num_edges', encoded_data)
        self.assertIn('edges', encoded_data)

        # now adding another graph should result in two items
        graph = nx.complete_graph(4)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)
        with conn as cur:
            pmc.insert_graph(cur, nodelist, edgelist)
            graphs = list(pmc.iter_graph(cur))
            self.assertEqual(len(graphs), 2)

    def test_feasible_configurations_insert_retrieve(self):
        conn = self.clean_connection

        feasible_configurations = {(-1, -1, -1): 0.0, (1, 1, 1): 0.0}

        with conn as cur:
            pmc.insert_feasible_configurations(cur, feasible_configurations)
            fcs = list(pmc.iter_feasible_configurations(cur))

            # should only be one and it should match
            self.assertEqual(len(fcs), 1)
            self.assertEqual([feasible_configurations], fcs)

            # reinsert, should not add
            pmc.insert_feasible_configurations(cur, feasible_configurations)
            fcs = list(pmc.iter_feasible_configurations(cur))

            # should only be one and it should match
            self.assertEqual(len(fcs), 1)
            self.assertEqual([feasible_configurations], fcs)

        feasible_configurations2 = {(-1, -1, -1): 0.0, (1, 1, 1): 0.0, (1, -1, 1): .4}
        with conn as cur:
            pmc.insert_feasible_configurations(cur, feasible_configurations2)
            fcs = list(pmc.iter_feasible_configurations(cur))

            self.assertIn(feasible_configurations2, fcs)

    def test_ising_model_insert_retrieve(self):
        conn = self.clean_connection

        graph = nx.path_graph(5)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        linear = {v: 0. for v in nodelist}
        quadratic = {(u, v): -1. for u, v in edgelist}
        offset = 0.0

        with conn as cur:
            pmc.insert_ising_model(cur, nodelist, edgelist, linear, quadratic, offset)

            ims = list(pmc.iter_ising_model(cur))

            # should be only one and it should match
            self.assertEqual(len(ims), 1)
            (nodelist_, edgelist_, linear_, quadratic_, offset_), = ims
            self.assertEqual(nodelist_, nodelist)
            self.assertEqual(edgelist_, edgelist)
            self.assertEqual(linear_, linear)
            self.assertEqual(quadratic_, quadratic)

        with conn as cur:
            # reinsert
            pmc.insert_ising_model(cur, nodelist, edgelist, linear, quadratic, offset)

            ims = list(pmc.iter_ising_model(cur))

            # should be only one and it should match
            self.assertEqual(len(ims), 1)

    def test_penalty_model_insert_retrieve(self):
        conn = self.clean_connection

        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, pm.SPIN)

        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=pm.SPIN)

        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        with conn as cur:
            pmc.insert_penalty_model(cur, widget)

        with conn as cur:
            pms = list(pmc.iter_penalty_model_from_specification(cur, spec))

            self.assertEqual(len(pms), 1)
            widget_, = pms
            self.assertEqual(widget_, widget)
