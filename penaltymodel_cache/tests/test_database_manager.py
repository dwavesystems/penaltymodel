import unittest
import time
import sqlite3

import networkx as nx

import penaltymodel_cache as pmc


def fresh_database():
    """New, unique database path. Puts it in a temp directory off the current
    working directory"""
    dir_ = os.path.join(os.getcwd(), 'tmp')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    return os.path.join(dir_, 'tmp-%.6f.db' % time.clock())


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
        # get a new clean database in memory and instatiate the tables
        self.clean_conn = pmc.cache_connect(':memory:')

    def tearDown(self):
        self.clean_conn.close()

    def test_get_graph_id(self):
        # create some graphs we can insert
        conn = self.clean_conn

        G = nx.complete_graph(5)

        nodelist = sorted(G.nodes)
        edgelist = sorted(tuple(sorted(edge)) for edge in G.edges)

        gid = pmc.graph_id(conn, nodelist, edgelist)

        # the same graph again should give the same id
        self.assertEqual(gid, pmc.graph_id(conn, nodelist, edgelist))

        # new graph should have different id
        H = nx.barbell_graph(5, 6)
        nodelist = sorted(H.nodes)
        edgelist = sorted(tuple(sorted(edge)) for edge in H.edges)
        hid = pmc.graph_id(conn, nodelist, edgelist)
        self.assertNotEqual(gid, hid)

    def test_get_configurations_id(self):

        conn = self.clean_conn

        configurations = {(-1, -1, 1), (1, -1, 1)}

        rid = pmc.get_configurations_id(conn, configurations)

        # should stay the same
        self.assertEqual(rid, pmc.get_configurations_id(conn, configurations))

        # differnt config should be differnt
        configs2 = {(-1, 1, -1)}
        self.assertNotEqual(rid, pmc.get_configurations_id(conn, configs2))

    def test_query_penalty_model(self):

        conn = self.clean_conn

        graph = nx.complete_graph(3)
        decision_variables = (0, 1)
        feasible_configurations = {(-1, -1), (-1, 1)}

        # returns penaltymodel as an iterator, so should be empty at this point
        penalty_models = pmc.query_penalty_model(conn, graph, decision_variables,
                                                 feasible_configurations)

        with self.assertRaises(StopIteration):
            next(penalty_models)
