import unittest

import networkx as nx

import penaltymodel_cache as pmc


class TestConnectionAndInstantiation(unittest.TestCase):
    """Test the creation of the database and tables"""
    def test_connection(self):
        # TODO
        pass

    def test_connection_specified_file(self):
        # TODO
        pass

    def test_connection_specified_directory(self):
        # TODO
        pass

    def test_instatiation(self):
        # create a new clean connection
        conn = pmc.connection(':memory:')

        # run the instantiation
        pmc.instantiate_database(conn)

        # TODO

        conn.close()


class TestManager(unittest.TestCase):
    """These tests assume that the database has been created or already
    exists correctly"""
    def setUp(self):
        # get a new clean database in memory and instatiate the tables
        self.clean_conn = clean_conn = pmc.connection(':memory:')
        pmc.instantiate_database(clean_conn)

    def tearDown(self):
        self.clean_conn.close()

    def test_get_graph_id(self):
        # create some graphs we can insert
        conn = self.clean_conn

        G = nx.complete_graph(5)

        gid = pmc.get_graph_id(conn, G)

        # the same graph again should give the same id
        self.assertEqual(gid, pmc.get_graph_id(conn, G))

        # new graph should have different id
        H = nx.barbell_graph(5, 6)
        hid = pmc.get_graph_id(conn, H)
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
