import unittest

import networkx as nx

import penaltymodel_cache as pmc


class TestConnectionAndConfiguration(unittest.TestCase):
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

    def test_configure_database(self):
        """configure_database function should add schema to a new database, do nothing to a database
        built by configure_database under a version of pmc with the same major and minor version
        and should throw an exception if build under a different version.

        TODO: would be good to directly test the schema, but should be covered by other unittests
        indirectly
        """
        # create a new clean connection
        conn = pmc.connection(':memory:')

        # run the function
        pmc.configure_database(conn)

        # now run it again (shouldn't do anything)
        pmc.configure_database(conn)

        # let's change the version to something new and try that
        pmc.database_manager.__version__ = '-1.-1.-1'
        # in this case it should overwrite the database
        with self.assertRaises(pmc.OutdatedDatabaseException):
            pmc.configure_database(conn)
        # reset the version back to the correct version
        pmc.database_manager.__version__ = pmc.version.__version__

        # close the connection
        conn.close()


class TestManager(unittest.TestCase):
    """These tests assume that the database has been created or already
    exists correctly"""
    def setUp(self):
        # get a new clean database in memory and instatiate the tables
        self.clean_conn = clean_conn = pmc.connection(':memory:')

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
