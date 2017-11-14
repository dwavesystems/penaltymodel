import unittest
import time
import sqlite3
import os

from collections import defaultdict

import networkx as nx
import penaltymodel as pm

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
        # get a new clean database in memory, only lasts as long as the unittest
        self.clean_conn = pmc.cache_connect(':memory:')

    def tearDown(self):
        # close the memory connection
        self.clean_conn.close()

    def test_penalty_model_id(self):
        """Typical test for the penalty_model_id function.
        Running it twice should retreive the same penalty_model_id."""
        conn = self.clean_conn

        # set up a penalty model we can use in the test
        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        p = pm.PenaltyModel(spec, model, 2, -2)

        pmid = pmc.penalty_model_id(conn, p)

        # rerunning should return the same id
        self.assertEqual(pmid, pmc.penalty_model_id(conn, p))

    def test_get_penalty_model_from_specification(self):
        """Typical test for the penalty_model_id function.
        Save and retrieve one penalty model."""
        conn = self.clean_conn

        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})

        # set up a penalty model we can use put into the database
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        penalty_model = pm.PenaltyModel(spec, model, 2, -2)

        # load it into the database
        pmc.penalty_model_id(conn, penalty_model)

        # now let's try to get it back
        ret_penalty_model = pmc.get_penalty_model_from_specification(conn, spec)

        # check that everything is the same
        self.assertEqual(penalty_model, ret_penalty_model)

    def test_get_penalty_model_from_specification_multiple_specs(self):
        """For models with similar specs, should return the correct model"""
        conn = self.clean_conn

        spec1 = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})

        # spec with smaller quadratic energy range
        spec2 = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)},
                                 None,
                                 defaultdict(lambda: (-.5, .5)))

        # set up a penalty model we can use put into the database
        model_1 = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        penalty_model_1 = pm.PenaltyModel(spec1, model_1, 2, -2)
        pmc.penalty_model_id(conn, penalty_model_1)

        # now another penalty model that can come back from the same query
        model_2 = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -.5}, 0, pm.SPIN)
        penalty_model_2 = pm.PenaltyModel(spec1, model_2, 2, -2)
        pmc.penalty_model_id(conn, penalty_model_2)

        # we should get the one with the larger classical gap
        penalty_model = pmc.get_penalty_model_from_specification(conn, spec1)
        self.assertEqual(penalty_model, penalty_model_1)

        # smaller classical gap (also a spec that we didn't use)
        penalty_model = pmc.get_penalty_model_from_specification(conn, spec2)
        self.assertEqual(penalty_model, penalty_model_2)
