import unittest
import os
import time
import multiprocessing

import networkx as nx
import penaltymodel as pm

import penaltymodel_cache as pmc


def fresh_database():
    """New, unique database path. Puts it in a temp directory off the current
    working directory"""
    dir_ = os.path.join(os.getcwd(), 'tmp')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    time.sleep(.0001)  # to make sure they are unique
    return os.path.join(dir_, 'tmp-%.6f.db' % time.time())


class TestInterfaceFunctions(unittest.TestCase):
    def setUp(self):
        self.clean_database = fresh_database()

    def test_typical(self):
        """typical use case. Serves as a smoke test"""
        db = self.clean_database

        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})

        # for a clean database, should be nothing in it
        with self.assertRaises(pm.MissingPenaltyModel):
            pmc.get_penalty_model(spec, database=db)

        # load it into the database
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        penalty_model = pm.PenaltyModel(spec, model, 2, -2)
        pmc.cache_penalty_model(penalty_model, database=db)

        # model should come out
        penalty_model_out = pmc.get_penalty_model(spec, database=db)
        self.assertEqual(penalty_model, penalty_model_out)

    def test_many_writes(self):
        """many simultaneous invocations of cache_penalty_model."""

        # get a penalty model
        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)},
                                {v: (-2, 2) for v in range(2)},
                                {(0, 1): (-1, 1)})
        model = pm.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0, pm.SPIN)
        penalty_model = pm.PenaltyModel(spec, model, 2, -2)

        # we need to do this a large number of times, because it won't always catch
        for __ in range(100):
            db = self.clean_database

            def _add_model():
                pmc.cache_penalty_model(penalty_model, database=db)

            processes = [multiprocessing.Process(target=_add_model)
                         for __ in range(10)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            all_models = pmc.dump_penalty_models(db)

            self.assertEqual(len(all_models), 1)
            self.assertIn(penalty_model, all_models)
