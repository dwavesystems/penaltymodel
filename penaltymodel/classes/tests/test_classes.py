import unittest
import random
import itertools

import networkx as nx

import penaltymodel as pm


class TestSpecification(unittest.TestCase):
    def test_serialize(self):
        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})
        serial = spec.serialize()

        #
        # todo
        #


class TestPenaltyModel(unittest.TestCase):
    pass
