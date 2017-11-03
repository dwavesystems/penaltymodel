import unittest

import networkx as nx
from penaltymodel import Specification, PenaltyModel

import penaltymodel_cache as pmc


# Let's set up some specifications and penalty models that can be used in tests
test_specifications = []
test_models = []

# empty
spec0 = Specification(nx.Graph(), [], {})
pm0 = PenaltyModel()
pm0.load_from_specification(spec0, {}, {}, 0.0)
spec1 = Specification(nx.Graph(), [], set())
pm1 = PenaltyModel()
pm1.load_from_specification(spec0, {}, {})

test_specifications += [spec0, spec1]
test_models += [pm0, pm1]

# one node
spec = Specification(nx.complete_graph(1), [0], {(-1,)})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 1}, {}, 1)
test_specifications.append(spec)
test_models.append(pm)

# two nodes
spec = Specification(nx.complete_graph(2), [0], {(-1,): 0})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 1, 1: 0}, {(0, 1): 0}, 1)
test_specifications.append(spec)
test_models.append(pm)

spec = Specification(nx.complete_graph(2), [0], {(-1,): 0})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 1, 1: 0}, {}, 1)
test_specifications.append(spec)
test_models.append(pm)

spec = Specification(nx.complete_graph(2), [0], {(-1,): 0})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 1, 1: 0}, {(0, 1): 1}, 0)
test_specifications.append(spec)
test_models.append(pm)

spec = Specification(nx.complete_graph(2), [0], {(-1, 1): 0, (-1, -1): 0})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 1, 1: 0}, {(0, 1): 0}, 0)
test_specifications.append(spec)
test_models.append(pm)

spec = Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})
pm = PenaltyModel()
pm.load_from_specification(spec, {0: 0, 1: 0}, {(0, 1): -1}, 0)
test_specifications.append(spec)
test_models.append(pm)


class TestInterfaceFunctions(unittest.TestCase):
    def test_cache_penalty_model_empty_penaltymodel(self):
        for pm in test_models:
            pmc.cache_penalty_model(pm)

    # def test_get_penalty_model_from_specification(self):
    #     # set up a penalty model (which acts as a specification)

    #     pm = PenaltyModel()
    #     pm.graph = nx.complete_graph(3)
