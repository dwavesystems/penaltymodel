import unittest

import networkx as nx

import penaltymodel as pm


class TestSpecification(unittest.TestCase):
    def test_creation(self):
        graph = nx.complete_graph(4)
        decision_variables = [0, 1]
        feasible_configurations = {(0, 0), (1, 1)}
        specification = pm.Specification(graph, decision_variables, feasible_configurations)


class TestPenaltyModel(unittest.TestCase):
    pass


class TestIsing(unittest.TestCase):
    def test_trivial(self):
        model = pm.Ising({}, {})
        self.assertEqual(model.quadratic, {})
        self.assertEqual(model.offset, 0.0)


class TestQUBO(unittest.TestCase):
    def test_trivial(self):
        model = pm.QUBO({}, 0.0)

        self.assertEqual(model.quadratic, {})
        self.assertEqual(model.offset, 0.0)

        model = pm.QUBO({(0, 0): -1.}, .5)

        self.assertEqual(model.quadratic, {(0, 0): -1.})
        self.assertEqual(model.offset, .5)

    def test_convert_from_ising(self):

        ising_model = pm.Ising({0: 1, 1: -1}, {(0, 1): -.5})

        model = pm.QUBO(ising_model)

        # TODO, more here
