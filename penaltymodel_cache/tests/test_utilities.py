import dimod
import networkx as nx
import unittest

import penaltymodel.core as pm
from penaltymodel.core.utilities import get_uniform_penaltymodel


class TestPenaltyModelBalance(unittest.TestCase):
    def check_balance(self, balanced_pmodel, original_pmodel, tol=10**-12):
        # Sample the balanced penaltymodel
        sampleset = dimod.ExactSolver().sample(balanced_pmodel.model)
        sample_states = sampleset.lowest().record.sample

        # Reorder sample columns to match feasible_configuration
        index_dict = {v: i for i, v in enumerate(sampleset.variables)}
        indices = [index_dict[dv] for dv in original_pmodel.decision_variables]
        decision_states = list(map(tuple, sample_states[:, indices]))

        # Check that gap is larger than min_classical_gap with some tolerance
        self.assertGreaterEqual(balanced_pmodel.classical_gap,
                                original_pmodel.min_classical_gap - tol)

        # Check that there are no duplicates
        self.assertEqual(len(set(decision_states)), len(decision_states),
                         msg="There are duplicate states in balanced solution")

        # Check that we have the correct number of states
        self.assertEqual(len(decision_states),
                         len(original_pmodel.feasible_configurations),
                         msg="Incorrect number of states in balanced solution")

        # Check that all states are valid
        for state in decision_states:
            self.assertIn(state, original_pmodel.feasible_configurations,
                          msg="{} is not a feasible configuration".format(state))

    def test_balance_with_empty_penaltymodel(self):
        # Build a penaltymodel with an empty bqm
        vartype = dimod.SPIN
        empty_model = dimod.BinaryQuadraticModel.empty(vartype)
        n_nodes = 3
        graph = nx.complete_graph(n_nodes)
        decision_variables = list(range(n_nodes - 1))
        feasible_configurations = {(-1, -1), (1, 1)}
        classical_gap = 2
        ground_energy = 0
        pmodel = pm.PenaltyModel(graph, decision_variables,
                                 feasible_configurations, vartype, empty_model,
                                 classical_gap, ground_energy)

        with self.assertRaises(ValueError):
            get_uniform_penaltymodel(pmodel, n_tries=10)

    def test_balance_with_impossible_model(self):
        """Test impossible to further balance"""
        # Set up
        decision_variables = ['a', 'b', 'c', 'd']
        g = nx.complete_graph(decision_variables + ['aux0', 'aux1'])
        feasible_config = [(0, 0, 0, 1),
                           (0, 1, 1, 0),
                           (0, 1, 1, 1),
                           (1, 0, 0, 0),
                           (1, 1, 1, 1)]
        vartype = dimod.BINARY

        # Construct a balanced BQM to put in penaltymodel
        linear = {'a': -6.666666666666666,
                  'b': -2.666666666666666,
                  'c': -4.0,
                  'd': -2.666666666666666,
                  'aux0': -4.0,
                  'aux1': -12.0}
        quadratic = {('a', 'b'): 2.6666666666666665,
                     ('a', 'c'): 2.6666666666666665,
                     ('a', 'd'): 1.3333333333333333,
                     ('a', 'aux0'): 4.0,
                     ('a', 'aux1'): 4.0,
                     ('b', 'c'): -4.0,
                     ('b', 'd'): 0.0,
                     ('b', 'aux0'): 4.0,
                     ('b', 'aux1'): 4.0,
                     ('c', 'd'): -1.3333333333333333,
                     ('c', 'aux0'): 4.0,
                     ('c', 'aux1'): 4.0,
                     ('d', 'aux0'): -4.0,
                     ('d', 'aux1'): 4.0,
                     ('aux0', 'aux1'): 4.0}
        offset = 14.666666666666664
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # Construct and rebalance penaltymodel
        pmodel = pm.PenaltyModel(g, decision_variables, feasible_config,
                                 vartype, model, classical_gap=2,
                                 ground_energy=0)
        new_pmodel = get_uniform_penaltymodel(pmodel)

        self.assertEqual(model, pmodel.model)

    def test_balance_with_already_balanced_model(self):
        """Test balance on an already balanced NOT-gate penaltymodel"""
        # Set up
        g = nx.Graph([('in', 'out')])
        decision_variables = ['in', 'out']
        linear_biases = {}
        quadratic_biases = {('in', 'out'): 1}
        feasible_config = {(-1, +1), (+1, -1)}  # not-gate
        vartype = dimod.SPIN
        offset = 0

        # Construct a balanced BQM to put in penaltymodel
        model = dimod.BinaryQuadraticModel(linear_biases, quadratic_biases,
                                           offset, vartype)

        # Construct and rebalance penaltymodel
        pmodel = pm.PenaltyModel(g, decision_variables, feasible_config,
                                 vartype, model, classical_gap=2,
                                 ground_energy=0)
        new_pmodel = get_uniform_penaltymodel(pmodel)

        self.assertEqual(model, new_pmodel.model)

    def test_balance_with_qubo(self):
        decision_variables = ['a', 'b']
        feasible_config = {(1, 0), (0, 1)}
        vartype = dimod.BINARY
        classical_gap = 0.5
        ground_energy = -1
        g = nx.complete_graph(decision_variables)

        model = dimod.BinaryQuadraticModel({'a': -1, 'b': 0.5, 'c': -.5},
                                           {'ab': 1, 'bc': -1, 'ac': 0.5},
                                           0, vartype)
        pmodel = pm.PenaltyModel(g, decision_variables, feasible_config,
                                 vartype, model, classical_gap, ground_energy)
        new_pmodel = get_uniform_penaltymodel(pmodel)
        self.check_balance(new_pmodel, pmodel)

    def test_balance_with_ising(self):
        # Constructing three-input AND-gate graph
        decision_variables = ['in0', 'in1', 'in2', 'out']
        g = nx.Graph([('in0', 'out'), ('in1', 'out'), ('in2', 'out')])
        aux_edges = [(dv, aux) for dv in decision_variables for aux
                     in ['aux0', 'aux1']]
        g.add_edges_from(aux_edges)

        # Construct an imbalanced penaltymodel of the above AND-gate
        # Note: the following imbalanced penaltymodel has 12 ground states
        linear_biases = {'in0': -1, 'in1': -.5, 'in2': 0,       # Shore 0
                         'out': 2, 'aux0': .5, 'aux1': 1}       # Shore 1
        quadratic_biases = \
            {('in0', 'out'): -1, ('in0', 'aux0'): -.5, ('in0', 'aux1'): -.5,
             ('in1', 'out'): -1, ('in1', 'aux0'): 1, ('in1', 'aux1'): -.5,
             ('in2', 'out'): -1, ('in2', 'aux0'): 0, ('in2', 'aux1'): 1}
        feasible_config = {(-1, -1, -1, -1),
                           (-1, -1, +1, -1),
                           (-1, +1, -1, -1),
                           (-1, +1, +1, -1),
                           (+1, -1, -1, -1),
                           (+1, -1, +1, -1),
                           (+1, +1, -1, -1),
                           (+1, +1, +1, +1)}
        offset = 4.5
        vartype = dimod.SPIN
        classical_gap = 2
        ground_energy = 0
        tol = 1e-12

        model = dimod.BinaryQuadraticModel(linear_biases, quadratic_biases,
                                           offset, vartype)
        pmodel = pm.PenaltyModel(g, decision_variables, feasible_config,
                                 vartype, model, classical_gap, ground_energy)

        new_pmodel = get_uniform_penaltymodel(pmodel, tol=tol)
        self.check_balance(new_pmodel, pmodel)