# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dimod
import networkx as nx
import numpy as np
import random
import unittest

import penaltymodel.core as pm
from penaltymodel.core.utilities import (get_uniform_penaltymodel,
                                         random_indices_generator,
                                         get_ordered_state_matrix)


class TestHelperFunctions(unittest.TestCase):
    #TODO: empty case
    def test_random_indices_generator(self):
        random.seed(0)
        expected = [(3, 1, 0), (2, 1, 6), (2, 1, 5), (1, 0, 4), (1, 0, 9)]

        ind_gen = random_indices_generator([3, 1, 10], 5)
        for ind, expected_ind in zip(ind_gen, expected):
            self.assertTupleEqual(tuple(ind), expected_ind)

    #TODO: possible quadratic with no linear?
    def test_get_ordered_state_matrix(self):
        """
        linear_labels = ['a', 'b', 'c']
        quadratic_labels = ['bc', 'ac']

        expected_labels = linear_labels + quadratic_labels
        ordered_states, column_labels = get_ordered_state_matrix(linear_labels,
                                                                 quadratic_labels)

        col_a = np.zeros(8)
        col_a[4:] = 1

        col_b = [0]*8
        col_b[::]
    """


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
        graph = nx.complete_graph(0)
        decision_variables = []
        feasible_configurations = {}
        classical_gap = 2
        ground_energy = 0
        pmodel = pm.PenaltyModel(graph, decision_variables,
                                 feasible_configurations, vartype, empty_model,
                                 classical_gap, ground_energy)

        with self.assertRaises(ValueError):
            get_uniform_penaltymodel(pmodel, n_tries=10)

    def test_balance_with_impossible_model(self):
        """Test impossible to further balance"""
        # Set up 3-input XOR gate
        decision_variables = ['a', 'b', 'c', 'd']
        g = nx.complete_graph(decision_variables + ['a1', 'a2'])
        feasible_config = [(1, 0, 0, 1),
                           (0, 1, 0, 1),
                           (0, 0, 1, 1),
                           (1, 1, 1, 1)]
        vartype = dimod.BINARY

        # Construct a balanced BQM to put in penaltymodel
        # Note: qubo is (a + b + c + d - 2*a1 - 2*a2) **2
        qubo = {('a', 'a'): 1, ('a', 'b'): 2, ('a', 'c'): 2, ('a', 'd'): 2, ('a1', 'a'): -4, ('a2', 'a'): -4,
                ('b', 'b'): 1, ('b', 'c'): 2, ('b', 'd'): 2, ('a1', 'b'): -4, ('a2', 'b'): -4,
                ('c', 'c'): 1, ('c', 'd'): 2, ('a1', 'c'): -4, ('a2', 'c'): -4,
                ('d', 'd'): 1, ('a1', 'd'): -4, ('a2', 'd'): -4,
                ('a1', 'a1'): 4, ('a1', 'a2'): 8,
                ('a2', 'a2'): 4}
        model = dimod.BinaryQuadraticModel.from_qubo(qubo)

        # Construct and rebalance penaltymodel
        pmodel = pm.PenaltyModel(g, decision_variables, feasible_config,
                                 vartype, model, classical_gap=2,
                                 ground_energy=0)
        with self.assertRaises(ValueError):
            get_uniform_penaltymodel(pmodel)

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
