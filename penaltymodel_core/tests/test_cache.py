# Copyright 2021 D-Wave Systems Inc.
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

import unittest

import networkx as nx
import dimod

from penaltymodel import MissingPenaltyModel
from penaltymodel.cache import PenaltyModelCache


class TestPenaltyModelCache(unittest.TestCase):

    def assertEqual(self, first, second, *args, **kwargs):
        if isinstance(first, nx.Graph) and isinstance(second, nx.Graph):
            self.assertEqual(first.nodes, second.nodes, *args, **kwargs)
            self.assertEqual(first.edges, second.edges, *args, **kwargs)
        else:
            return super().assertEqual(first, second, *args, **kwargs)

    def setUp(self):
        self.cache = PenaltyModelCache(':memory:')

    def tearDown(self):
        self.cache.close()

    def test_graph_insert_retrieve(self):
        cache = self.cache

        graph = nx.barbell_graph(8, 8)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        cache.insert_graph(graph)

        self.assertEqual(list(cache.iter_graphs())[0], graph)

        # insert it again, and check that there is still only one graph
        cache.insert_graph(graph)
        self.assertEqual(len(list(cache.iter_graphs())), 1)

        # now adding another graph should result in two items
        graph = nx.complete_graph(4)
        cache.insert_graph(graph)
        self.assertEqual(len(list(cache.iter_graphs())), 2)

    def test_feasible_configurations_insert_retrieve(self):
        cache = self.cache

        fc1 = {(-1, -1, -1): 0.0, (1, 1, 1): 0.0}

        cache.insert_table(fc1)
        fcs = list(cache.iter_tables())
        self.assertEqual(len(fcs), 1)
        self.assertEqual([fc1], fcs)

        # entering it again shouldn't change anything
        cache.insert_table(fc1)
        fcs = list(cache.iter_tables())
        self.assertEqual(len(fcs), 1)
        self.assertEqual([fc1], fcs)

        # put two more in, one with different configs, one with different
        # energies
        fc2 = {(-1, -1, +1): 0.0, (1, 1, 1): 0.0}
        cache.insert_table(fc2)
        fc3 = {(-1, -1, -1): 0.0, (1, 1, 1): 1.0}
        cache.insert_table(fc3)

        fcs = list(cache.iter_tables())
        self.assertEqual(len(fcs), 3)
        self.assertIn(fc1, fcs)
        self.assertIn(fc2, fcs)
        self.assertIn(fc3, fcs)

        # finally test binary rather than spin
        fc4 = {(0, 0, 0): 0.0, (1, 1, 1): 0.0}  # same as fc1
        cache.insert_table(fc4)
        fc5 = {(1, 1, 1): 1.5, (0, 1, 0): 5}
        cache.insert_table(fc5)

        fcs = list(cache.iter_tables())
        self.assertEqual(len(fcs), 4)
        self.assertIn(fc1, fcs)
        self.assertIn(fc2, fcs)
        self.assertIn(fc3, fcs)
        self.assertIn({(1, 1, 1): 1.5, (-1, 1, -1): 5}, fcs)  # comes out as spin

    def test_bqm_insert_retrieve(self):
        cache = self.cache

        bqm0 = dimod.generators.gnp_random_bqm(10, .5, 'SPIN', random_state=53)

        cache.insert_binary_quadratic_model(bqm0)
        self.assertEqual(list(cache.iter_binary_quadratic_models()), [bqm0])

        # insert again, should not duplicate
        cache.insert_binary_quadratic_model(bqm0)
        self.assertEqual(list(cache.iter_binary_quadratic_models()), [bqm0])

        # make a new bqm with reversed nodeorder
        bqm2 = dimod.BinaryQuadraticModel('SPIN')
        bqm2.add_linear_from((v, bqm0.get_linear(v)) for v in reversed(range(bqm0.num_variables)))
        bqm2.add_quadratic_from(bqm0.iter_quadratic())
        bqm2.offset = bqm0.offset

        cache.insert_binary_quadratic_model(bqm2)
        self.assertEqual(list(cache.iter_binary_quadratic_models()), [bqm0])

        bqm3 = dimod.BinaryQuadraticModel(bqm0, dtype=object)
        cache.insert_binary_quadratic_model(bqm3)
        self.assertEqual(list(cache.iter_binary_quadratic_models()), [bqm0])

    def test_and_gate_insert_retrieve(self):
        cache = self.cache

        classical_gap = 2
        bqm = dimod.generators.and_gate(0, 1, 2, strength=classical_gap).change_vartype('SPIN', inplace=True)
        table = {(-1, -1, -1): 0,
                 (-1, 1, -1): 0,
                 (1, -1, -1): 0,
                 (1, 1, 1): 0}
        decision_variables = [0, 1, 2]

        cache.insert_penalty_model(bqm, table, decision_variables, classical_gap)

        li = list(cache.iter_penalty_models())
        self.assertEqual(len(li), 1)
        pm, = li
        self.assertEqual(pm.bqm, bqm)
        self.assertEqual(pm.table, table)
        self.assertEqual(pm.decision_variables, decision_variables)
        self.assertEqual(pm.classical_gap, classical_gap)

    def test_retrieve(self):
        cache = self.cache

        table = {(-1, -1, -1): 0,
                 (-1, 1, -1): 0,
                 (1, -1, -1): 0,
                 (1, 1, 1): 0}
        decision_variables = [0, 1, 2]

        # put some AND gates into the cache
        bqm1 = dimod.generators.and_gate(0, 1, 2, strength=1).change_vartype('SPIN', inplace=True)
        cache.insert_penalty_model(bqm1, table, decision_variables, classical_gap=1)
        bqm2 = dimod.generators.and_gate(0, 1, 2, strength=2).change_vartype('SPIN', inplace=True)
        cache.insert_penalty_model(bqm2, table, decision_variables, classical_gap=2)

        bqm, classical_gap = cache.retrieve(nx.complete_graph(3), table, decision_variables)
        self.assertEqual(bqm, bqm2)  # largest gap, fits within bounds

        bqm, classical_gap = cache.retrieve(nx.complete_graph(3), table, decision_variables, linear_bound=(-.5, .5), min_classical_gap=1)
        self.assertEqual(bqm, bqm1)  # since this fits in the given bounds

        with self.assertRaises(MissingPenaltyModel):
            cache.retrieve(nx.complete_graph(3), table, decision_variables, linear_bound=(-.5, .5))
