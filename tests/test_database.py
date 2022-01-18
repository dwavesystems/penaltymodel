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

import dimod
import networkx as nx
import numpy as np

from penaltymodel import MissingPenaltyModel
from penaltymodel.database import patch_cache


class TestBQMCache(unittest.TestCase):
    @patch_cache()
    def test_bqm_insert_retrieve(self, cache):
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


class TestGraphCache(unittest.TestCase):
    def assertEqual(self, first, second, *args, **kwargs):
        if isinstance(first, nx.Graph) and isinstance(second, nx.Graph):
            self.assertEqual(first.nodes, second.nodes, *args, **kwargs)
            self.assertEqual(first.edges, second.edges, *args, **kwargs)
        else:
            return super().assertEqual(first, second, *args, **kwargs)

    @patch_cache()
    def test_graph_insert_retrieve(self, cache):
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

    @patch_cache()
    def test_graph_like(self, cache):
        cache.insert_graph(10)
        cache.insert_graph(range(10))
        cache.insert_graph(nx.complete_graph(10))
        cache.insert_graph(range(11))
        cache.insert_graph(nx.complete_graph(12))

        # we should see three graphs
        graphs = list(cache.iter_graphs())

        self.assertEqual(len(graphs), 3)
        self.assertTrue(any(nx.complete_graph(10).nodes == G.nodes for G in graphs))
        self.assertTrue(any(nx.complete_graph(11).nodes == G.nodes for G in graphs))
        self.assertTrue(any(nx.complete_graph(12).nodes == G.nodes for G in graphs))
        self.assertTrue(any(nx.complete_graph(10).edges == G.edges for G in graphs))
        self.assertTrue(any(nx.complete_graph(11).edges == G.edges for G in graphs))
        self.assertTrue(any(nx.complete_graph(12).edges == G.edges for G in graphs))


class TestPenaltyModelCache(unittest.TestCase):
    @patch_cache()
    def test_and_gate_insert_retrieve(self, cache):
        classical_gap = 2
        bqm = dimod.generators.and_gate(0, 1, 2, strength=classical_gap).change_vartype('SPIN', inplace=True)
        sampleset = dimod.ExactSolver().sample(bqm).lowest()

        cache.insert_penalty_model(bqm, sampleset, classical_gap)

        li = list(cache.iter_penalty_models())
        self.assertEqual(len(li), 1)
        pm, = li
        self.assertEqual(pm.bqm, bqm)
        data = list(pm.sampleset.data())
        for datum in sampleset.data():
            self.assertIn(datum, data)
        self.assertEqual(pm.classical_gap, classical_gap)


class TestRetrieve(unittest.TestCase):
    @patch_cache()
    def test_retrieve(self, cache):

        # put some AND gates into the cache
        bqm1 = dimod.generators.and_gate(0, 1, 2, strength=1).change_vartype('SPIN', inplace=True)
        cache.insert_penalty_model(bqm1, dimod.ExactSolver().sample(bqm1).lowest(), classical_gap=1)
        bqm2 = dimod.generators.and_gate(0, 1, 2, strength=2).change_vartype('SPIN', inplace=True)
        cache.insert_penalty_model(bqm2, dimod.ExactSolver().sample(bqm2).lowest(), classical_gap=2)

        samples = [[-1, -1, -1], [-1, +1, -1], [+1, -1, -1], [+1, +1, +1]]

        bqm, classical_gap = cache.retrieve(samples, nx.complete_graph(3))
        self.assertEqual(bqm, bqm2)  # largest gap, fits within bounds

        bqm, classical_gap = cache.retrieve(samples, nx.complete_graph(3), linear_bound=(-.5, .5), min_classical_gap=1)
        self.assertEqual(bqm, bqm1)  # since this fits in the given bounds

        with self.assertRaises(MissingPenaltyModel):
            cache.retrieve(samples, nx.complete_graph(3), linear_bound=(-.5, .5))


class TestSampleSetCache(unittest.TestCase):
    @patch_cache()
    def test_sampleset_insert_retrieve(self, cache):
        samples = [[-1, -1, +1], [+1, +1, +1]]
        cache.insert_sampleset(samples)
        sampleset, = cache.iter_samplesets()
        np.testing.assert_array_equal(samples, sampleset.record.sample)

    @patch_cache()
    def test_large(self, cache):
        samples = 2*np.random.randint(0, 2, size=(1, 32))-1
        cache.insert_sampleset(samples)
        sampleset, = cache.iter_samplesets()
        np.testing.assert_array_equal(samples, sampleset.record.sample)
