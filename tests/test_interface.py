# Copyright 2022 D-Wave Systems Inc.
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

import itertools
import unittest
import unittest.mock

import dimod
import networkx as nx

from penaltymodel import get_penalty_model
from penaltymodel.database import isolated_cache


class TestGetPenaltyModel(unittest.TestCase):
    @isolated_cache()
    def test_different_energy_levels(self):
        samples_like = dimod.SampleSet.from_samples([[-1, -1, -1], [1, 1, 1]], energy=[0, .5], vartype='BINARY')

        bqm, gap = get_penalty_model(samples_like)

        self.assertAlmostEqual(bqm.energy([-1, -1, -1]), 0)
        self.assertAlmostEqual(bqm.energy([1, 1, 1]), .5)

    @isolated_cache()
    def test_different_energy_levels_graph_like(self):
        samples_like = dimod.SampleSet.from_samples([[-1, -1, -1], [1, 1, 1]], energy=[0, .5], vartype='BINARY')

        bqm, gap = get_penalty_model(samples_like, graph_like=nx.complete_graph(3))

        self.assertAlmostEqual(bqm.energy([-1, -1, -1]), 0)
        self.assertAlmostEqual(bqm.energy([1, 1, 1]), .5)

    @isolated_cache()
    def test_single_labelled(self):
        bqm, gap = get_penalty_model({'a': 1, 'b': 0})

        self.assertEqual(bqm, dimod.BQM({'a': -2.0, 'b': 2.0}, {('b', 'a'): 1.0}, 5.0, 'SPIN'))
        self.assertEqual(gap, 6)

        # now do it again, but make sure we use the cache
        with unittest.mock.patch('penaltymodel.interface.generate') as mock:
            mock.side_effect = Exception('boom')
            new = get_penalty_model({'a': 1, 'b': 0})

        self.assertEqual((bqm, gap), new)

    @isolated_cache()
    def test_subgraph_labelled(self):
        G = nx.Graph(itertools.product('abc', 'def'))

        bqm, gap = get_penalty_model(([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], 'dbf'), G)

        ground = dimod.keep_variables(dimod.ExactSolver().sample(bqm), 'dbf').lowest().aggregate()

        self.assertEqual(len(ground), 4)
        for sample in ground.samples():
            self.assertEqual(sample['d'] > 0 and sample['b'] > 0, sample['f'] > 0)

    @isolated_cache()
    def test_unorded_range_labels(self):
        # NAE
        samples_like = ([(0, 0, 1, 0),
                         (0, 1, 1, 1),
                         (1, 0, 1, 1),
                         (0, 1, 0, 0),
                         (1, 0, 0, 0),
                         (1, 1, 0, 1)],
                        (0, 1, 3, 2))

        bqm, gap = get_penalty_model(samples_like)

        ground = dimod.ExactSolver().sample(bqm).lowest().aggregate()

        self.assertEqual(len(ground), 6)
        for sample in ground.samples():
            self.assertTrue(len(set(sample.values())) > 1)
