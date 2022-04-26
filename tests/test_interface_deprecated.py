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

import time
import unittest
import unittest.mock

import dimod
import networkx as nx

import penaltymodel

from penaltymodel.core import Specification, PenaltyModel
from penaltymodel.core import iter_caches, iter_factories, get_penalty_model
from penaltymodel.database import isolated_cache


class TestInterfaceWithCache(unittest.TestCase):
    @isolated_cache()
    def test_retrieval(self):
        # put some stuff in the database

        spec = Specification(nx.path_graph(2), (0, 1), {(-1, -1), (1, 1)}, vartype='SPIN')
        model = dimod.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): -1}, 0.0, vartype='SPIN')
        widget = PenaltyModel.from_specification(spec, model, 2, -1)

        with self.assertWarns(DeprecationWarning):
            cache, = iter_caches()

        cache(widget)

        # now try to get it back
        with self.assertWarns(DeprecationWarning):
            new_widget = get_penalty_model(spec)

        self.assertEqual(widget, new_widget)

    @isolated_cache()
    def test_generation(self):
        # put some stuff in the database

        spec = Specification(nx.path_graph(2), (0, 1), {(-1, -1), (1, 1)}, vartype='SPIN')

        # get a penalty model with the desired properties
        with self.assertWarns(DeprecationWarning):
            pm = get_penalty_model(spec)

        with self.assertWarns(DeprecationWarning):
            pm2 = get_penalty_model(spec)

        self.assertEqual(pm, pm2)


class TestIterFactories(unittest.TestCase):
    def test_iter_factories(self):
        with self.assertWarns(DeprecationWarning):
            factory, = iter_factories()

        self.assertIs(factory, get_penalty_model)


class TestCoreNamespace(unittest.TestCase):
    def test_import(self):
        from penaltymodel.core import ImpossiblePenaltyModel