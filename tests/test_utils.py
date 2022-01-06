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

import unittest

import networkx as nx
import penaltymodel


class TestAsGraph(unittest.TestCase):
    def test_int(self):
        self.assertEqual(penaltymodel.as_graph(0).nodes, nx.Graph().nodes)

        K10 = penaltymodel.as_graph(10)
        self.assertEqual(K10.nodes, nx.complete_graph(10).nodes)
        self.assertEqual(K10.edges, nx.complete_graph(10).edges)

    def test_sequence(self):
        self.assertEqual(penaltymodel.as_graph('').nodes, nx.Graph().nodes)

        K10 = penaltymodel.as_graph('abc')
        self.assertEqual(K10.nodes, nx.complete_graph('abc').nodes)
        self.assertEqual(K10.edges, nx.complete_graph('abc').edges)

    def test_graph(self):
        K0 = nx.Graph()
        self.assertIs(K0, penaltymodel.as_graph(K0))

        P6 = nx.path_graph(6)
        self.assertIs(P6, penaltymodel.as_graph(P6))
