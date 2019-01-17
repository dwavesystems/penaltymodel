# Copyright 2019 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
from collections import defaultdict

import networkx as nx
import pysmt.test

from pysmt.shortcuts import GT, LT, And, Equals, GE, LE, Not

from penaltymodel.maxgap.theta import Theta, limitReal


class TestTheta(pysmt.test.TestCase):
    def setUp(self):
        pysmt.test.TestCase.setUp(self)

    def assertConsistentGraphTheta(self, graph, theta):
        # ok, let's check that the set of nodes are the same for graph and theta
        self.assertEqual(set(graph.nodes), set(theta.linear))

        # let's also check that they both have the same edges
        for u, v in graph.edges:
            self.assertIn(u, theta.adj)
            self.assertIn(v, theta.adj[u])
            self.assertIn(v, theta.adj)
            self.assertIn(u, theta.adj[v])
        for v, u in theta.quadratic:
            self.assertIn(u, theta.adj)
            self.assertIn(v, theta.adj[u])
            self.assertIn(v, theta.adj)
            self.assertIn(u, theta.adj[v])
            self.assertIn(u, graph)
            self.assertIn(v, graph[u])
            self.assertIn(v, graph)
            self.assertIn(u, graph[v])

            # make sure that everything points to the same bias
            self.assertEqual(id(theta.quadratic[(v, u)]), id(theta.adj[u][v]))
            self.assertEqual(id(theta.quadratic[(v, u)]), id(theta.adj[v][u]))

        # check that each bias is unique
        self.assertEqual(len(set(id(bias) for __, bias in theta.linear.items())),
                         len(theta.linear))
        self.assertEqual(len(set(id(bias) for __, bias in theta.quadratic.items())),
                         len(theta.quadratic))

    def test_from_graph_K5(self):
        """Check that everything in theta as built correctly"""

        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        graph = nx.complete_graph(5)

        theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

        self.assertConsistentGraphTheta(graph, theta)

    def test_energy_ranges_K5(self):
        """Check that the energy ranges were set the way we expect"""
        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        graph = nx.complete_graph(5)
        theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

        for v, bias in theta.linear.items():
            min_, max_ = linear_ranges[v]
            self.assertUnsat(And(GT(bias, limitReal(max_)), And(theta.assertions)))
            self.assertUnsat(And(LT(bias, limitReal(min_)), And(theta.assertions)))

        for (u, v), bias in theta.quadratic.items():
            min_, max_ = quadratic_ranges[(u, v)]
            self.assertUnsat(And(GT(bias, limitReal(max_)), And(theta.assertions)))
            self.assertUnsat(And(LT(bias, limitReal(min_)), And(theta.assertions)))

    def test_copy(self):
        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        graph = nx.complete_graph(5)
        theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

        cp_theta = theta.copy()

        # should all point to the same
        for v, bias in theta.linear.items():
            self.assertSat(Equals(bias, cp_theta.linear[v]))
            self.assertUnsat(Not(Equals(bias, cp_theta.linear[v])))
