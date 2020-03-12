# Copyright 2017 D-Wave Systems Inc.
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
import os
import time
import multiprocessing
import itertools

import networkx as nx
import penaltymodel.core as pm
import dimod

import penaltymodel.cache as pmc

tmp_database_name = 'tmp_test_database_manager_{}.db'.format(time.time())


class TestInterfaceFunctions(unittest.TestCase):
    def setUp(self):
        self.database = pmc.cache_file(filename=tmp_database_name)

    def test_typical(self):
        dbfile = self.database

        # insert a penalty model
        graph = nx.path_graph(3)
        decision_variables = (0, 2)
        feasible_configurations = {(-1, -1): 0., (+1, +1): 0.}
        spec = pm.Specification(graph, decision_variables, feasible_configurations, dimod.SPIN)
        linear = {v: 0 for v in graph}
        quadratic = {edge: -1 for edge in graph.edges}
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        widget = pm.PenaltyModel.from_specification(spec, model, 2., -2)

        # cache the penaltymodel
        pmc.cache_penalty_model(widget, database=dbfile)

        # retrieve it
        widget_ = pmc.get_penalty_model(spec, database=dbfile)

        self.assertEqual(widget_, widget)

    def test_arbitrary_labels(self):
        dbfile = self.database

        # set up a specification and a corresponding penaltymodel
        graph = nx.Graph()
        for i in 'abcd':
            for j in 'efgh':
                graph.add_edge(i, j)

        decision_variables = ('a', 'e')
        feasible_configurations = ((-1, -1), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.SPIN)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 0 for edge in graph.edges}
        if decision_variables in quadratic:
            quadratic[decision_variables] = -1
        else:
            u, v = decision_variables
            assert (v, u) in quadratic
            quadratic[(v, u)] = -1
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # now cache the pmodel to make sure there is something to find
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # now try to retrieve it
        retreived_pmodel = pmc.get_penalty_model(spec, database=dbfile)

        self.assertIs(retreived_pmodel.model.vartype, dimod.SPIN)

        # check that the specification is equal to the retreived_pmodel
        self.assertTrue(spec.__eq__(retreived_pmodel))

    def test_binary_specification(self):
        dbfile = self.database

        # set up a specification and a corresponding penaltymodel
        graph = nx.Graph()
        for i in 'abcd':
            for j in 'efgh':
                graph.add_edge(i, j)

        decision_variables = ('a', 'e')
        feasible_configurations = ((0, 0), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 0 for edge in graph.edges}
        if decision_variables in quadratic:
            quadratic[decision_variables] = -1
        else:
            u, v = decision_variables
            assert (v, u) in quadratic
            quadratic[(v, u)] = -1
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # now cache the pmodel to make sure there is something to find
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # now try to retrieve it
        retreived_pmodel = pmc.get_penalty_model(spec, database=dbfile)

        self.assertIs(retreived_pmodel.model.vartype, dimod.BINARY)

        # check that the specification is equal to the retreived_pmodel
        self.assertTrue(spec.__eq__(retreived_pmodel))

    def test_arbitrary_labels_on_k44(self):
        dbfile = self.database

        graph = nx.Graph()
        for i in range(3):
            for j in range(3, 6):
                graph.add_edge(i, j)

        decision_variables = (0, 5)
        feasible_configurations = ((0, 0), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)

        linear = {v: 0 for v in graph}
        quadratic = {edge: 0 for edge in graph.edges}
        if decision_variables in quadratic:
            quadratic[decision_variables] = -1
        else:
            u, v = decision_variables
            assert (v, u) in quadratic
            quadratic[(v, u)] = -1
        model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # now cache the pmodel to make sure there is something to find

        for thingy in itertools.permutations(range(6)):
            mapping = dict(enumerate(thingy))
            pmodel = pmodel.relabel_variables(mapping, inplace=False)
            pmc.cache_penalty_model(pmodel, database=dbfile)

        # now relabel some variables
        mapping = {1: '1', 2: '2', 3: '3', 4: '4'}

        new_spec = spec.relabel_variables(mapping, inplace=True)

        # retrieve from the new_spec
        # now try to retrieve it
        retreived_pmodel = pmc.get_penalty_model(new_spec, database=dbfile)

    def test_insert_retrieve(self):
        dbfile = self.database

        linear = {'1': 0.0, '0': -0.5, '3': 1.0, '2': -0.5}
        quadratic = {('0', '3'): -1.0, ('1', '2'): 1.0, ('0', '2'): 0.5, ('1', '3'): 1.0}
        offset = 0.0
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.SPIN)

        graph = nx.Graph()
        graph.add_edges_from(quadratic)
        decision_variables = ('0', '2', '3')
        feasible_configurations = ((-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1))
        spec = pm.Specification(graph, decision_variables, feasible_configurations, dimod.SPIN)

        classical_gap = 2
        ground = -2.5

        pmodel = pm.PenaltyModel.from_specification(spec, model, classical_gap, ground)

        pmc.cache_penalty_model(pmodel, database=dbfile)

        # print(spec.feasible_configurations)
        # print(spec.decision_variables)

        # get it back
        ret_pmodel = pmc.get_penalty_model(spec, database=dbfile)

        # now get back one with a different decision_variables
        spec2 = pm.Specification(graph, ('3', '0', '2'), feasible_configurations, dimod.SPIN)
        try:
            ret_pmodel = pmc.get_penalty_model(spec2, database=dbfile)
            self.assertNotEqual(ret_pmodel, pmodel)
        except:
            pass

    def test_one_variable_insert_retrieve(self):
        """Test case when there is no quadratic contribution (i.e. cache will
        receive an empty value for the quadratic contribution)
        """
        dbfile = self.database

        # generate one variable model (i.e. no quadratic terms)
        spec = pm.Specification(graph=nx.complete_graph(1),
                                decision_variables=[0],
                                feasible_configurations=[(-1,)],
                                min_classical_gap=2, vartype='SPIN')
        pmodel = pm.get_penalty_model(spec)

        # insert model into cache
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # retrieve model back from cache
        retrieved_model = pmc.get_penalty_model(spec, database=dbfile)
        self.assertEqual(pmodel, retrieved_model)

    def test_no_linear_bias_insert_retrieve(self):
        """Test case when there is no linear bias"""
        dbfile = self.database

        # define specifications
        spec = pm.Specification(graph=nx.complete_graph(2),
                                decision_variables=[0, 1],
                                feasible_configurations=[(-1, +1), (+1, -1)],
                                min_classical_gap=2, vartype='SPIN')

        # make a model
        # note: model must satisfy specifications and not have a nonzero linear bias
        linear = {}
        quadratic = {(0, 1): 1}
        offset = 0.0
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.SPIN)
        pmodel = pm.PenaltyModel.from_specification(spec, model, 2, -1)

        # insert model into cache
        pmc.cache_penalty_model(pmodel, database=dbfile)

        # retrieve model back from cache
        retrieved_model = pmc.get_penalty_model(spec, database=dbfile)
        self.assertEqual(pmodel, retrieved_model)
