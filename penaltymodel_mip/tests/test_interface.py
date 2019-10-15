# Copyright 2018 D-Wave Systems Inc.
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
import itertools

import networkx as nx
import penaltymodel.core as pm
import dimod

import penaltymodel.mip as mip


class TestInterface(unittest.TestCase):
    """We assume that the generation code works correctly.
    Test that the interface gives a penalty model corresponding to the specification"""
    def test_typical(self):
        graph = nx.complete_graph(3)
        spec = pm.Specification(graph, [0, 1], {(-1, -1): 0, (+1, +1): 0}, dimod.SPIN)

        widget = mip.get_penalty_model(spec)

        # some quick test to see that the penalty model propogated in
        for v in graph:
            self.assertIn(v, widget.model.linear)
        for (u, v) in graph.edges:
            self.assertIn(u, widget.model.adj[v])

    def test_binary_specification(self):
        graph = nx.Graph()
        for i in range(4):
            for j in range(4, 8):
                graph.add_edge(i, j)

        decision_variables = (0, 1)
        feasible_configurations = ((0, 0), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=dimod.BINARY)
        widget = mip.get_penalty_model(spec)

        self.assertIs(widget.model.vartype, dimod.BINARY)

        # test the correctness of the widget
        energies = {}
        for decision_config in itertools.product((0, 1), repeat=2):
            energies[decision_config] = float('inf')

            for aux_config in itertools.product((0, 1), repeat=6):
                sample = dict(enumerate(decision_config + aux_config))
                energy = widget.model.energy(sample)

                energies[decision_config] = min(energies[decision_config], energy)

        for decision_config, energy in energies.items():
            if decision_config in feasible_configurations:
                self.assertAlmostEqual(energy, widget.ground_energy)
            else:
                self.assertGreaterEqual(energy, widget.ground_energy + widget.classical_gap - 10**-6)

    def test_and_on_k44(self):
        graph = nx.Graph()
        for i in range(3):
            for j in range(3, 6):
                graph.add_edge(i, j)

        decision_variables = (0, 2, 3)
        feasible_configurations = AND(2)

        mapping = {0: '0', 1: '1', 2: '2', 3: '3'}
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = tuple(mapping[x] for x in decision_variables)

        spin_configurations = tuple([tuple([2 * i - 1 for i in b]) for b in feasible_configurations])
        spec = pm.Specification(graph, decision_variables, spin_configurations, vartype=dimod.SPIN)

        pm0 = mip.get_penalty_model(spec)

        self.check_generated_ising_model(pm0.feasible_configurations, pm0.decision_variables,
                                         pm0.model.linear, pm0.model.quadratic, pm0.ground_energy - pm0.model.offset,
                                         pm0.classical_gap)

    def test_eight_variable(self):

        def f(a, b, c, d, e, f, g, h):
            if a and b:
                return False
            if c and d:
                return False
            if e and f:
                return False
            return not (g and h)

        configs = {config for config in itertools.product((0, 1), repeat=8) if f(*config)}
        decision = list('abcdefgh')
        spec = pm.Specification(nx.complete_graph(decision), decision, configs, pm.BINARY)

        model = mip.get_penalty_model(spec)

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        bqm.add_variables_from((v, 1.0) for v in decision)
        bqm.add_interactions_from((u, v, 0.0) for u, v in itertools.combinations(decision, 2))
        bqm.add_interaction('a', 'b', 1)
        bqm.add_interaction('c', 'd', 1)
        bqm.add_interaction('e', 'f', 1)
        bqm.add_interaction('g', 'h', 1)

        bqm.add_offset(4)

        self.assertEqual(model.model.spin, bqm)

    def check_generated_ising_model(self, feasible_configurations, decision_variables,
                                    linear, quadratic, ground_energy, infeasible_gap):
        """Check that the given Ising model has the correct energy levels"""
        if not feasible_configurations:
            return

        from dimod import ExactSolver

        response = ExactSolver().sample_ising(linear, quadratic)

        # samples are returned in order of energy
        sample, ground = next(iter(response.data(['sample', 'energy'])))
        gap = float('inf')

        self.assertIn(tuple(sample[v] for v in decision_variables), feasible_configurations)

        seen_configs = set()

        for sample, energy in response.data(['sample', 'energy']):
            config = tuple(sample[v] for v in decision_variables)

            # we want the minimum energy for each config of the decisison variables,
            # so once we've seen it once we can skip
            if config in seen_configs:
                continue

            if config in feasible_configurations:
                self.assertAlmostEqual(energy, ground)
                seen_configs.add(config)
            else:
                gap = min(gap, energy - ground)

        self.assertAlmostEqual(ground_energy, ground)
        self.assertAlmostEqual(gap, infeasible_gap)


def AND(n):
    # AND of n inputs
    feas = list()
    for x in itertools.product((0, 1), repeat=n):
        y = int(all(x))
        feas.append(x + (y,))
    return tuple(feas)
