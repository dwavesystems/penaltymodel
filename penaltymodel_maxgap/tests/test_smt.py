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
import itertools
from fractions import Fraction

import pysmt.test
from pysmt.shortcuts import GT, LT, And, Equals, GE, LE, Not

import networkx as nx
import dwave_networkx as dnx

from penaltymodel.maxgap.smt import Theta, Table, limitReal


def F(x): return Fraction(x).limit_denominator()


class TestTable(pysmt.test.TestCase):
    def test_trivial(self):
        #
        # empty graph
        #

        graph = nx.Graph()
        decision_variables = []
        h = {}
        J = {}
        self.check_table_energies_exact(graph, decision_variables, h, J)

        #
        # singleton graph
        #
        graph = nx.complete_graph(1)
        decision_variables = (0, )
        h = {0: -.5}
        J = {}
        self.check_table_energies_exact(graph, decision_variables, h, J)

        #
        # one edge
        #
        graph = nx.complete_graph(2)
        h = {0: -.56, 1: .43}
        J = {(0, 1): -1}

        decision_variables = (0,)
        self.check_table_energies_exact(graph, decision_variables, h, J)
        decision_variables = (1,)
        self.check_table_energies_exact(graph, decision_variables, h, J)
        decision_variables = (0, 1)
        self.check_table_energies_exact(graph, decision_variables, h, J)

        #
        # two disconnected nodes
        #
        graph = nx.complete_graph(1)
        graph.add_node(1)
        h = {0: -.56, 1: .43}
        J = {}

        decision_variables = (0,)
        self.check_table_energies_exact(graph, decision_variables, h, J)
        decision_variables = (1,)
        self.check_table_energies_exact(graph, decision_variables, h, J)
        decision_variables = (0, 1)
        self.check_table_energies_exact(graph, decision_variables, h, J)

    def test_basic(self):
        """more interesting graph types"""

        #
        # path graph with decision on an end
        #
        graph = nx.path_graph(3)
        decision_variables = (0, )
        h = {0: -1, 1: 1, 2: -1}
        J = {(u, v): 1 for u, v in graph.edges}
        self.check_table_energies_exact(graph, decision_variables, h, J)

        #
        # path graph with decision in the middle
        #
        graph = nx.path_graph(5)
        decision_variables = (2, )
        h = {0: .1, 2: .4, 1: -.1, 3: 0, 4: 8}
        J = {(u, v): 1 for u, v in graph.edges}
        self.check_table_energies_exact(graph, decision_variables, h, J)

    def test_typical(self):
        """check tables for a chimera tile"""
        graph = dnx.chimera_graph(1)
        decision_variables = (0, 1, 2)
        h = {0: 1, 1: -1, 2: 1, 3: -1, 4: 1, 5: 0, 6: .4, 7: -1.3}
        J = {(u, v): 1 for u, v in graph.edges}
        self.check_table_energies_exact(graph, decision_variables, h, J)

    def check_table_energies_exact(self, graph, decision_variables, h, J):
        """For a given ising problem, check that the table gives the correct
        energies when linear and quadratic energies are specified exactly.
        """

        # determine the aux variables
        aux_variables = tuple(v for v in graph if v not in decision_variables)

        # now generate a theta that sets linear and quadratic equal to h, J
        linear_ranges = {v: (bias, bias) for v, bias in h.items()}
        quadratic_ranges = {edge: (bias, bias) for edge, bias in J.items()}

        # and now the table
        table = Table(graph, decision_variables, linear_ranges, quadratic_ranges)

        # ok, time to do some energy calculations
        for config in itertools.product((-1, 1), repeat=len(decision_variables)):
            spins = dict(zip(decision_variables, config))

            # first we want to know the minimum classical energy
            energy = float('inf')
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):

                aux_spins = dict(zip(aux_variables, aux_config))
                aux_spins.update(spins)
                aux_energy = ising_energy(h, J, aux_spins)

                if aux_energy < energy:
                    energy = aux_energy

            # collect assertions
            assertions = table.assertions
            theta = table.theta  # so we can set the offset directly

            # table assertions by themselves should be SAT
            self.assertSat(And(assertions))

            # ok, does the exact energy calculated by the table match?
            table_energy = table.energy(spins, break_aux_symmetry=False)
            for offset in [0, -.5, -.3, -.1, .23, 1, 106]:

                self.assertSat(And([Equals(table_energy, limitReal(energy + offset)),
                                    And(assertions),
                                    Equals(theta.offset, limitReal(offset))]),
                               msg='exact energy equality is not SAT')
                self.assertUnsat(And([Not(Equals(table_energy, limitReal(energy + offset))),
                                      And(assertions),
                                      Equals(theta.offset, limitReal(offset))]),
                                 msg='exact energy inequality is not UNSAT')

            # how about the upper bound?
            table_energy_upperbound = table.energy_upperbound(spins)
            for offset in [-.5, -.3, -.1, 0, .23, 1, 106]:
                self.assertSat(And([GE(table_energy_upperbound, limitReal(energy + offset)),
                                    And(assertions),
                                    Equals(theta.offset, limitReal(offset))]),
                               msg='energy upperbound is not SAT')


def ising_energy(h, J, sample):
    energy = F(0)

    # add the contribution from the linear biases
    for v in h:
        energy += F(h[v]) * sample[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        energy += F(J[(v0, v1)]) * sample[v0] * sample[v1]

    return energy
