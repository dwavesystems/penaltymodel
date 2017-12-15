from collections import defaultdict
import itertools
from fractions import Fraction

import pysmt.test
from pysmt.shortcuts import GT, LT, And, Equals, GE, LE, Not

import networkx as nx
import dwave_networkx as dnx

from penaltymodel_maxgap.smt import Theta, Table, limitReal


F = lambda x: Fraction(x).limit_denominator()


class TestTheta(pysmt.test.TestCase):

    def setUp(self):
        pysmt.test.TestCase.setUp(self)
        # create a set of graphs to test against
        graphs = [nx.complete_graph(5), dnx.chimera_graph(1), nx.path_graph(12), nx.Graph(), nx.complete_graph(1)]

        disconnect1 = nx.Graph()
        disconnect1.add_nodes_from([0, 1])
        graphs.append(disconnect1)

        disconnect2 = nx.complete_graph(4)
        disconnect2.add_edge(4, 5)
        graphs.append(disconnect2)

        self.graphs = graphs

    def test_theta_construction(self):
        """Check that everything in theta as built correctly"""

        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        for graph in self.graphs:
            theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

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

            # check that the edges are unique
            for u, v in theta.quadratic:
                self.assertNotIn((v, u), theta.quadratic)

            # check that each bias is unique
            self.assertEqual(len(set(id(bias) for __, bias in theta.linear.items())),
                             len(theta.linear))
            self.assertEqual(len(set(id(bias) for __, bias in theta.quadratic.items())),
                             len(theta.quadratic))

    def test_energy_ranges(self):
        """Check that the energy ranges were set the way we expect"""
        linear_ranges = defaultdict(lambda: (-2., 2.))
        quadratic_ranges = defaultdict(lambda: (-1., 1.))

        for graph in self.graphs:
            theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

            for v, bias in theta.linear.items():
                min_, max_ = linear_ranges[v]
                self.assertUnsat(And(GT(bias, limitReal(max_)), And(theta.assertions)))
                self.assertUnsat(And(LT(bias, limitReal(min_)), And(theta.assertions)))

            for (u, v), bias in theta.quadratic.items():
                min_, max_ = quadratic_ranges[(u, v)]
                self.assertUnsat(And(GT(bias, limitReal(max_)), And(theta.assertions)))
                self.assertUnsat(And(LT(bias, limitReal(min_)), And(theta.assertions)))

    def test_energy(self):

        # set the values exactly
        linear_ranges = defaultdict(lambda: (1., 1.))
        quadratic_ranges = defaultdict(lambda: (-1., -1.))

        for graph in self.graphs:
            theta = Theta.from_graph(graph, linear_ranges, quadratic_ranges)

            spins = {v: 1 for v in graph}

            # classical energy
            energy = 6.  # offset = 6
            for v in graph:
                energy += spins[v] * linear_ranges[v][0]
            for u, v in graph.edges:
                energy += spins[v] * spins[u] * quadratic_ranges[(u, v)][0]

            smt_energy = theta.energy(spins)

            self.assertSat(And([Equals(limitReal(energy), smt_energy),
                                And(theta.assertions),
                                Equals(theta.offset, limitReal(6.))]))
            self.assertUnsat(And([Equals(limitReal(energy), smt_energy),
                                  And(theta.assertions),
                                  Equals(theta.offset, limitReal(6.1))]))

            # let's also test the energy of subtheta
            # fixing all of the variable puts the whole value into offset
            subtheta = theta.fix_variables(spins)
            self.assertSat(And([Equals(limitReal(energy), subtheta.offset),
                                And(theta.assertions),
                                Equals(theta.offset, limitReal(6.))]))
            self.assertUnsat(And([Equals(limitReal(energy), subtheta.offset),
                                  And(theta.assertions),
                                  Equals(theta.offset, limitReal(6.1))]))

            # finally let's try fixing a subset of variables
            if len(graph) < 3:
                continue

            subspins = {0: 1, 1: 1}
            subtheta = theta.fix_variables(subspins)

            smt_energy = subtheta.energy(spins)

            self.assertSat(And([Equals(limitReal(energy), smt_energy),
                                And(theta.assertions),
                                Equals(theta.offset, limitReal(6.))]))
            self.assertUnsat(And([Equals(limitReal(energy), smt_energy),
                                  And(theta.assertions),
                                  Equals(theta.offset, limitReal(6.1))]))


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
