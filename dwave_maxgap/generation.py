"""
Generation of max-gap penalty models using pysmt.
"""
import itertools

import networkx as nx
import dwave_networkx as dnx

from pysmt.shortcuts import Symbol, GE, LE, Real, Plus, Equals, And
from pysmt.shortcuts import Solver
from pysmt.typing import REAL

from dwave_maxgap.penalty_model import PenaltyModel

__all__ = ['generate_ising_no_aux', 'generate_ising']


class Theta(nx.Graph):
    """Encodes the smt variables
    """
    def __init__(self):
        nx.Graph.__init__(self)
        self.assertions = set()
        self.offset = Symbol('offset', REAL)

    def add_linear_biases_from(self, nodes, energy_ranges):

        assertions = self.assertions

        def linear_bias(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = energy_ranges[v]

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            return bias

        self.add_nodes_from((v, {'bias': linear_bias(v)}) for v in nodes)

    def add_quadratic_biases_from(self, edges, energy_ranges):

        assertions = self.assertions

        def quadratic_bias(u, v):
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in energy_ranges:
                min_, max_ = energy_ranges[(v, u)]
            else:
                min_, max_ = energy_ranges[(u, v)]

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            return bias

        self.add_edges_from((u, v, {'bias': quadratic_bias(u, v)}) for u, v in edges)

    def fix_variables(self, values):
        subtheta = self.subgraph(v for v in self if v not in values)
        subtheta.offset = self.offset

        # now we need to add the eliminated values to the offset or linear
        # biases
        for v, val in values.items():
            subtheta.offset = Plus(Real(val) * self.nodes[v]['bias'], subtheta.offset)

        for u, v in self.edges:
            if u in values and v in values:
                subtheta.offset = Plus(Real(values[u] * values[v]) * self[u][v]['bias'], subtheta.offset)
            elif u in values:
                pass


        print(subtheta.offset)


class Table(object):
    """TODO"""
    def __init__(self, graph, decision_variables, theta):
        aux_subgraph = graph.subgraph(v for v in graph if v not in decision_variables)
        __, order = dnx.treewidth_branch_and_bound(aux_subgraph)

        self.theta = theta

    def energy_upperbound(self, values):

        subtheta = self.theta.fix_variables(values)

        # now we need to add the elimina


def generate_ising_no_aux(graph, configurations, decision_variables,
                          linear_energy_ranges, quadratic_energy_ranges):
    """TODO"""

    # this only works for models with no auxiliary variables
    num_nodes = len(graph)

    if len(decision_variables) != num_nodes:
        raise ValueError("every node must correspond to a decision variable")

    # ok, this thing is simple so we can just use exact energies for each
    # configuration

    # need the smt variables
    theta = Theta()
    theta.add_linear_biases_from(graph, linear_energy_ranges)
    theta.add_quadratic_biases_from(graph.edges(), quadratic_energy_ranges)

    assertions = set()
    gap = Symbol('gap', REAL)
    # ok, let's start iterating through all configs!
    for config in itertools.product((-1, 1), repeat=num_nodes):

        # get the exact energy for the configuration
        spins = dict(zip(decision_variables, config))

        linear_energy = Plus([Real(spins[v]) * theta.nodes[v]['bias'] for v in theta])
        quadratic_energy = Plus([Real(spins[u] * spins[v]) * theta[u][v]['bias']
                                 for u, v in theta.edges])
        energy = Plus([linear_energy, quadratic_energy, theta.offset])

        if config in configurations:
            assertions.add(Equals(energy, Real(0.0)))
        else:
            assertions.add(GE(energy, gap))

    assertions.update(theta.assertions)
    # ok, problem is set up, let's get solving

    with Solver() as solver:
        solver.add_assertion(And(assertions))

        g = 0.0
        while solver.solve():
            model = solver.get_model()

            solver.add_assertion(GE(gap, Real(g)))

            g += .1

    # finally we need to convert our values back into python floats.
    # we use limit_denominator to deal with some of the rounding
    # issues.
    h = {v: float(model.get_py_value(theta.nodes[v]['bias']).limit_denominator())
         for v in theta}
    J = {(u, v): float(model.get_py_value(theta[u][v]['bias']).limit_denominator())
         for u, v in theta.edges}
    offset = float(model.get_py_value(theta.offset).limit_denominator())
    gap = float(model.get_py_value(gap).limit_denominator())

    return h, J, offset, gap


def generate_ising(graph, configurations, decision_variables,
                   linear_energy_ranges, quadratic_energy_ranges):
    """TODO"""

    num_nodes = len(graph)

    # need the smt variables
    theta = Theta()
    theta.add_linear_biases_from(graph, linear_energy_ranges)
    theta.add_quadratic_biases_from(graph.edges(), quadratic_energy_ranges)

    # we need to build a table of messages
    table = Table(graph, decision_variables, theta)

    gap = Symbol('gap', REAL)
    for config in itertools.product((-1, 1), repeat=num_nodes):

        # get the exact energy for the configuration
        spins = dict(zip(decision_variables, config))

        if config in configurations:
            energy = table.energy(spins)
            assertions.add(Equals(energy, Real(0.0)))
        else:
            energy = table.energy_upperbound(spins)
            assertions.add(GE(energy, gap))
