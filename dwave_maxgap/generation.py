"""
Generation of max-gap penalty models using pysmt.
"""
import itertools

import networkx as nx
import dwave_networkx as dnx
from six import iteritems

from pysmt.shortcuts import Symbol, GE, LE, Real, Plus, Equals, And
from pysmt.shortcuts import Solver
from pysmt.typing import REAL

from dwave_maxgap.penalty_model import PenaltyModel

__all__ = ['generate_ising_no_aux', 'generate_ising']


class Theta(object):
    """Encodes the smt variables.

    Args:
        graph
        linear_energy_ranges
        quadratic_energy_ranges

    Attributes:
        linear
        quadratic
        adjacency
        offset
        assertions

    """
    def __init__(self, graph, linear_energy_ranges, quadratic_energy_ranges):
        # we need to track all of the range assertions in one place
        # so set up the appropriate file
        self.assertions = assertions = set()

        # there is a real-valued offset
        if graph:
            self.offset = Symbol('offset', REAL)
        else:
            self.offset = Real(0.0)

        # next we need a variable for each of the linear biases
        def linear_bias(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            return bias

        self.linear = {v: linear_bias(v) for v in graph}

        # finally we want the quadratic biases both in an edge
        # and adjacency form
        self.adj = adj = {v: {} for v in graph}

        def quadratic_bias(u, v):
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in quadratic_energy_ranges:
                min_, max_ = quadratic_energy_ranges[(v, u)]
            else:
                min_, max_ = quadratic_energy_ranges[(u, v)]

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            adj[u][v] = bias
            adj[v][u] = bias

            return bias

        self.quadratic = {(u, v): quadratic_bias(u, v) for u, v in graph.edges()}

    def fix_variables(self, spins):
        """TODO"""
        # build a new theta from an empty graph
        subtheta = Theta(nx.Graph(), {}, {})

        # offset is initially the same
        subtheta.offset = self.offset

        # now, for each variable in self, if it is spins then its bias
        # gets added to the offset, otherwise it gets added to subtheta
        for v, bias in iteritems(self.linear):
            if v in spins:
                subtheta.offset = Plus(subtheta.offset, Real(spins[v]) * bias)
            else:
                subtheta.linear[v] = bias

        # and now the quadratic biases get allocated.
        for (u, v), bias in iteritems(self.quadratic):
            if u in spins and v in spins:
                subtheta.offset = Plus(subtheta.offset, Real(spins[v] * spins[u]) * bias)
            elif u in spins:
                subtheta.linear[v] = Plus(subtheta.linear[v], Real(spins[u]) * bias)
            elif v in spins:
                subtheta.linear[u] = Plus(subtheta.linear[u], Real(spins[v]) * bias)
            else:
                subtheta.quadratic[(u, v)] = bias

        # finally build subtheta's adjacency
        adj = subtheta.adj
        for (u, v), bias in iteritems(subtheta.quadratic):
            if u in adj:
                adj[u][v] = bias
            else:
                adj[u] = {v: bias}
            if v in adj:
                adj[v][u] = bias
            else:
                adj[v] = {u: bias}

        return subtheta


class Table(object):
    """TODO"""
    def __init__(self, graph, decision_variables, theta):
        aux_subgraph = graph.subgraph(v for v in graph if v not in decision_variables)
        __, order = dnx.treewidth_branch_and_bound(aux_subgraph)

        self.theta = theta

    def energy_upperbound(self, values):

        subtheta = self.theta.fix_variables(values)

        # now we need to add the elimina
        raise NotImplementedError


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
    theta = Theta(graph, linear_energy_ranges, quadratic_energy_ranges)

    assertions = set()
    gap = Symbol('gap', REAL)
    # ok, let's start iterating through all configs!
    for config in itertools.product((-1, 1), repeat=num_nodes):

        # get the exact energy for the configuration
        spins = dict(zip(decision_variables, config))

        linear_energy = Plus([Real(spins[v]) * bias for v, bias in iteritems(theta.linear)])
        quadratic_energy = Plus([Real(spins[u] * spins[v]) * bias
                                 for (u, v), bias in iteritems(theta.quadratic)])
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
    h = {v: float(model.get_py_value(bias).limit_denominator())
         for v, bias in iteritems(theta.linear)}
    J = {(u, v): float(model.get_py_value(bias).limit_denominator())
         for (u, v), bias in iteritems(theta.quadratic)}
    offset = float(model.get_py_value(theta.offset).limit_denominator())
    gap = float(model.get_py_value(gap).limit_denominator())

    return h, J, offset, gap


def generate_ising(graph, configurations, decision_variables,
                   linear_energy_ranges, quadratic_energy_ranges):
    """TODO"""
    pass

    num_nodes = len(graph)

    # need the smt variables for the biases
    theta = Theta(graph, linear_energy_ranges, quadratic_energy_ranges)

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
