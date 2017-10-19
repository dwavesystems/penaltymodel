"""
Generation of max-gap penalty models using pysmt.
"""
import itertools

import networkx as nx
import dwave_networkx as dnx
from six import iteritems

from pysmt.shortcuts import Symbol, FreshSymbol, GE, LE, Real, Plus, Equals, And, Times, Ite
from pysmt.shortcuts import Solver
from pysmt.typing import REAL, BOOL

from dwave_maxgap.penalty_model import PenaltyModel

__all__ = ['generate_ising']


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
                subtheta.offset = Plus(subtheta.offset, Times(Real(spins[v]), bias))
            else:
                subtheta.linear[v] = bias

        # and now the quadratic biases get allocated.
        for (u, v), bias in iteritems(self.quadratic):
            if u in spins and v in spins:
                subtheta.offset = Plus(subtheta.offset, Times(Real(spins[v] * spins[u]), bias))
            elif u in spins:
                subtheta.linear[v] = Plus(subtheta.linear[v], Times(Real(spins[u]), bias))
            elif v in spins:
                subtheta.linear[u] = Plus(subtheta.linear[u], Times(Real(spins[v]), bias))
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
        __, self.order = dnx.treewidth_branch_and_bound(aux_subgraph)

        self.theta = theta

        self.assertions = set()

        self.fresh_auxvar = 0  # let's us make fresh aux variables

    def energy_upperbound(self, values):

        subtheta = self.theta.fix_variables(values)

        # ok, let's start eliminating variables
        order = list(self.order)

        if order:
            return self.message_upperbound(order, {}, subtheta)
        else:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            return subtheta.offset

    def energy(self, values):

        subtheta = self.theta.fix_variables(values)

        # we need aux variables
        av = self.fresh_auxvar
        auxvars = {v: Symbol('aux{}_{}'.format(av, v), BOOL) for v in subtheta.linear}
        self.fresh_auxvar += 1

        # ok, let's start eliminating variables
        order = list(self.order)

        if order:
            return self.message(order, {}, subtheta, auxvars)
        else:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            return subtheta.offset

    def message(self, order, spins, subtheta, auxvars):
        # get the last variable in the elimination order
        v = order.pop()
        aux = auxvars[v]

        # build an iterable over all of the energies contributions
        # that we can exactly determine given v and our known spins
        # in these contributions we assume that v is positive
        def energy_contributions():
            yield subtheta.linear[v]

            for u, bias in iteritems(subtheta.adj[v]):
                if u in spins:
                    yield Times(Real(spins[u]), bias)

        energy = Plus(energy_contributions())

        # if there are no more variables in the order, we can stop
        # otherwise we need the next message variable
        if order:
            spins[v] = 1.0
            plus = self.message(order, spins, subtheta, auxvars)
            spins[v] = -1.0
            minus = self.message(order, spins, subtheta, auxvars)
            del spins[v]
        else:
            plus = minus = Real(0.0)

        # we now need a real-valued smt variable to be our message
        m = FreshSymbol(REAL)

        self.assertions.update({LE(m, Plus(energy, plus)),
                                LE(m, Plus(Times(energy, Real(-1.)), minus)),
                                Ite(aux,
                                    GE(m, Plus(energy, plus)),
                                    GE(m, Plus(Times(energy, Real(-1.)), minus)))
                                })

        order.append(v)

        return m

    def message_upperbound(self, order, spins, subtheta):

        # get the last variable in the elimination order
        v = order.pop()

        # build an iterable over all of the energies contributions
        # that we can exactly determine given v and our known spins
        # in these contributions we assume that v is positive
        def energy_contributions():
            yield subtheta.linear[v]

            for u, bias in iteritems(subtheta.adj[v]):
                if u in spins:
                    yield Times(Real(spins[u]), bias)

        energy = Plus(energy_contributions())

        # if there are no more variables in the order, we can stop
        # otherwise we need the next message variable
        if order:
            spins[v] = 1.
            plus = self.message_upperbound(order, spins, subtheta)
            spins[v] = -1.
            minus = self.message_upperbound(order, spins, subtheta)
            del spins[v]
        else:
            plus = minus = Real(0.0)

        # we now need a real-valued smt variable to be our message
        m = FreshSymbol(REAL)

        self.assertions.update({LE(m, Plus(energy, plus)),
                                LE(m, Plus(Times(energy, Real(-1.)), minus))})

        order.append(v)

        return m


def generate_ising(graph, configurations, decision_variables,
                   linear_energy_ranges, quadratic_energy_ranges):
    """TODO"""
    pass

    num_nodes = len(graph)
    num_variables = len(decision_variables)

    # need the smt variables for the biases
    theta = Theta(graph, linear_energy_ranges, quadratic_energy_ranges)

    # we need to build a table of messages
    table = Table(graph, decision_variables, theta)

    assertions = set()
    gap = Symbol('gap', REAL)
    for config in itertools.product((-1, 1), repeat=num_variables):
        # get the exact energy for the configuration
        spins = dict(zip(decision_variables, config))

        if config in configurations:
            energy = table.energy(spins)
            assertions.add(Equals(energy, Real(0.0)))
        else:
            energy = table.energy_upperbound(spins)
            assertions.add(GE(energy, gap))

    assertions.update(theta.assertions)
    assertions.update(table.assertions)

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
