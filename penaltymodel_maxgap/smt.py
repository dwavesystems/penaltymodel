"""module abstracts setting up the smt problem.
"""
import itertools
from fractions import Fraction

from six import iteritems, itervalues

import dwave_networkx as dnx

from pysmt.shortcuts import Symbol, FreshSymbol, Real
from pysmt.shortcuts import LE, GE, Plus, Times, Implies, Not, And, Equals, GT
from pysmt.typing import REAL, BOOL


def limitReal(x, max_denominator=1000000):
    """Creates an pysmt Real constant from x.

    Casts x to the nearest fraction that has denominator at most
    max_denominator.
    """
    f = Fraction(x).limit_denominator(max_denominator)
    return Real((f.numerator, f.denominator))


def SpinTimes(spin, bias):
    """Define our own multiplication for bias times spins. This allows for
    cleaner log code as well as value checking.

    Args:
        spin (int): -1 or 1
        bias (pysmt.shortcuts.Symbol): The bias

    Returns:
        spins * bias

    """
    if not isinstance(spin, int):
        raise TypeError('spin must be an int')
    if spin == -1:
        return Times(Real((-1, 1)), bias)
    elif spin == 1:
        # identity
        return bias
    else:
        raise ValueError('expected spins to be -1., or 1.')


class Theta(object):
    """Encodes the smt variables.

    Args:
        graph
        linear_energy_ranges
        quadratic_energy_ranges

    Attributes:
        linear
        quadratic
        adj
        offset
        assertions

    """
    def __init__(self):
        self.assertions = set()
        self.offset = limitReal(0)
        self.linear = {}
        self.quadratic = {}
        self.adj = {}

    def build_from_graph(self, graph, linear_energy_ranges, quadratic_energy_ranges):
        # we need to track all of the range assertions in one place
        # so set up the appropriate file
        assertions = self.assertions

        # there is a real-valued offset
        self.offset = Symbol('offset', REAL)

        # next we need a variable for each of the linear biases
        def linear_bias(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            assertions.add(LE(bias, limitReal(max_)))
            assertions.add(GE(bias, limitReal(min_)))
            smtlog.debug('{} <= {} <= {}'.format(min_, bias, max_))

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

            assertions.add(LE(bias, limitReal(max_)))
            assertions.add(GE(bias, limitReal(min_)))
            smtlog.debug('{} <= {} <= {}'.format(min_, bias, max_))

            adj[u][v] = bias
            adj[v][u] = bias

            return bias

        self.quadratic = {(u, v): quadratic_bias(u, v) for u, v in graph.edges()}

    def fix_variables(self, spins):
        """TODO"""
        # build a new theta from an empty graph
        subtheta = Theta()

        # offset is initially the same
        subtheta.offset = self.offset

        # now, for each variable in self, if it is spins then its bias
        # gets added to the offset, otherwise it gets added to subtheta
        for v, bias in iteritems(self.linear):
            if v in spins:
                subtheta.offset = Plus(subtheta.offset, Times(limitReal(spins[v]), bias))
            else:
                subtheta.linear[v] = bias

        # and now the quadratic biases get allocated.
        for (u, v), bias in iteritems(self.quadratic):
            if u in spins and v in spins:
                subtheta.offset = Plus(subtheta.offset, SpinTimes(spins[v] * spins[u], bias))
            elif u in spins:
                subtheta.linear[v] = Plus(subtheta.linear[v], SpinTimes(spins[u], bias))
            elif v in spins:
                subtheta.linear[u] = Plus(subtheta.linear[u], SpinTimes(spins[v], bias))
            else:
                subtheta.quadratic[(u, v)] = bias

        # finally build subtheta's adjacency
        adj = subtheta.adj
        for v in subtheta.linear:
            adj[v] = {}
        for (u, v), bias in iteritems(subtheta.quadratic):
            adj[u][v] = bias
            adj[v][u] = bias

        return subtheta

    def energy(self, spins):
        """The formula that calculates the energy of theta.

        Args:
            spins (dict): A dict of the form {v: s, ...} where v is
                every variable in theta and s is -.0 or 1.0.

        Returns:
            The formula for the energy of theta given spins.

        Raises:
            KeyError: If and v in theta is not in spins.
            ValueError: If any spin is not -1.0 or 1.0.

        """
        # get the energy of theta with every variable set in spins
        linear_energy = (SpinTimes(spins[v], bias) for v, bias in iteritems(self.linear))
        quadratic_energy = (SpinTimes(spins[v] * spins[u], bias)
                            for (u, v), bias in iteritems(self.quadratic))
        return Plus(itertools.chain(linear_energy, quadratic_energy, [self.offset]))


# def _determine_elimination(graph, decision_variables):
#     """get the elimination order and the induced elimination sets
#     for the auxiliary subgraph.
#     """
#     # auxiliary variables are any variables that are not decision
#     auxiliary_variables = set(n for n in graph if n not in decision_variables)

#     # get the adjacency of the auxiliary subgraph
#     adj = {v: {u for u in graph[v] if u in auxiliary_variables}
#            for v in graph if v in auxiliary_variables}

#     # get the elimination order that minimizes treewidth
#     tw, order = dnx.treewidth_branch_and_bound(adj)

#     # we need the elimination set, that is the set of variables that determine
#     # the spin of v for each v in order
#     elimination_sets = {}
#     for n in order:
#         elimination_sets[n] = tuple(adj[n])

#         # now make v simplicial by making its neighborhood a clique, then
#         # continue
#         neighbors = adj[n]
#         for u, v in itertools.combinations(neighbors, 2):
#             adj[u].add(v)
#             adj[v].add(u)
#         for v in neighbors:
#             adj[v].discard(n)
#         del adj[n]

#     assert tw == max(len(es) for es in elimination_sets.values())

#     return order, elimination_sets


def _elimination_trees(theta, decision_variables):
    # auxiliary variables are any variables that are not decision
    auxiliary_variables = set(n for n in theta.linear if n not in decision_variables)

    # get the adjacency of the auxiliary subgraph
    adj = {v: {u for u in theta.adj[v] if u in auxiliary_variables}
           for v in theta.adj if v in auxiliary_variables}

    # get the elimination order that minimizes treewidth
    tw, order = dnx.treewidth_branch_and_bound(adj)

    ancestors = {}
    for n in order:
        ancestors[n] = set(adj[n])

        # now make v simplicial by making its neighborhood a clique, then
        # continue
        neighbors = adj[n]
        for u, v in itertools.combinations(neighbors, 2):
            adj[u].add(v)
            adj[v].add(u)
        for v in neighbors:
            adj[v].discard(n)
        del adj[n]

    roots = {}
    nodes = {v: {} for v in ancestors}
    for vidx in range(len(order) - 1, -1, -1):
        v = order[vidx]

        if ancestors[v]:
            for u in order[vidx + 1:]:
                if u in ancestors[v]:
                    # v is a child of u
                    nodes[u][v] = nodes[v]  # nodes[u][v] = children of v
                    break
        else:
            roots[v] = nodes[v]  # roots[v] = children of v

    return roots, ancestors


class Table(object):
    """TODO"""
    def __init__(self, graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges):
        # self.order, self.elimination_sets = _determine_elimination(graph, decision_variables)

        smtlog.debug(';;; NEW TABLE')

        self.theta = theta = Theta()
        theta.build_from_graph(graph, linear_energy_ranges, quadratic_energy_ranges)

        self.trees, self.ancestors = _elimination_trees(theta, decision_variables)

        self.assertions = assertions = theta.assertions

        self.fresh_auxvar = 0  # let's us make fresh aux variables

        self.gap = gap = Symbol('gap', REAL)
        assertions.add(GT(gap, Real(0)))

    def energy_upperbound(self, values):

        smtlog.debug(';;; determining energy upper bound for {}'.format(values))

        subtheta = self.theta.fix_variables(values)

        # ok, let's start eliminating variables
        trees = self.trees

        if trees:
            energy = Plus(self.message_upperbound(trees, {}, subtheta), subtheta.offset)
            smtlog.debug(';;; energy <= %s', energy)
            return energy
        else:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            return subtheta.offset

    def energy(self, values, break_aux_symmetry=True):
        # NB: only break aux symmetry with symmetric energy ranges

        smtlog.debug(';;; determining energy for {}'.format(values))

        subtheta = self.theta.fix_variables(values)

        # we need aux variables
        av = self.fresh_auxvar
        auxvars = {v: Symbol('aux{}_{}'.format(av, v), BOOL) for v in subtheta.linear}
        if break_aux_symmetry and av == 0:
            # without loss of generatlity, we can assume that the aux variables are all
            # spin-up for one configuration
            self.assertions.update(set(itervalues(auxvars)))
            for bias in itervalues(auxvars):
                smtlog.debug('%s', bias)

        self.fresh_auxvar += 1

        trees = self.trees

        if trees:
            energy = Plus(self.message(trees, {}, subtheta, auxvars), subtheta.offset)
            smtlog.debug(';;; energy == %s', energy)
            return energy
        else:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            smtlog.debug(';;; energy == %s', subtheta.offset)
            return subtheta.offset

    def message(self, tree, spins, subtheta, auxvars):
        # given the current tree, determine the energy

        energy_sources = set()
        for v, children in iteritems(tree):
            aux = auxvars[v]

            assert all(u in spins for u in self.ancestors[v])

            # build an iterable over all of the energies contributions
            # that we can exactly determine given v and our known spins
            # in these contributions we assume that v is positive
            def energy_contributions():
                yield subtheta.linear[v]

                for u, bias in iteritems(subtheta.adj[v]):
                    if u in spins:
                        yield SpinTimes(spins[u], bias)

            plus_energy = Plus(energy_contributions())
            minus_energy = SpinTimes(-1, plus_energy)

            # if the variable has children, we need to recursively determine their energies
            if children:
                # set v to be positive
                spins[v] = 1
                plus_energy = Plus(plus_energy, self.message(children, spins, subtheta, auxvars))
                spins[v] = -1
                minus_energy = Plus(minus_energy, self.message(children, spins, subtheta, auxvars))
                del spins[v]

            # we now need a real-valued smt variable to be our message
            m = FreshSymbol(REAL)

            ancestor_aux = {auxvars[u] if spins[u] > 0 else Not(auxvars[u])
                            for u in self.ancestors[v]}
            plus_aux = And({aux}.union(ancestor_aux))
            minus_aux = And({Not(aux)}.union(ancestor_aux))

            self.assertions.update({LE(m, plus_energy),
                                    LE(m, minus_energy),
                                    Implies(plus_aux, GE(m, plus_energy)),
                                    Implies(minus_aux, GE(m, minus_energy))
                                    })
            smtlog.debug(';;; v={}, message={}, fixed={}'.format(v, m, spins))
            smtlog.debug('%s <= %s', m, plus_energy)
            smtlog.debug('%s <= %s', m, minus_energy)
            smtlog.debug('%s implies %s >= %s', plus_aux, m, plus_energy)
            smtlog.debug('%s implies %s >= %s', minus_aux, m, minus_energy)

            energy_sources.add(m)

        return Plus(energy_sources)

    def message_upperbound(self, tree, spins, subtheta):

        energy_sources = set()
        for v, subtree in iteritems(tree):

            assert all(u in spins for u in self.ancestors[v])

            # build an iterable over all of the energies contributions
            # that we can exactly determine given v and our known spins
            # in these contributions we assume that v is positive
            def energy_contributions():
                yield subtheta.linear[v]

                for u, bias in iteritems(subtheta.adj[v]):
                    if u in spins:
                        yield Times(limitReal(spins[u]), bias)

            energy = Plus(energy_contributions())

            # if there are no more variables in the order, we can stop
            # otherwise we need the next message variable
            if subtree:
                spins[v] = 1.
                plus = self.message_upperbound(subtree, spins, subtheta)
                spins[v] = -1.
                minus = self.message_upperbound(subtree, spins, subtheta)
                del spins[v]
            else:
                plus = minus = limitReal(0.0)

            # we now need a real-valued smt variable to be our message
            m = FreshSymbol(REAL)

            self.assertions.update({LE(m, Plus(energy, plus)),
                                    LE(m, Plus(Times(energy, limitReal(-1.)), minus))})
            smtlog.debug('%s <= %s', m, Plus(energy, plus))
            smtlog.debug('%s <= %s', m, Plus(Times(energy, limitReal(-1.)), minus))

            energy_sources.add(m)

        return Plus(energy_sources)

    def set_energy(self, spins, target_energy):
        """TODO"""
        spin_energy = self.energy(spins)
        self.assertions.add(Equals(spin_energy, Real(target_energy)))

    def set_energy_upperbound(self, spins):
        spin_energy = self.energy_upperbound(spins)
        self.assertions.add(GE(spin_energy, self.gap))

    def gap_bound_assertion(self, gap_lowerbound):
        return GE(self.gap, limitReal(gap_lowerbound))
