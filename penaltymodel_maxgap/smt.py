"""All functions relating to defining the SMT problem.

All calls to pysmt live in this sub module.
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

    Args:
        x (number): A number to be cast to a pysmt constant.
        max_denominator (int, optional): The maximum size of the denominator.
            Default 1000000.

    Returns:
        A Real constant with the given value and the denominator limited.

    """
    f = Fraction(x).limit_denominator(max_denominator)
    return Real((f.numerator, f.denominator))


def SpinTimes(spin, bias):
    """Define our own multiplication for bias times spins. This allows for
    cleaner log code as well as value checking.

    Args:
        spin (int): -1 or 1
        bias (:class:`pysmt.shortcuts.Symbol`): The bias

    Returns:
        spins * bias

    """
    if not isinstance(spin, int):
        raise TypeError('spin must be an int')
    if spin == -1:
        return Times(Real((-1, 1)), bias)  # -1 / 1
    elif spin == 1:
        # identity
        return bias
    else:
        raise ValueError('expected spins to be -1., or 1.')


class Theta(object):
    """Represents the Binary Quadratic Model with smt Symbols.

    Theta is the collection of linear and quadratic biases and the offset that together
    define a binary quadratic program.

    Args:
        linear (dict[variable, Symbol]): A dict mapping variables to their
            associated linear bias represented by a Symbol.
        quadratic (dict[edge, Symbol]): A dict mapping pairs of variables
            (called an edge here) to their associated quadratic bias represented
            as a Symbol.
        adj (dict[variable, dict[variable, Symbol]]): The adjacency dict for Theta.
        offset (Symbol): The offset for theta represented by a Symbol.
        assertions (set, optional): The set of smt assertions over the set of all
            Symbols associated with theta.

    Notes:
        No input checking is applied to linear, quadratic, offset or assertions.

    """
    def __init__(self, linear, quadratic, offset, assertions=None):
        if assertions is None:
            assertions = set()
        elif not isinstance(assertions, set):
            raise TypeError("expected input 'assertions' to be a set")
        self.assertions = assertions

        self.offset = offset
        self.linear = linear
        self.quadratic = quadratic

        # build adj from quadratic
        self.adj = adj = {v: dict() for v in linear}
        for (u, v), bias in iteritems(quadratic):
            adj[u][v] = bias
            adj[v][u] = bias

    @classmethod
    def from_graph(cls, graph, linear_energy_ranges, quadratic_energy_ranges):
        """Derives Theta from a graph.

        Args:
            graph (:class:`networkx.Graph`): A graph.
            linear_energy_ranges (dict): The linear energy range for each
                linear bias.
            quadratic_energy_ranges (dict): The quadratic energy range for each
                quadratic bias.

        Returns:
            :class:`.Theta`

        """
        assertions = set()

        #
        # there is a real-valued offset
        #
        offset = Symbol('offset', REAL)

        #
        # linear
        #
        def linear_bias(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            assertions.add(LE(bias, limitReal(max_)))
            assertions.add(GE(bias, limitReal(min_)))

            return bias

        linear = {v: linear_bias(v) for v in graph}

        #
        # quadratic
        #
        def quadratic_bias(u, v):
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in quadratic_energy_ranges:
                min_, max_ = quadratic_energy_ranges[(v, u)]
            else:
                min_, max_ = quadratic_energy_ranges[(u, v)]

            assertions.add(LE(bias, limitReal(max_)))
            assertions.add(GE(bias, limitReal(min_)))

            return bias

        quadratic = {(u, v): quadratic_bias(u, v) for u, v in graph.edges}

        return cls(linear, quadratic, offset, assertions)

    def fix_variables(self, spins):
        """Create a new theta for which some spins are fixed.

        Args:
            spins (dict[hash, int])

        Returns:
            :class:`.Theta`: Another Theta that has some spins fixed.

        """
        if not isinstance(spins, dict):
            raise TypeError("expected input 'spins' to be a dict")
        # spins type-checking is handled by SpinTimes

        # offset is initially the same
        offset = self.offset

        # linear
        linear = {}
        for v, bias in iteritems(self.linear):
            if v in spins:
                # fixed linear biases get added to offset
                offset = Plus(offset, Times(limitReal(spins[v]), bias))
            else:
                linear[v] = bias

        # quadratic
        quadratic = {}
        for (u, v), bias in iteritems(self.quadratic):
            if u in spins and v in spins:
                # when we know both values for a quadratic bias, it becomes an offset
                offset = Plus(offset, SpinTimes(spins[v] * spins[u], bias))
            elif u in spins:
                # when we know one value of a quadratic bias, it becomes linear
                linear[v] = Plus(linear[v], SpinTimes(spins[u], bias))
            elif v in spins:
                # when we know one value of a quadratic bias, it becomes linear
                linear[u] = Plus(linear[u], SpinTimes(spins[v], bias))
            else:
                # nothing has changed
                quadratic[(u, v)] = bias

        return Theta(linear, quadratic, offset, self.assertions)

    def energy(self, spins):
        """The formula that calculates the energy of theta.

        Args:
            spins (dict): A dict of the form {v: s, ...} where v is
                every variable in theta and s is -.0 or 1.0.

        Returns:
            The formula for the energy of theta given spins.

        """
        # get the energy of theta with every variable set in spins
        linear_energy = (SpinTimes(spins[v], bias) for v, bias in iteritems(self.linear))
        quadratic_energy = (SpinTimes(spins[v] * spins[u], bias)
                            for (u, v), bias in iteritems(self.quadratic))
        return Plus(itertools.chain(linear_energy, quadratic_energy, [self.offset]))


def _elimination_trees(theta, decision_variables):
    """From Theta and the decision variables, determine the elimination order and the induced
    trees.
    """
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
    """Table of energy relations.

    Args:
        graph (:class:`networkx.Graph`): The graph defining the structure
            of the desired Ising model.
        decision_variables (tuple): The set of nodes in the graph that
            represent decision variables in the desired Ising model.
        linear_energy_ranges (dict[node, (min, max)]): Maps each node to the
            range of the linear bias associated with the variable.
        quadratic_energy_ranges (dict[edge, (min, max)]): Maps each edge to
            the range of the quadratic bias associated with the edge.

    Attributes:
        assertions (set): The set of all smt assertions accumulated by the Table.
        theta (:class:`.Theta`): The linear biases, quadratic biases and the offset.
        gap (Symbol): The smt Symbol representing the classical gap.


    """
    def __init__(self, graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges):
        self.theta = theta = Theta.from_graph(graph, linear_energy_ranges, quadratic_energy_ranges)

        self._trees, self._ancestors = _elimination_trees(theta, decision_variables)

        self.assertions = assertions = theta.assertions

        self._auxvar_counter = itertools.count()  # let's us make fresh aux variables

        self.gap = gap = Symbol('gap', REAL)
        assertions.add(GT(gap, Real(0)))

    def energy_upperbound(self, spins):
        """A formula for an upper bound on the energy of Theta with spins fixed.

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.

        Returns:
            Formula that upper bounds the energy with spins fixed.

        """
        subtheta = self.theta.fix_variables(spins)

        # ok, let's start eliminating variables
        trees = self._trees

        if not trees:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            return subtheta.offset

        energy = Plus(self.message_upperbound(trees, {}, subtheta), subtheta.offset)

        return energy

    def energy(self, spins, break_aux_symmetry=True):
        """A formula for the exact energy of Theta with spins fixed.

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.
            break_aux_symmetry (bool, optional): Default True. If True, break
                the aux variable symmetry by setting all aux variable to 1
                for one of the feasible configurations. If the energy ranges
                are not symmetric then this can make finding models impossible.

        Returns:
            Formula for the exact energy of Theta with spins fixed.

        """
        subtheta = self.theta.fix_variables(spins)

        # we need aux variables
        av = next(self._auxvar_counter)
        auxvars = {v: Symbol('aux{}_{}'.format(av, v), BOOL) for v in subtheta.linear}
        if break_aux_symmetry and av == 0:
            # without loss of generality, we can assume that the aux variables are all
            # spin-up for one configuration
            self.assertions.update(set(itervalues(auxvars)))

        trees = self._trees

        if not trees:
            # if there are no variables to eliminate, then the offset of
            # subtheta is the exact value and we can just return it
            assert not subtheta.linear and not subtheta.quadratic
            return subtheta.offset

        energy = Plus(self.message(trees, {}, subtheta, auxvars), subtheta.offset)

        return energy

    def message(self, tree, spins, subtheta, auxvars):
        """Determine the energy of the elimination tree.

        Args:
            tree (dict): The current elimination tree
            spins (dict): The current fixed spins
            subtheta (dict): Theta with spins fixed.
            auxvars (dict): The auxiliary variables for the given spins.

        Returns:
            The formula for the energy of the tree.

        """
        energy_sources = set()
        for v, children in iteritems(tree):
            aux = auxvars[v]

            assert all(u in spins for u in self._ancestors[v])

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
                            for u in self._ancestors[v]}
            plus_aux = And({aux}.union(ancestor_aux))
            minus_aux = And({Not(aux)}.union(ancestor_aux))

            self.assertions.update({LE(m, plus_energy),
                                    LE(m, minus_energy),
                                    Implies(plus_aux, GE(m, plus_energy)),
                                    Implies(minus_aux, GE(m, minus_energy))
                                    })

            energy_sources.add(m)

        return Plus(energy_sources)

    def message_upperbound(self, tree, spins, subtheta):
        """Determine an upper bound on the energy of the elimination tree.

        Args:
            tree (dict): The current elimination tree
            spins (dict): The current fixed spins
            subtheta (dict): Theta with spins fixed.

        Returns:
            The formula for the energy of the tree.

        """
        energy_sources = set()
        for v, subtree in iteritems(tree):

            assert all(u in spins for u in self._ancestors[v])

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

            energy_sources.add(m)

        return Plus(energy_sources)

    def set_energy(self, spins, target_energy):
        """Set the energy of Theta with spins fixed to taget_energy.

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.
            target_energy (float): The desired energy for Theta with spins fixed.

        Notes:
            Add equality constraint to assertions.

        """
        spin_energy = self.energy(spins)
        self.assertions.add(Equals(spin_energy, limitReal(target_energy)))

    def set_energy_upperbound(self, spins):
        """Upper bound the energy of Theta with spins fixed to be greater than gap.

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.
            target_energy (float): The desired energy for Theta with spins fixed.

        Notes:
            Add equality constraint to assertions.

        """
        spin_energy = self.energy_upperbound(spins)
        self.assertions.add(GE(spin_energy, self.gap))

    def gap_bound_assertion(self, gap_lowerbound):
        """The formula that lower bounds the gap.

        Args:
            gap_lowerbound (float): Return the formula that sets a lower
                bound on the gap.

        """
        return GE(self.gap, limitReal(gap_lowerbound))
