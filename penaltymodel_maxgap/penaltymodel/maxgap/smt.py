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
"""All functions relating to defining the SMT problem.

All calls to pysmt live in this sub module.
"""

import itertools
from fractions import Fraction

import dwave_networkx as dnx

from pysmt.shortcuts import Symbol, FreshSymbol, Real
from pysmt.shortcuts import LE, GE, Plus, Times, Implies, Not, And, Equals, GT
from pysmt.typing import REAL, BOOL

from penaltymodel.maxgap.theta import Theta, limitReal


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
        subtheta = self.theta.copy()
        subtheta.fix_variables(spins)

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
        subtheta = self.theta.copy()
        subtheta.fix_variables(spins)

        # we need aux variables
        av = next(self._auxvar_counter)
        auxvars = {v: Symbol('aux{}_{}'.format(av, v), BOOL) for v in subtheta.linear}
        if break_aux_symmetry and av == 0:
            # without loss of generality, we can assume that the aux variables are all
            # spin-up for one configuration
            self.assertions.update(set(auxvars.values()))

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
        for v, children in tree.items():
            aux = auxvars[v]

            assert all(u in spins for u in self._ancestors[v])

            # build an iterable over all of the energies contributions
            # that we can exactly determine given v and our known spins
            # in these contributions we assume that v is positive
            def energy_contributions():
                yield subtheta.linear[v]

                for u, bias in subtheta.adj[v].items():
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
        for v, subtree in tree.items():

            assert all(u in spins for u in self._ancestors[v])

            # build an iterable over all of the energies contributions
            # that we can exactly determine given v and our known spins
            # in these contributions we assume that v is positive
            def energy_contributions():
                yield subtheta.linear[v]

                for u, bias in subtheta.adj[v].items():
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
        """Set the energy of Theta with spins fixed to target_energy.

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.
            target_energy (float): The desired energy for Theta with spins fixed.

        Notes:
            Add equality constraint to assertions.

        """
        spin_energy = self.energy(spins)
        self.assertions.add(Equals(spin_energy, limitReal(target_energy)))

    def set_energy_upperbound(self, spins, offset=0):
        """Upper bound the energy of Theta with spins fixed to be greater than (gap + offset).

        Args:
            spins (dict): Spin values for a subset of the variables in Theta.
            offset (float): A value that is added to the upper bound. Default value is 0.

        Notes:
            Add equality constraint to assertions.

        """
        spin_energy = self.energy_upperbound(spins)
        self.assertions.add(GE(spin_energy, self.gap + offset))

    def gap_bound_assertion(self, gap_lowerbound):
        """The formula that lower bounds the gap.

        Args:
            gap_lowerbound (float): Return the formula that sets a lower
                bound on the gap.

        """
        return GE(self.gap, limitReal(gap_lowerbound))
