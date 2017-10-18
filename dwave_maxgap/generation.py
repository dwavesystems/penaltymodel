"""
Generation of max-gap penalty models using pysmt.
"""
import itertools

from pysmt.shortcuts import Symbol, GE, LE, Real, Plus, Equals, And
from pysmt.shortcuts import Solver
from pysmt.typing import REAL

from dwave_maxgap.penalty_model import PenaltyModel

__all__ = ['generate_small_no_aux']


class _Theta(object):
    """Encodes the smt variables

    Examples:
        >>> theta = _Theta(nx.complete_graph(2),
        ...                {0: (-2., 2.), 1: (-2., 2)}, {(0, 1): (-1., 1.)})
    """
    def __init__(self, graph, linear_energy_ranges, quadratic_energy_ranges):

        # we want to store all of the smt assertions in a set
        self.assertions = assertions = set()

        # need a stable edgelist
        self.edgelist = edgelist = list(graph.edges())

        # next we need to nodes/linear biases
        def linear_bias(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            return bias

        self.linear = {v: linear_bias(v) for v in graph}

        self.adjacency = adjacency = {v: dict() for v in graph}

        def quadratic_bias(u, v):
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in quadratic_energy_ranges:
                min_, max_ = quadratic_energy_ranges[(v, u)]
            else:
                min_, max_ = quadratic_energy_ranges[(u, v)]

            adjacency[u][v] = bias
            adjacency[v][u] = bias

            assertions.add(LE(bias, Real(max_)))
            assertions.add(GE(bias, Real(min_)))

            return bias

        self.quadratic = {(u, v): quadratic_bias(u, v) for u, v in edgelist}

        self.offset = Symbol('offset', REAL)

    # def __getitem__(self, v):
    #     return self._quadratic[v]

    # def quadratic(self):
    #     for u, v in itertools.combinations(list(self._linear), 2):
    #         if v in self._quadratic[u]
    #         yield u, v, self._quadratic[u][v]

    # def linear(self):
    #     return ((v, bias) for v, bias in self._linear.items())


def generate_ising_no_aux(graph, configurations, decision_variables,
                          linear_energy_ranges=None, quadratic_energy_ranges=None):
    """TODO"""

    # this only works for models with no auxiliary variables and less than
    # 4 nodes
    num_nodes = len(graph)

    if len(decision_variables) != num_nodes:
        raise ValueError("every node must correspond to a decision variable")

    if linear_energy_ranges is None:
        linear_energy_ranges = {v: (-2., 2.) for v in graph}
    if quadratic_energy_ranges is None:
        quadratic_energy_ranges = {(u, v): (-1., 1.) for u, v in graph.edges}

    # ok, this thing is simple so we can just use exact energies for each
    # configuration

    # need the smt variables
    theta = _Theta(graph, linear_energy_ranges, quadratic_energy_ranges)

    assertions = set()
    gap = Symbol('gap', REAL)
    # ok, let's start iterating through all configs!
    for config in itertools.product((-1, 1), repeat=num_nodes):

        # get the exact energy for the configuration
        spins = dict(zip(decision_variables, config))

        linear_energy = Plus([Real(spins[v]) * bias for v, bias in theta.linear.items()])
        quadratic_energy = Plus([Real(spins[u] * spins[v]) * bias
                                 for (u, v), bias in theta.quadratic.items()])
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
         for v, bias in theta.linear.items()}
    J = {edge: float(model.get_py_value(bias).limit_denominator())
         for edge, bias in theta.quadratic.items()}

    return h, J
