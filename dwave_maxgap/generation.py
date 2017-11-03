"""
Generation of max-gap penalty models using pysmt.
"""
import itertools
from collections import defaultdict

from six import iteritems

from pysmt.shortcuts import Equals, GE, Real, Solver, And, GT, Plus, Max, Times

from dwave_maxgap.smt import Theta, Table, allocate_gap, limitReal

__all__ = ['generate_ising']


def generate_ising(graph, configurations, decision_variables,
                   linear_energy_ranges=None, quadratic_energy_ranges=None):
    """TODO"""

    if linear_energy_ranges is None:
        linear_energy_ranges = defaultdict(lambda: (-2., 2.))
    if quadratic_energy_ranges is None:
        quadratic_energy_ranges = defaultdict(lambda: (-1., 1.))

    num_nodes = len(graph)
    num_variables = len(decision_variables)

    if any(len(config) != num_variables for config in configurations):
        raise ValueError('mismatched configurations and decision_variables')

    # we need to build a table of messages
    table = Table(graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges)

    assertions = set()
    gap = allocate_gap()
    for config in itertools.product((-1, 1), repeat=num_variables):
        # get the exact energy for the configuration
        assert len(config) == len(decision_variables)

        spins = dict(zip(decision_variables, config))

        if config in configurations:
            energy = table.energy(spins)
            assertions.add(Equals(energy, Real(0.0)))
        else:
            energy = table.energy_upperbound(spins)
            assertions.add(GE(energy, gap))

    assertions.update(table.assertions)

    theta = table.theta

    # we also want to go ahead and state the the gap is > 0
    assertions.add(GT(gap, Real(0)))

    with Solver() as solver:

        solver.add_assertion(And(assertions))

        if solver.solve():

            gmin = 0
            gmax = sum(max(abs(r) for r in linear_energy_ranges[v]) for v in graph)
            gmax += sum(max(abs(r) for r in quadratic_energy_ranges[(u, v)])
                        for (u, v) in graph.edges)

            # gmax = 6
            g = 2  # our desired gap

            while abs(gmax - gmin) >= .01:
                solver.push()

                solver.add_assertion(GE(gap, limitReal(g)))

                if solver.solve():
                    model = solver.get_model()
                    gmin = float(model.get_py_value(gap).limit_denominator())
                else:
                    solver.pop()
                    gmax = g

                g = min(gmin + .1, (gmax + gmin) / 2)

        else:
            raise NotImplementedError('no model found')

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
