import itertools
from collections import defaultdict

from six import iteritems
from pysmt.shortcuts import Solver

from dwave_maxgap.smt import Theta, Table, limitReal

__all__ = ['generate_ising']


def generate_ising(graph, feasible_configurations, decision_variables,
                   linear_energy_ranges=None, quadratic_energy_ranges=None):
    """Generates the Ising model that induces the given feasible configurations.

    Args:
        graph (nx.Graph): The target graph on which the Ising model is build.
        feasible_configurations (set/dict): The set of feasible configurations
            of the decision variables. If a set it is assumes that the energy
            for each feasible configuration should be ground. If a dict, then
            the value is the target energy.
        decision_variables (list/tuple): Which variables in the graph are
            assigned as decision variables.
        linear_energy_ranges (dict, optional): A dict of the form
            {v: (min_, max_), ...} where min_ and max_ are the range
            of values allowed to v. Default is (-2., 2.) for each v.
        quadratic_energy_ranges (dict, optional): A dict of the form
            {(u, v): (min_, max_), ...} where min_ and max_ are the range
            of values allowed to (u, v). Default is (-1., 1.) for each
            edge (u, v).

    """

    if linear_energy_ranges is None:
        linear_energy_ranges = defaultdict(lambda: (-2., 2.))
    if quadratic_energy_ranges is None:
        quadratic_energy_ranges = defaultdict(lambda: (-1., 1.))

    num_nodes = len(graph)
    num_variables = len(decision_variables)

    if any(len(config) != num_variables for config in feasible_configurations):
        raise ValueError('mismatched feasible_configurations and decision_variables lengths')

    # we need to build a table of messages
    table = Table(graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges)

    for config in itertools.product((-1, 1), repeat=num_variables):
        # get the exact energy for the configuration
        assert len(config) == len(decision_variables)

        spins = dict(zip(decision_variables, config))

        if config in feasible_configurations:
            table.set_energy(spins, 0.0)
        else:
            table.set_energy_upperbound(spins)

    with Solver() as solver:

        for assertion in table.assertions:
            solver.add_assertion(assertion)

        if solver.solve():

            gmin = 0
            gmax = sum(max(abs(r) for r in linear_energy_ranges[v]) for v in graph)
            gmax += sum(max(abs(r) for r in quadratic_energy_ranges[(u, v)])
                        for (u, v) in graph.edges)

            # gmax = 6
            g = 2  # our desired gap

            while abs(gmax - gmin) >= .01:
                solver.push()

                gap_assertion = table.gap_bound_assertion(g)
                solver.add_assertion(gap_assertion)

                if solver.solve():
                    model = solver.get_model()
                    gmin = float(model.get_py_value(table.gap).limit_denominator())
                else:
                    solver.pop()
                    gmax = g

                g = min(gmin + .1, (gmax + gmin) / 2)

        else:
            raise NotImplementedError('no model found')

    # finally we need to convert our values back into python floats.
    # we use limit_denominator to deal with some of the rounding
    # issues.
    theta = table.theta
    linear = {v: float(model.get_py_value(bias).limit_denominator())
              for v, bias in iteritems(theta.linear)}
    quadratic = {(u, v): float(model.get_py_value(bias).limit_denominator())
                 for (u, v), bias in iteritems(theta.quadratic)}
    ground_energy = -float(model.get_py_value(theta.offset).limit_denominator())
    classical_gap = float(model.get_py_value(table.gap).limit_denominator())

    return linear, quadratic, ground_energy, classical_gap
