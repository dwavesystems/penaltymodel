"""
.. [DO] Bian et al., "Discrete optimization using quantum annealing on sparse Ising models",
        https://www.frontiersin.org/articles/10.3389/fphy.2014.00056/full

.. [MC] Z. Bian, F. Chudak, R. Israel, B. Lackey, W. G. Macready, and A. Roy
        "Mapping constrained optimization problems to quantum annealing with application to fault diagnosis"
        https://arxiv.org/pdf/1603.03111.pdf
"""

import itertools
from collections import defaultdict

from six import iteritems
import penaltymodel.core as pm
from pysmt.shortcuts import Solver

from penaltymodel.maxgap.smt import Table

__all__ = ['generate_ising']


def generate_ising(graph, feasible_configurations, decision_variables,
                   linear_energy_ranges, quadratic_energy_ranges,
                   smt_solver_name):
    """Generates the Ising model that induces the given feasible configurations.

    Args:
        graph (nx.Graph): The target graph on which the Ising model is to be built.
        feasible_configurations (dict): The set of feasible configurations
            of the decision variables. The key is a feasible configuration
            as a tuple of spins, the values are the associated energy.
        decision_variables (list/tuple): Which variables in the graph are
            assigned as decision variables.
        linear_energy_ranges (dict, optional): A dict of the form
            {v: (min, max, ...} where min and max are the range
            of values allowed to v.
        quadratic_energy_ranges (dict): A dict of the form
            {(u, v): (min, max), ...} where min and max are the range
            of values allowed to (u, v).
        smt_solver_name (str/None): The name of the smt solver. Must
            be a solver available to pysmt. If None, uses the pysmt default.

    Returns:
        tuple: A 4-tuple contiaing:

            dict: The linear biases of the Ising problem.

            dict: The quadratic biases of the Ising problem.

            float: The ground energy of the Ising problem.

            float: The classical energy gap between ground and the first
            excited state.

    Raises:
        ImpossiblePenaltyModel: If the penalty model cannot be built. Normally due
            to a non-zero infeasible gap.

    """
    # we need to build a Table. The table encodes all of the information used by the smt solver
    table = Table(graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges)

    # iterate over every possible configuration of the decision variables.
    for config in itertools.product((-1, 1), repeat=len(decision_variables)):

        # determine the spin associated with each varaible in decision variables.
        spins = dict(zip(decision_variables, config))

        if config in feasible_configurations:
            # if the configuration is feasible, we require that the mininum energy over all
            # possible aux variable settings be exactly its target energy (given by the value)
            table.set_energy(spins, feasible_configurations[config])
        else:
            # if the configuration is infeasible, we simply want its minimum energy over all
            # possible aux variable settings to be an upper bound on the classical gap.
            table.set_energy_upperbound(spins)

    # now we just need to get a solver
    with Solver(smt_solver_name) as solver:

        # add all of the assertions from the table to the solver
        for assertion in table.assertions:
            solver.add_assertion(assertion)

        # check if the model is feasible at all.
        if solver.solve():

            # we want to increase the gap until we have found the max classical gap
            gmin = 0
            gmax = sum(max(abs(r) for r in linear_energy_ranges[v]) for v in graph)
            gmax += sum(max(abs(r) for r in quadratic_energy_ranges[(u, v)])
                        for (u, v) in graph.edges)

            # 2 is a good target gap
            g = 2.

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
            raise pm.ImpossiblePenaltyModel("Model cannot be built")

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
