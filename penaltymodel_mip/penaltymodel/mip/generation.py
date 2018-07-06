import itertools
import random

from collections import Mapping, defaultdict

import dimod
import networkx as nx

from ortools.linear_solver import pywraplp

import penaltymodel.core as pm


def generate_bqm(graph, table, decision,
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 precision=7, max_decision=4, max_variables=8,
                 return_auxiliary=False):
    """Get a binary quadratic model with specific ground states.

    Args:
        graph (:obj:`~networkx.Graph`):
            Defines the structure of the generated binary quadratic model.

        table (iterable):
            Iterable of valid configurations (of spin-values). Each configuration is a tuple of
            variable assignments ordered by `decision`.

        decision (list/tuple):
            The variables in the binary quadratic model which have specified configurations.

        linear_energy_ranges (dict, optional):
            Dict of the form {v: (min, max, ...} where min and max are the range of values allowed
            to v. The default range is [-2, 2].

        quadratic_energy_ranges (dict, optional):
            Dict of the form {(u, v): (min, max), ...} where min and max are the range of values
            allowed to (u, v). The default range is [-1, 1].

        precision (int, optional, default=7):
            Values returned by the optimization solver are rounded to `precision` digits of
            precision.

        max_decision (int, optional, default=4):
            Maximum number of decision variables allowed. The algorithm is valid for arbitrary
            sizes of problem but can be extremely slow.

        max_variables (int, optional, default=4):
            Maximum number of variables allowed. The algorithm is valid for arbitrary
            sizes of problem but can be extremely slow.

        return_auxiliary (bool, optional, False):
            If True, the auxiliary configurations are returned for each configuration in table.

    Returns:
        If return_auxiliary is False:

            :obj:`dimod.BinaryQuadraticModel`: The binary quadratic model.

            float: The classical gap.

        If return_auxiliary is True:

            :obj:`dimod.BinaryQuadraticModel`: The binary quadratic model.

            float: The classical gap.

            dict: The auxiliary configurations, keyed on the configurations in table.

    Raises:
        ImpossiblePenaltyModel: If the penalty model cannot be built. Normally due
            to a non-zero infeasible gap.

    """

    # Developer note: This function is input checking and output formatting. The logic is
    # in _generate_ising

    if not isinstance(graph, nx.Graph):
        raise TypeError("expected input graph to be a NetworkX Graph.")

    if not set().union(*table).issubset({-1, 1}):
        raise ValueError("expected table to be spin-valued")
    elif isinstance(table, Mapping) and any(table.values()):
        raise ValueError("cannot handle non-zero target energy levels")

    if not isinstance(decision, list):
        decision = list(decision)  # handle iterables
    if not all(v in graph for v in decision):
        raise ValueError("given graph does not match the variable labels in decision variables")

    num_var = len(decision)
    if any(len(config) != num_var for config in table):
        raise ValueError("number of decision variables does not match all of the entires in the table")

    if len(decision) > max_decision:
        raise ValueError(("The table is too large. Note that larger models can be built by setting "
                          "max_decision to a higher number, but generation could be extremely slow."))

    if len(graph) > max_variables:
        raise ValueError(("The graph is too large. Note that larger models can be built by setting "
                          "max_variables to a higher number, but generation could be extremely slow."))

    if linear_energy_ranges is None:
        linear_energy_ranges = defaultdict(lambda: (-2, 2))
    if quadratic_energy_ranges is None:
        quadratic_energy_ranges = defaultdict(lambda: (-1, 1))

    if return_auxiliary:
        h, J, offset, gap, aux = _generate_ising(graph, table, decision,
                                                 linear_energy_ranges, quadratic_energy_ranges,
                                                 return_auxiliary)
    else:
        h, J, offset, gap = _generate_ising(graph, table, decision,
                                            linear_energy_ranges, quadratic_energy_ranges,
                                            return_auxiliary)

    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, round(bias, precision)) for v, bias in h.items())
    bqm.add_interactions_from((u, v, round(bias, precision)) for (u, v), bias in J.items())
    bqm.add_offset(round(offset, precision))

    if return_auxiliary:
        return bqm, round(gap, precision), aux
    else:
        return bqm, round(gap, precision)


def _generate_ising(graph, table, decision, linear_energy_ranges, quadratic_energy_ranges,
                    return_auxiliary):

    if not table:
        # if there are no feasible configurations then the gap is 0 and the model is empty
        h = {v: 0.0 for v in graph.nodes}
        J = {edge: 0.0 for edge in graph.edges}
        offset = 0.0
        gap = 0.0
        if return_auxiliary:
            return h, J, offset, gap, {}
        else:
            return h, J, offset, gap

    auxiliary = [v for v in graph if v not in decision]
    variables = decision + auxiliary

    solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    h = {v: solver.NumVar(linear_energy_ranges[v][0], linear_energy_ranges[v][1], 'h_%s' % v)
         for v in graph.nodes}

    J = {}
    for u, v in graph.edges:
        if (u, v) in quadratic_energy_ranges:
            low, high = quadratic_energy_ranges[(u, v)]
        else:
            low, high = quadratic_energy_ranges[(v, u)]
        J[(u, v)] = solver.NumVar(low, high, 'J_%s,%s' % (u, v))

    offset = solver.NumVar(-solver.infinity(), solver.infinity(), 'offset')

    gap = solver.NumVar(0, solver.infinity(), 'classical_gap')

    # Let x, a be the decision, auxiliary variables respectively
    # Let E(x, a) be the energy of x and a
    # Let F be the feasible configurations of x
    # Let g be the classical gap
    # Let a*(x) be argmin_a E(x, a) - the config of aux variables that minimizes the energy with x fixed

    # We want:
    #   E(x, a) >= 0  forall x in F, forall a
    #   E(x, a) - g >= 0  forall x not in F, forall a
    for config in itertools.product((-1, 1), repeat=len(variables)):
        spins = dict(zip(variables, config))

        const = solver.Constraint(0.0, solver.infinity())

        if tuple(spins[v] for v in decision) not in table:
            # we want energy greater than gap for decision configs not in feasible
            const.SetCoefficient(gap, -1)

        # add the energy for the configuration
        for v, bias in h.items():
            const.SetCoefficient(bias, spins[v])
        for (u, v), bias in J.items():
            const.SetCoefficient(bias, spins[u] * spins[v])
        const.SetCoefficient(offset, 1)

    if not auxiliary:
        # We have no auxiliary variables. We want:
        #   E(x) <= 0 forall x in F
        for decision_config in table:
            spins = dict(zip(decision, decision_config))

            const = solver.Constraint(-solver.infinity(), 0.0)

            # add the energy for the configuration
            for v, bias in h.items():
                const.SetCoefficient(bias, spins[v])
            for (u, v), bias in J.items():
                const.SetCoefficient(bias, spins[u] * spins[v])
            const.SetCoefficient(offset, 1)

    else:
        # We have auxiliary variables. So that each feasible config has at least one ground we want:
        #   E(x, a) - 100*|| a - a*(x) || <= 0  forall x in F, forall a

        # we need a*(x) forall x in F
        a_star = {config: {v: solver.IntVar(0, 1, 'a*(%s)_%s' % (config, v)) for v in auxiliary} for config in table}

        for decision_config in table:
            for aux_config in itertools.product((-1, 1), repeat=len(variables) - len(decision)):
                spins = dict(zip(variables, decision_config+aux_config))

                ub = 0

                # the E(x, a) term
                coefficients = {bias: spins[v] for v, bias in h.items()}
                coefficients.update({bias: spins[u] * spins[v] for (u, v), bias in J.items()})
                coefficients[offset] = 1

                # # the -100*|| a - a*(x) || term
                auxiliary_coefficients = {}
                for v in auxiliary:
                    # we don't have absolute value, so we check what a is and order the subtraction accordingly
                    if spins[v] == -1:
                        # a*(x)_v - a_v
                        coefficients[a_star[decision_config][v]] = -200
                    else:
                        # a_v - a*(x)_v
                        assert spins[v] == 1  # sanity check
                        coefficients[a_star[decision_config][v]] = +200
                        ub += 200

                const = solver.Constraint(-solver.infinity(), ub)
                for var, coef in coefficients.items():
                    const.SetCoefficient(var, coef)

        # without loss of generality we can fix the auxiliary variables associated with
        # one of the feasible configurations. Do so randomly.
        for var in next(iter(a_star.values())).values():
            val = random.randint(0, 1)
            const = solver.Constraint(val, val)  # equality constrait
            const.SetCoefficient(var, 1)

    objective = solver.Objective()
    objective.SetCoefficient(gap, 1)
    objective.SetMaximization()

    result_status = solver.Solve()

    # read everything back into floats
    h = {v: bias.solution_value() for v, bias in h.items()}
    J = {(u, v): bias.solution_value() for (u, v), bias in J.items()}
    offset = offset.solution_value()
    gap = gap.solution_value()

    if not gap:
        raise pm.ImpossiblePenaltyModel("No positive gap can be found for the given model")

    if return_auxiliary:
        if auxiliary:
            aux_configs = {config: {v: val.solution_value()*2 - 1 for v, val in a_star[config].items()}
                           for config in table}
        else:
            aux_configs = {config: dict() for config in table}
        return h, J, offset, gap, aux_configs
    else:
        return h, J, offset, gap
