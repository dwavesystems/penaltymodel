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
import itertools

import dimod

from pysmt.shortcuts import Solver

from penaltymodel.core import ImpossiblePenaltyModel
from penaltymodel.maxgap.smt import Table

__all__ = 'generate',

MAX_GAP_DELTA = 0.01


def generate(graph, feasible_configurations, decision_variables,
             linear_energy_ranges, quadratic_energy_ranges, min_classical_gap,
             smt_solver_name=None):
    """Generates the Ising model that induces the given feasible configurations. The code is based
    on the papers [#do]_ and [#mc]_.

    Args:
        graph (nx.Graph): The target graph on which the Ising model is to be built.
        feasible_configurations (dict): The set of feasible configurations
            of the decision variables. The key is a feasible configuration
            as a tuple of spins, the values are the associated energy.
        decision_variables (list/tuple): Which variables in the graph are
            assigned as decision variables.
        linear_energy_ranges (dict, optional): A dict of the form
            {v: (min, max), ...} where min and max are the range
            of values allowed to v.
        quadratic_energy_ranges (dict): A dict of the form
            {(u, v): (min, max), ...} where min and max are the range
            of values allowed to (u, v).
        min_classical_gap (float): The minimum energy gap between the highest feasible state and the
            lowest infeasible state.
        smt_solver_name (str/None): The name of the smt solver. Must
            be a solver available to pysmt. If None, uses the pysmt default.

    Returns:
        tuple: A 4-tuple containing:

            dict: The linear biases of the Ising problem.

            dict: The quadratic biases of the Ising problem.

            :obj:`dimod.BinaryQuadraticModel`

            float: The classical energy gap between ground and the first
            excited state.

    Raises:
        ImpossiblePenaltyModel: If the penalty model cannot be built. Normally due
            to a non-zero infeasible gap.

    .. [#do] Bian et al., "Discrete optimization using quantum annealing on sparse Ising models",
        https://www.frontiersin.org/articles/10.3389/fphy.2014.00056/full

    .. [#mc] Z. Bian, F. Chudak, R. Israel, B. Lackey, W. G. Macready, and A. Roy
        "Mapping constrained optimization problems to quantum annealing with application to fault diagnosis"
        https://arxiv.org/pdf/1603.03111.pdf

    """
    if len(graph) == 0:
        return dimod.BinaryQuadraticModel.empty(dimod.SPIN), float('inf')

    # we need to build a Table. The table encodes all of the information used by the smt solver
    table = Table(graph, decision_variables, linear_energy_ranges, quadratic_energy_ranges)

    # iterate over every possible configuration of the decision variables.
    for config in itertools.product((-1, 1), repeat=len(decision_variables)):

        # determine the spin associated with each variable in decision variables.
        spins = dict(zip(decision_variables, config))

        if config in feasible_configurations:
            # if the configuration is feasible, we require that the minimum energy over all
            # possible aux variable settings be exactly its target energy (given by the value)
            table.set_energy(spins, feasible_configurations[config])
        else:
            # if the configuration is infeasible, we simply want its minimum energy over all
            # possible aux variable settings to be an upper bound on the classical gap.
            if isinstance(feasible_configurations, dict) and feasible_configurations:
                highest_feasible_energy = max(feasible_configurations.values())
            else:
                highest_feasible_energy = 0

            table.set_energy_upperbound(spins, highest_feasible_energy)

    # now we just need to get a solver
    with Solver(smt_solver_name) as solver:

        # add all of the assertions from the table to the solver
        for assertion in table.assertions:
            solver.add_assertion(assertion)

        # add min classical gap assertion
        gap_assertion = table.gap_bound_assertion(min_classical_gap)
        solver.add_assertion(gap_assertion)

        # check if the model is feasible at all.
        if solver.solve():
            # since we know the current model is feasible, grab the initial model.
            model = solver.get_model()

            # we want to increase the gap until we have found the max classical gap
            # note: gmax is the maximum possible gap for a particular set of variables. To find it,
            #   we take the sum of the largest coefficients possible and double it. We double it
            #   because in Ising, the largest gap possible from the largest coefficient is the
            #   negative of said coefficient. Example: consider a graph with one node A, with a
            #   energy range of [-2, 1]. The largest energy gap between spins +1 and -1 is 4;
            #   namely, the largest absolute coefficient -2 with the ising spins results to
            #   gap = (-2)(-1) - (-2)(1) = 4.
            gmin = min_classical_gap
            gmax = sum(max(abs(r) for r in linear_energy_ranges[v]) for v in graph)
            gmax += sum(max(abs(r) for r in quadratic_energy_ranges[(u, v)])
                        for (u, v) in graph.edges)
            gmax *= 2

            # 2 is a good target gap
            g = max(2., gmin)

            while abs(gmax - gmin) >= MAX_GAP_DELTA:
                solver.push()

                gap_assertion = table.gap_bound_assertion(g)
                solver.add_assertion(gap_assertion)

                if solver.solve():
                    model = solver.get_model()
                    gmin = float(model.get_py_value(table.gap))

                else:
                    solver.pop()
                    gmax = g

                g = min(gmin + .1, (gmax + gmin) / 2)

        else:
            raise ImpossiblePenaltyModel("Model cannot be built")

    # finally we need to convert our values back into python floats.

    classical_gap = float(model.get_py_value(table.gap))

    # if the problem is fully specified (or empty) it has infinite gap
    if (len(decision_variables) == len(graph) and
            decision_variables and  # at least one variable
            len(feasible_configurations) == 2**len(decision_variables)):
        classical_gap = float('inf')

    return table.theta.to_bqm(model), classical_gap

