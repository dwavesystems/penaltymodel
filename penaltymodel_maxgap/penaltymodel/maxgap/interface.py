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
import penaltymodel.core as pm
import dimod

from penaltymodel.maxgap.generation import generate

__all__ = 'get_penalty_model',


@pm.penaltymodel_factory(-100)  # set the priority to low
def get_penalty_model(specification):
    """Factory function for penaltymodel_maxgap.

    Args:
        specification (penaltymodel.Specification): The specification
            for the desired penalty model.

    Returns:
        :class:`penaltymodel.PenaltyModel`: Penalty model with the given specification.

    Raises:
        :class:`penaltymodel.ImpossiblePenaltyModel`: If the penalty cannot be built.

    Parameters:
        priority (int): -100

    """

    # check that the feasible_configurations are spin
    feasible_configurations = specification.feasible_configurations
    if specification.vartype is dimod.BINARY:
        feasible_configurations = {tuple(2 * v - 1 for v in config): en
                                   for config, en in feasible_configurations.items()}

    # convert ising_quadratic_ranges to the form we expect
    ising_quadratic_ranges = specification.ising_quadratic_ranges
    quadratic_ranges = {(u, v): ising_quadratic_ranges[u][v] for u, v in specification.graph.edges}

    bqm, gap = generate(specification.graph,
                        feasible_configurations,
                        specification.decision_variables,
                        specification.ising_linear_ranges,
                        quadratic_ranges,
                        specification.min_classical_gap,
                        None)  # unspecified smt solver

    try:
        ground = max(feasible_configurations.values())
    except ValueError:
        ground = 0.0  # if empty

    return pm.PenaltyModel.from_specification(specification, bqm, gap, ground)
