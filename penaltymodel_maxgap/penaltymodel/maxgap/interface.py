from six import iteritems
import penaltymodel.core as pm
import dimod

from penaltymodel.maxgap.generation import generate_ising

__all__ = ['get_penalty_model']


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
                                   for config, en in iteritems(feasible_configurations)}

    # convert ising_quadratic_ranges to the form we expect
    ising_quadratic_ranges = specification.ising_quadratic_ranges
    quadratic_ranges = {(u, v): ising_quadratic_ranges[u][v] for u, v in specification.graph.edges}

    linear, quadratic, ground, gap = generate_ising(specification.graph,
                                                    feasible_configurations,
                                                    specification.decision_variables,
                                                    specification.ising_linear_ranges,
                                                    quadratic_ranges,
                                                    None)  # unspecified smt solver

    # set of the model with 0.0 offset
    model = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.SPIN)

    return pm.PenaltyModel.from_specification(specification, model, gap, ground)
