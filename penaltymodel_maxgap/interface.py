import penaltymodel as pm

from penaltymodel_maxgap.generation import generate_ising

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

    linear, quadratic, ground, gap = generate_ising(specification.graph,
                                                    specification.feasible_configurations,
                                                    specification.decision_variables,
                                                    specification.linear_energy_ranges,
                                                    specification.quadratic_energy_ranges,
                                                    None)  # unspecified smt solver

    # set of the model with 0.0 offset
    model = pm.BinaryQuadraticModel(linear, quadratic, 0.0, pm.SPIN)

    return pm.PenaltyModel.from_specification(specification, model, gap, ground)
