from penaltymodel import PenaltyModel, BinaryQuadraticModel
# from penaltymodel.plugins import penaltymodel_factory

from penaltymodel_maxgap.generation import generate_ising

__all__ = ['get_penalty_model']


# @penaltymodel_factory(-100)  # set the priority to low
def get_penalty_model(specification):
    """TODO"""

    linear, quadratic, ground, gap = generate_ising(specification.graph,
                                                    specification.feasible_configurations,
                                                    specification.decision_variables,
                                                    specification.linear_energy_ranges,
                                                    specification.quadratic_energy_ranges,
                                                    None)  # unspecified smt solver

    # set of the model with 0.0 offset
    model = BinaryQuadraticModel(linear, quadratic, 0.0, BinaryQuadraticModel.SPIN)

    return PenaltyModel(specification, model, gap, ground)
