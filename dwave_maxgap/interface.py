from penaltymodel import PenaltyModel, Ising
from penaltymodel.decorators import entry_point

from dwave_maxgap.generation import generate_ising

__all__ = ['get_penalty_model']


@entry_point(-100)
def get_penalty_model(graph, decision_variables, constraint, **kwargs):
    """dummy for testing"""

    h, J, offset, gap = generate_ising(graph, constraint, decision_variables)

    model = Ising(h, J, offset)

    return PenaltyModel(graph, decision_variables, constraint, model, gap)
