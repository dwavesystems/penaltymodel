from penaltymodel import PenaltyModel, Ising
from penaltymodel.decorators import entry_point

from dwave_maxgap.generation import generate_ising

__all__ = ['get_penalty_model']


@entry_point(100)
def get_penalty_model(graph, decision_variables, constraint, **kwargs):
    """dummy for testing"""

    model = Ising({}, {})

    return PenaltyModel(graph, decision_variables, constraint, model, gap)


def cache_penalty_model(penalty_model):
    pass
