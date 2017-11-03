from penaltymodel import PenaltyModel
from penaltymodel.plugins import entry_point
from penaltymodel.exceptions import MissingPenaltyModel

from penaltymodel_cache.database_manager import load_penalty_model, connection

__all__ = ['get_penalty_model_from_specification',
           'cache_penalty_model']


@entry_point(100)
def get_penalty_model_from_specification(specification, database=None):
    """TODO"""

    # this cache is only interested in the graph, the feasible_configurations
    # and the decision_variables
    graph = specification.graph
    decision_variables = specification.decision_variables
    feasible_configurations = specification.feasible_configurations

    penalty_models = query_penalty_model(conn, graph, decision_variables, feasible_configurations)

    for penalty_model in penalty_models:
        raise NotImplementedError

    raise MissingPenaltyModel('no penalty model found in penaltymodel_cache')


def cache_penalty_model(penalty_model):
    """TODO"""

    conn = connection()

    load_penalty_model(conn, graph, decision_variables, feasible_configurations,
                       linear_biases, quadratic_biases, offset, classical_gap)


    conn.close()

    raise NotImplementedError
