"""This module has the primary public-facing methods for the project.
"""
from penaltymodel.plugins import penaltymodel_factory
from penaltymodel.exceptions import MissingPenaltyModel

from penaltymodel_cache.database_manager import cache_connect, get_penalty_model_from_specification
from penaltymodel_cache.database_manager import penalty_model_id, iter_penalty_models


__all__ = ['get_penalty_model',
           'cache_penalty_model',
           'dump_penalty_models']


@penaltymodel_factory(100)
def get_penalty_model(specification, database=None):
    """Factory function for penaltymodel_cache.

    Args:
        specification (penaltymodel.Specification): The specification
            for the desired penalty model.
        database (str, optional): The path to the desired sqlite database
            file. If None, will use the default.

    Returns:
        penaltymodel.PenaltyModel: Penalty model with the given specification.

    Raises:
        penaltymodel.MissingPenaltyModel: If the penalty model is not in the
            cache.

    Parameters:
        priority (int): 100

    """
    # only handles index-labelled nodes
    if not all(isinstance(v, int) for v in specification.graph):
        raise ValueError('graph variables must be index-labelled')

    # connect to the database. Note that once the connection is made it cannot be
    # broken up between several processes.
    conn = cache_connect(database)

    # get the model
    model = get_penalty_model_from_specification(conn, specification)

    # close the connection
    conn.close()

    if model is None:
        raise MissingPenaltyModel("no penalty model with the given specification found in cache")

    return model


def cache_penalty_model(penalty_model, database=None):
    """Caching function for penaltymodel_cache.

    Args:
        penalty_model (penaltymodel.PenaltyModel): Penalty model to
            be cached.
        database (str, optional): The path to the desired sqlite database
            file. If None, will use the default.

    """

    # only handles index-labelled nodes
    if not all(isinstance(v, int) for v in penalty_model.graph):
        raise ValueError('graph variables must be index-labelled')

    # connect to the database. Note that once the connection is made it cannot be
    # broken up between several processes.
    conn = cache_connect(database)

    # load into the database
    penalty_model_id(conn, penalty_model)

    # close the connection
    conn.close()


def dump_penalty_models(database=None):
    """Return all penalty models in the database in a list.

    Args:
        database (str, optional): The path to the desired sqlite database
            file. If None, will use the default.

    Returns:
        list: All of the :class:`penaltymodel.PenaltyModel`)s in the database.

    """
    # connect to the database. Note that once the connection is made it cannot be
    # broken up between several processes.
    conn = cache_connect(database)

    # in order to close the connection, we need to dump all of the contents to a list
    penalty_models = list(iter_penalty_models(conn))

    # close the connection
    conn.close()

    return penalty_models
