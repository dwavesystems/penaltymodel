"""This module has the primary public-facing methods for the project.
"""
import penaltymodel as pm

from penaltymodel_cache.database_manager import cache_connect, insert_penalty_model, \
    iter_penalty_model_from_specification


__all__ = ['get_penalty_model',
           'cache_penalty_model']


@pm.interface.penaltymodel_factory(100)
def get_penalty_model(specification, database=None):
    """Factory function for penaltymodel_cache.

    Args:
        specification (penaltymodel.Specification): The specification
            for the desired penalty model.
        database (str, optional): The path to the desired sqlite database
            file. If None, will use the default.

    Returns:
        :class:`penaltymodel.PenaltyModel`: Penalty model with the given specification.

    Raises:
        :class:`penaltymodel.MissingPenaltyModel`: If the penalty model is not in the
            cache.

    Parameters:
        priority (int): 100

    """
    # only handles index-labelled nodes
    if not all(isinstance(v, int) for v in specification.graph):
        raise ValueError('graph variables must be index-labelled')

    # connect to the database. Note that once the connection is made it cannot be
    # broken up between several processes.
    if database is None:
        conn = cache_connect
    else:
        conn = cache_connect(database)

    # get the penalty_model
    with conn as cur:
        widget = next(iter_penalty_model_from_specification(cur, specification))

    # close the connection
    conn.close()

    if widget is None:
        raise penaltymodel.MissingPenaltyModel("no penalty model with the given specification found in cache")

    return widget


def cache_penalty_model(penalty_model, database=None):
    """Caching function for penaltymodel_cache.

    Args:
        penalty_model (:class:`penaltymodel.PenaltyModel`): Penalty model to
            be cached.
        database (str, optional): The path to the desired sqlite database
            file. If None, will use the default.

    """

    # only handles index-labelled nodes
    if not all(isinstance(v, int) for v in penalty_model.graph):
        raise ValueError('graph variables must be index-labelled')

    # connect to the database. Note that once the connection is made it cannot be
    # broken up between several processes.
    if database is None:
        conn = cache_connect
    else:
        conn = cache_connect(database)

    # load into the database
    with conn as cur:
        insert_penalty_model(cur, penalty_model)

    # close the connection
    conn.close()
