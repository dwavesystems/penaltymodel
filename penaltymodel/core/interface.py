# Copyright 2017 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""penaltymodel provides functionality for accessing PenaltyModel factories.

Accessing Factories
-------------------

Any factories that have been identified through the :const:`FACTORY_ENTRYPOINT` entrypoint
and installed on the python path can be accessed through the :func:`get_penalty_model`
function.

Examples:
    >>> import networkx as nx
    >>> import dimod
    >>> graph = nx.path_graph(5)
    >>> decision_variables = (0, 4)  # the ends of the path
    >>> feasible_configurations = {(-1, -1), (1, 1)}  # we want the ends of the path to agree
    >>> spec = pm.Specification(graph, decision_variables, feasible_configurations, dimod.SPIN)
    >>> widget = pm.get_penalty_model(spec)

Functions and Utilities
-----------------------
"""
import warnings

import dimod

from penaltymodel.database import PenaltyModelCache
from penaltymodel.core.classes import PenaltyModel, Specification
from penaltymodel.interface import get_penalty_model as _get_penalty_model
from penaltymodel.utils import table_to_sampleset

__all__ = ['FACTORY_ENTRYPOINT', 'CACHE_ENTRYPOINT', 'get_penalty_model', 'penaltymodel_factory',
           'iter_factories', 'iter_caches']

# endpoints kept for backwards compatibility, but they do nothing
FACTORY_ENTRYPOINT = 'penaltymodel_factory'
CACHE_ENTRYPOINT = 'penaltymodel_cache'


def get_penalty_model(specification):
    """Retrieve a PenaltyModel from one of the available factories.

    Args:
        specification (:class:`.Specification`): The specification
            for the desired PenaltyModel.

    Returns:
        :class:`.PenaltyModel`/None: A PenaltyModel as returned by
        the highest priority factory, or None if no factory could
        produce it.

    Raises:
        :exc:`ImpossiblePenaltyModel`: If the specification
            describes a penalty model that cannot be built by any
            factory.

    """
    warnings.warn(
        "penaltymodel.core.get_penalty_model() function is deprecated "
        "and will be removed in penaltymodel 2.0.0, "
        "use penaltymodel.get_penalty_model() instead, which has a different "
        "interface.",
        DeprecationWarning, stacklevel=3)

    # we used to specify them per variable, but now just do it globally
    linear_bound = (
        min((b for b, _ in specification.ising_linear_ranges.values()), default=-2),
        max((b for _, b in specification.ising_linear_ranges.values()), default=2)
        )

    quadratic_bound = (
        min((b for n in specification.ising_quadratic_ranges.values() for b, _ in n.values()), default=-1),
        max((b for n in specification.ising_quadratic_ranges.values() for _, b in n.values()), default=1)
        )

    bqm, gap = _get_penalty_model(
        table_to_sampleset(specification.feasible_configurations, specification.decision_variables, specification.vartype),
        specification.graph,
        linear_bound=linear_bound,
        quadratic_bound=quadratic_bound,
        min_classical_gap=specification.min_classical_gap,
        )

    return PenaltyModel.from_specification(
        specification,
        model=bqm.change_vartype(specification.vartype, inplace=True),
        classical_gap=gap,
        ground_energy=min(specification.feasible_configurations.values(), default=0),
        )


def penaltymodel_factory(priority):
    """Decorator to assign a `priority` attribute to the decorated function.

    Args:
        priority (int): The priority of the factory. Factories are queried
            in order of decreasing priority.

    Examples:
        Decorate penalty model factories like:

        >>> @pm.penaltymodel_factory(105)
        ... def factory_function(spec):
        ...     pass
        >>> factory_function.priority
        105

    """
    warnings.warn(
        "penaltymodel_factory() is deprecated and will be removed in penaltymodel 2.0.0",
        DeprecationWarning, stacklevel=2)

    # just do nothing
    def _entry_point(f):
        return f
    return _entry_point


def iter_factories():
    """Iterate through all factories identified by the factory entrypoint.

    Yields:
        function: A function that accepts a :class:`.Specification` and
        returns a :class:`.PenaltyModel`.

    """
    warnings.warn(
        "iter_factories() is deprecated and will be removed in penaltymodel 2.0.0, "
        "use penaltymodel.get_penalty_model() directly instead.",
        DeprecationWarning, stacklevel=2)

    # This used to iterate over installed factories, but now there is only one
    # so just yield it
    yield get_penalty_model


def iter_caches():
    """Iterator over the PenaltyModel caches.

    Yields:
        function: A function that accepts a :class:`PenaltyModel` and caches
        it.

    """
    warnings.warn(
        "iter_caches() is deprecated and will be removed in penaltymodel 2.0.0, "
        "use penaltymodel.PenaltyModelCache directly instead.",
        DeprecationWarning, stacklevel=2)

    cache = PenaltyModelCache()

    def cache_function(pm: PenaltyModel):
        sampleset = table_to_sampleset(pm.feasible_configurations, pm.decision_variables, pm.vartype)
        cache.insert_penalty_model(pm.model, sampleset, pm.classical_gap)

    # This used to iterate over installed caches, but now there is only one
    # so just yield it
    yield cache_function
