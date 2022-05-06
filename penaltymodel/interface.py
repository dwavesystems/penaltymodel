# Copyright 2021 D-Wave Systems Inc.
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

r"""This package implements the generation and caching of :term:`penalty model`\ s."""

import copy

from typing import Mapping, Optional, Sequence, Tuple

import dimod
import networkx as nx

from dimod.typing import Variable

from penaltymodel.database import PenaltyModelCache
from penaltymodel.exceptions import MissingPenaltyModel
from penaltymodel.generation import generate
from penaltymodel.typing import GraphLike

__all__ = ['get_penalty_model']


def get_penalty_model(samples_like,
                      graph_like: Optional[GraphLike] = None,
                      *,
                      linear_bound: Tuple[float, float] = (-2, 2),
                      quadratic_bound: Tuple[float, float] = (-1, 1),
                      min_classical_gap: float = 2,
                      use_cache: bool = True,
                      ) -> Tuple[dimod.BinaryQuadraticModel, float]:
    """Get a penalty model for a specific graph and set of target states.

    Args:
        samples_like:
            The set of feasible states that form the ground states of the
            generated binary quadratic model.

            ``samples_like`` is an extension of NumPy's array_like_.
            See :func:`dimod.as_samples`.

            .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

        graph_like:
            Defines the structure of the desired binary quadratic model. Each
            node in the graph represents a variable and each edge defines an
            interaction between two variables.
            Can be given as a :class:`networkx.Graph`, a :class:`int`, or as
            a sequence of variable labels.

            If given as a sequence of labels, the structure will be
            fully-connected, with the variables labelled according to the
            sequence.

            If given as an int, the structure will be
            fully-connected with the variables labelled ``range(n)``.

            The nodes of the graph must be a superset of the labels of
            ``samples_like``.

            If not provided, defaults to a fully connected graph with nodes
            that are the variables of ``samples_like``.

        linear_bound:
            The range allowed for the linear biases of the binary quadratic
            model.

        quadratic_bound:
            The range allowed for the quadratic biases of the binary quadratic
            model.

        min_classical_gap:
            This is a threshold value for the classical gap. It describes the
            minimum energy gap between the highest feasible state and the
            lowest infeasible state.

        use_cache:
            Whether to attempt to retrieve models from the cache. If ``False``,
            a new model will always be generated.

    Returns:
        A 2-tuple of the binary quadratic model and the classical gap. Note
        that the binary quadratic model always has vartype ``'SPIN'``.

    Raises:
        ImpossiblePenaltyModel:
            If it is not possible to construct a penalty model for the given
            structure and feasible states.

    Examples:

        >>> import dimod
        >>> import penaltymodel

        This example generates a penalty model for an AND gate.

        >>> bqm, gap = penaltymodel.get_penalty_model([[0, 0, 0],
        ...                                            [0, 1, 0],
        ...                                            [1, 0, 0],
        ...                                            [1, 1, 1]])

        We can then solve the generated binary quadratic model using a
        brute-force solver to confirm that we have the expected ground states.

        >>> print(dimod.ExactSolver().sample(bqm))
           0  1  2 energy num_oc.
        0 -1 -1 -1    0.0       1
        1 +1 -1 -1    0.0       1
        3 -1 +1 -1    0.0       1
        5 +1 +1 +1    0.0       1
        2 +1 +1 -1    2.0       1
        4 -1 +1 +1    2.0       1
        6 +1 -1 +1    2.0       1
        7 -1 -1 +1    6.0       1
        ['SPIN', 8 rows, 8 samples, 3 variables]

    """

    # by default, just make a compelte graph from the samples
    if graph_like is None:
        samples, labels = dimod.as_samples(samples_like)
        graph_like = nx.complete_graph(labels)

    if use_cache:
        with PenaltyModelCache() as cache:
            try:
                return cache.retrieve(samples_like=samples_like,
                                      graph_like=graph_like,
                                      linear_bound=linear_bound,
                                      quadratic_bound=quadratic_bound,
                                      min_classical_gap=min_classical_gap,
                                      )
            except MissingPenaltyModel:
                pass  # generate

    bqm, gap, _ = generate(graph_like=graph_like,
                           samples_like=samples_like,
                           linear_bound=linear_bound,
                           quadratic_bound=quadratic_bound,
                           min_classical_gap=min_classical_gap,
                           )

    if use_cache:
        with PenaltyModelCache() as cache:
            cache.insert_penalty_model(bqm, samples_like, gap)

    return bqm, gap
