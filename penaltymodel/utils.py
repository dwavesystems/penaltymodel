# Copyright 2022 D-Wave Systems Inc.
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

from typing import Mapping, Optional, Sequence, Tuple

import dimod
import networkx as nx
import numpy as np

from dimod.typing import Variable, VartypeLike

from penaltymodel.typing import GraphLike

__all__ = ['as_graph']


def as_graph(graph_like: GraphLike) -> nx.Graph:
    """Create a NetworkX graph from a graph-like.

    Args:
        graph_like:
            A NetworkX :class:`~networkx.Graph`, a list of nodes or a number.
            If it is a list of nodes, then a complete graph with those nodes is
            returned.
            If it is a number, then the list of nodes is treated as
            ``range(n)``.

    Returns:
        A NetworkX graph. If ``graph_like`` is a NetworkX graph then it is
        returned.

    """
    return graph_like if isinstance(graph_like, nx.Graph) else nx.complete_graph(graph_like)


def table_to_sampleset(table: Mapping[Tuple[int, ...], float],
                       decision: Sequence[Variable],
                       vartype: Optional[VartypeLike] = None) -> dimod.SampleSet:
    """Convert a table of feasible configurations into a sample set."""
    if table:
        samples, energies = zip(*table.items())
    else:
        samples = np.empty((0, 0))
        energies = np.empty(0)

    samples = np.asarray(samples, dtype=np.int8)

    if vartype is None:
        vartype = dimod.BINARY if (samples == 0).any() else dimod.SPIN

    return dimod.SampleSet.from_samples((samples, decision), vartype=vartype, energy=energies)
