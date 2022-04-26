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

"""The module is considered internal."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dimod
import networkx as nx
import numpy as np
import scipy.optimize

from dimod.typing import GraphLike, Variable

from penaltymodel.exceptions import ImpossiblePenaltyModel
from penaltymodel.utils import as_graph

__all__ = []


class Index:
    """Tracks the column indices of the various values of interest"""

    def __init__(self,
                 decision: Iterable[Variable],
                 auxiliary: Iterable[Variable],
                 interactions: Iterable[Tuple[Variable, Variable]]
                 ):

        start = 2  # after gap and offset
        self._decision = dict((v, i) for i, v in enumerate(decision, start))
        start += len(self._decision)
        self._auxiliary = dict((v, i) for i, v in enumerate(auxiliary, start))
        start += len(self._auxiliary)
        self._interactions = dict(
            (frozenset(e), i) for i, e in enumerate(interactions, start))
        self._stop = start + len(self._interactions)

    def __len__(self) -> int:
        return self._stop

    def auxiliaries(self) -> Sequence[int]:
        return range(2 + len(self._decision), 2 + self.num_variables())

    def decisions(self) -> Sequence[int]:
        return range(2, 2 + len(self._decision))

    @staticmethod
    def gap() -> int:
        return 0

    def interaction(self, u: Variable, v: Variable) -> int:
        return self._interactions[frozenset((u, v))]

    def num_variables(self) -> int:
        return len(self._decision) + len((self._auxiliary))

    def make_bounds(self,
                    min_classical_gap: float,
                    linear_bound: Tuple[float, float],
                    quadratic_bound: Tuple[float, float],
                    ) -> Sequence[Tuple[Optional[float], Optional[float]]]:
        bounds = [(min_classical_gap, None), (None, None)]
        bounds.extend(linear_bound for _ in range(self.num_variables()))
        bounds.extend(quadratic_bound for _ in range(len(bounds), len(self)))
        return bounds

    @staticmethod
    def offset() -> int:
        return 1

    def variable(self, v: Variable) -> int:
        try:
            return self._decision[v]
        except KeyError:
            try:
                return self._auxiliary[v]
            except KeyError:
                raise ValueError(f"no variable {v!r}") from None

    def variables(self) -> Sequence[int]:
        return range(2, 2 + self.num_variables())


def all_possible(num_variables: int) -> np.ndarray:
    """Create an array of all possible spin configurations."""
    assert 0 <= num_variables < 1 << 8
    a = np.unpackbits(np.arange(1 << num_variables, dtype=np.uint8).reshape(1 << num_variables, 1),
                      axis=1, count=num_variables, bitorder='little').astype(np.int8)
    a *= 2
    a -= 1
    return a


def next_auxiliary(state: Tuple[int, ...]) -> Tuple[int, ...]:
    s = list(state)
    for i in range(len(s)-1, -1, -1):
        if s[i] <= 0:
            s[i] = 1
            break
        else:
            s[i] = -1
    return tuple(s)


def generate(graph_like: GraphLike,
             samples_like,
             *,
             linear_bound: Tuple[float, float] = (-2, 2),
             quadratic_bound: Tuple[float, float] = (-1, 1),
             min_classical_gap: float = 2,
             ) -> Tuple[dimod.BinaryQuadraticModel, float, Dict[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate a penalty model.

    This function is considered internal, it is recommended to use
    :func:`~penaltymodel.get_penalty_model` with ``use_cache=False`` instead.
    """
    graph = as_graph(graph_like)
    samples, decision = dimod.as_samples(samples_like)

    if any(v not in graph.nodes for v in decision):
        raise ValueError("the decision variables must be a subset of the graph nodes")

    # let's make things easier for ourselves by casting the samples into -1, +1
    if (samples == 0).any():
        samples = 2*samples - 1
    if not ((samples == +1) ^ (samples == -1)).all():
        raise ValueError("given samples should be 0/1 or -1/+1")

    auxiliaries = list(graph.nodes - decision)
    num_samples = samples.shape[0]
    num_variables = len(graph.nodes)
    num_auxiliary = num_variables - len(decision)

    if isinstance(samples_like, dimod.SampleSet):
        energies = samples_like.record.energy
    else:
        energies = np.zeros(num_samples)

    # construct the table
    if len(energies):
        table = {tuple(map(int, state)): energy for state, energy in zip(samples, energies)}
    else:
        table = {}

    # todo: more correctness checks, max variables==8

    # some edge cases we can easily eliminate
    if not table or not decision:
        bqm = dimod.BinaryQuadraticModel('SPIN')
        bqm.add_linear_from((v, 0) for v in graph.nodes)
        bqm.add_quadratic_from((u, v, 0) for u, v in graph.edges)
        return bqm, float('inf'), {}

    # create an object to track the columns in the LP matrix

    indexer = Index(decision, auxiliaries, graph.edges)

    # we'll use this to track where the ground states are. Note that we could
    # avoiding needing to store this in memory with some clever indexing, but
    # let's not worry too much about that unless it becomes a problem later
    ground: Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]] = dict()

    # ok, let's build our matrices for the LP

    b = np.full(1 << num_variables, max(table.values(), default=0), dtype=float)

    A = np.empty((1 << len(graph), len(indexer)), dtype=np.int8)
    A[:, indexer.variables()] = all_possible(num_variables)
    for u, v in graph.edges:
        A[:, indexer.interaction(u, v)] = A[:, indexer.variable(u)] * A[:, indexer.variable(v)]
    A[:, indexer.offset()] = 1

    # the gap and b are how we distinguish between values in the table and
    # not
    for i in range(1 << num_variables):
        decision_state = tuple(A[i, indexer.decisions()])

        if decision_state in table:
            A[i, indexer.gap()] = 0
            b[i] = table[decision_state]
            ground.setdefault(decision_state, {})[tuple(A[i, indexer.auxiliaries()])] = i
        else:
            A[i, indexer.gap()] = -1

    # for now we're just trying to find feasibility, we'll optimize at the end
    c = np.zeros(len(indexer))

    # bounds are fixed
    bounds = indexer.make_bounds(min_classical_gap, linear_bound, quadratic_bound)

    # ok, we have everything in hand to start solving!
    upper_bound = list(range(A.shape[0]))
    equality: List[int] = []
    auxiliary_configurations: Dict[Tuple[int, ...], Tuple[int, ...]] = OrderedDict()

    # WLOG, we can fix one right away
    decision_state = next(state for state in ground if state not in auxiliary_configurations)
    auxiliary_configurations[decision_state] = auxiliary_state = (-1,)*num_auxiliary
    i = ground[decision_state][auxiliary_state]
    upper_bound.remove(i)
    equality.append(i)

    while True:
        A_eq = A[equality, :]
        b_eq = b[equality]
        A_ub = -A[upper_bound, :]  # negate because we want A_ub <= b_ub
        b_ub = -b[upper_bound]

        res = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='highs')

        if res.success:
            if len(auxiliary_configurations) == len(table):
                break

            # fix a new state
            decision_state = next(state for state in ground if state not in auxiliary_configurations)
            auxiliary_configurations[decision_state] = auxiliary_state = (-1,)*num_auxiliary
            i = ground[decision_state][auxiliary_state]
            upper_bound.remove(i)
            equality.append(i)
        else:
            # ok, we didn't succeed. So first try changing the aux state of the
            # last set
            try:
                decision_state, auxiliary_state = auxiliary_configurations.popitem()
                upper_bound.append(equality.pop())  # put it back into inequality
                while all(s == 1 for s in auxiliary_state):
                    decision_state, auxiliary_state = auxiliary_configurations.popitem()
                    upper_bound.append(equality.pop())  # put it back into inequality
            except KeyError:
                raise ImpossiblePenaltyModel("There is no BQM that can encode the given constraint") from None

            # iterate the auxiliary state
            auxiliary_configurations[decision_state] = auxiliary_state = next_auxiliary(auxiliary_state)
            i = ground[decision_state][auxiliary_state]
            upper_bound.remove(i)
            equality.append(i)

    assert res.success

    # having found something feasible, let's do one last run, this time optimizing the gap
    c[indexer.gap()] = -1
    res_opt = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='highs')
    if res_opt.success:
        res = res_opt
        gap = res.x[indexer.gap()]
    elif res_opt.status == 3:
        # error code 3 is unbounded objective, which can happen for a fully
        # specified problem
        gap = float('inf')
    else:
        raise RuntimeError("something went wrong")

    # let's make the BQM!
    bqm = dimod.BinaryQuadraticModel('SPIN')
    bqm.add_linear_from((v, res.x[indexer.variable(v)]) for v in graph.nodes)
    bqm.add_quadratic_from((u, v, res.x[indexer.interaction(u, v)]) for u, v in graph.edges)
    bqm.offset = res.x[indexer.offset()]

    # return which auxiliary variables are which
    aux = dict((state, dict(zip(auxiliaries, aux))) for state, aux in auxiliary_configurations.items())

    return bqm, gap, aux
