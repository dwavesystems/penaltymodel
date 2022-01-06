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

import copy

from typing import Mapping, Sequence, Tuple

import dimod
import networkx as nx

from dimod.typing import Variable

from penaltymodel.cache import PenaltyModelCache
from penaltymodel.exceptions import MissingPenaltyModel
from penaltymodel.lp import generate
from penaltymodel.typing import GraphLike, PenaltyModel


def get_penalty_model(graph_like: GraphLike,
                      samples_like,
                      *,
                      linear_bound: Tuple[float, float] = (-2, 2),
                      quadratic_bound: Tuple[float, float] = (-1, 1),
                      min_classical_gap: float = 2,
                      use_cache: bool = True,
                      ) -> PenaltyModel:

    if use_cache:
        with PenaltyModelCache() as cache:
            try:
                bqm, gap = cache.retrieve(graph_like, samples_like,
                                          linear_bound=linear_bound,
                                          quadratic_bound=quadratic_bound,
                                          min_classical_gap=min_classical_gap,
                                          )
            except MissingPenaltyModel:
                pass  # generate
            else:
                return PenaltyModel(
                    bqm=bqm, classical_gap=gap,
                    # everything should be immutable so deepcopy is overkill but might as well
                    sampleset=dimod.SampleSet.from_samples_bqm(samples_like, bqm)
                    )

    bqm, gap, _ = generate(graph_like, samples_like,
                           linear_bound=linear_bound,
                           quadratic_bound=quadratic_bound,
                           min_classical_gap=min_classical_gap,
                           )

    return PenaltyModel(
        bqm=bqm, classical_gap=gap,
        # everything should be immutable so deepcopy is overkill but might as well
        sampleset=dimod.SampleSet.from_samples_bqm(samples_like, bqm),
        )
