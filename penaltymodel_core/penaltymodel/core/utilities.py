# Copyright 2019 D-Wave Systems Inc.
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

from collections import OrderedDict
import dimod
import functools
import itertools
import numpy as np
import random
from scipy.optimize import linprog

from penaltymodel.core import PenaltyModel
from penaltymodel.core.constants import (DEFAULT_LINEAR_RANGE,
                                         DEFAULT_QUADRATIC_RANGE)


def random_indices_generator(bin_sizes, n_tries):
    """Generate random indices such that there is one index picked from each bin"""
    for _ in range(n_tries):
        random_tuple = (random.randint(0, bin_size) for bin_size in bin_sizes)
        yield random_tuple


def get_ordered_state_matrix(linear_labels, quadratic_labels):
    """Returns a state matrix following the order of [linear_labels, quadratic_labels]"""
    if not isinstance(linear_labels, list):
        raise TypeError("Linear labels must be contained in a list")
    if not isinstance(quadratic_labels, list):
        raise TypeError("Linear labels must be contained in a list")

    m_linear = len(linear_labels)
    m_quadratic = len(quadratic_labels)

    # Construct the states matrix
    # Construct linear portion of states matrix
    # Note: +2 columns in 'states' is for the offset and gap columns
    states = np.empty((2 ** m_linear, m_linear + m_quadratic + 2), dtype=int)
    states[:, :m_linear] = np.array([list(x) for x in
                                     itertools.product({-1, 1}, repeat=m_linear)])
    states[:, -2] = 1  # column for offset
    states[:, -1] = -1  # column for gap

    # Construct quadratic portion of states matrix
    labels = linear_labels + quadratic_labels
    indices = {k: i for i, k in enumerate(labels)}  # map labels to column index
    for a, b in quadratic_labels:
        a_ind = indices[a]
        b_ind = indices[b]
        ab_ind = indices[(a, b)]
        states[:, ab_ind] = states[:, a_ind] * states[:, b_ind]

    ordered_states = states
    return ordered_states, labels


def get_bias_vector(linear_bias_dict, quadratic_bias_dict, bias_order, offset=0):
    if len(bias_order) != len(linear_bias_dict) + len(quadratic_bias_dict):
        raise ValueError("The number of elements in the bias ordering does not"
                         "equal the number of elements in the biases")

    biases = []
    for k in bias_order:
        # Quadratic bias
        if isinstance(k, tuple) and len(k) == 2 and k[0] != k[1]:
            biases.append(quadratic_bias_dict[k])
            continue

        # Linear bias
        # Note: the linear bias key could either be a single element or a
        #  two-element tuple with identical elements
        biases.append(linear_bias_dict[k])

    biases.append(offset)

    return np.array(biases)


# TODO: this function might be overly specific
def get_bounds(linear_ranges, quadratic_ranges, order, min_classical_gap,
               default_linear_range=DEFAULT_LINEAR_RANGE,
               default_quadratic_range=DEFAULT_QUADRATIC_RANGE):
    # Note: Since ising has {-1, 1}, the largest possible gap is [-largest_bias,
    #   largest_bias], hence that 2 * sum(largest_biases)
    lr = linear_ranges
    qr = quadratic_ranges
    default_lr = default_linear_range
    default_qr = default_quadratic_range

    bounds = []
    for k in order:
        if isinstance(k, tuple) and len(k) == 2 and k[0] != k[1]:
            quadratic_bias_bound = qr.get(k[0], default_qr).get(k[1], default_qr)
            bounds.append(quadratic_bias_bound)
            continue

        linear_bias_bound = lr.get(k, default_lr)
        bounds.append(linear_bias_bound)

    max_gap = 2 * sum(max(abs(lbound), abs(ubound)) for lbound, ubound in bounds)
    bounds.append((None, None))  # Bound for offset
    bounds.append((min_classical_gap, max_gap))  # Bound for gap.

    return bounds


def get_binned_indices(arr, step):
    bins = []
    bin_sizes = []
    curr_bin = []

    for i, v in enumerate(np.nditer(arr, order="K")):
        if v > 0:
            curr_bin.append(i)

        if i % step == step-1 and curr_bin:
            bins.append(curr_bin)
            bin_sizes.append(len(curr_bin))
            curr_bin = []

    return bins, bin_sizes


def get_uniform_penaltymodel(pmodel, n_tries=100, tol=1e-12):
    """Returns a uniform penaltymodel

    Note: if pmodel is already uniform, it will simply return the pmodel.
    Otherwise, a pmodel with unique ground states is returned.

    pmodel(PenaltyModel): a penaltymodel
    n_tries(int): number of attempts at making a uniform penaltymodel
    tol(float): gap tolerance between uniform penaltymodel gap and
      pmodel.min_classical_gap
    """
    # TODO: could probably put the matrix construction in its own function
    if not pmodel.model:
        raise ValueError("There is no model to balance")

    # Convert BQM to spin and define a function to undo this conversion later on
    if pmodel.vartype == "dimod.SPIN":
        def convert_to_original_vartype(spin_bqm):
            return spin_bqm
        bqm = pmodel.model
    else:
        def convert_to_original_vartype(spin_bqm):
            return spin_bqm.change_vartype(dimod.BINARY)
        bqm = pmodel.model.change_vartype(dimod.SPIN)

    # Set up
    # Note: Linear labels are ordered such that decision variables start first;
    #   that way, when the state matrix is returned, the initial columns of the
    #   state matrix will correspond to the decision variables. This will make
    #   grouping states with the same feasible configurations much easier later on.
    aux_variables = [k for k in bqm.linear.keys() if k not in pmodel.decision_variables]
    linear_labels = list(pmodel.decision_variables) + aux_variables
    quadratic_labels = list(bqm.quadratic.keys())
    states, labels = get_ordered_state_matrix(linear_labels, quadratic_labels)

    # Construct biases and energy vectors
    biases = get_bias_vector(bqm.linear, bqm.quadratic, labels, bqm.offset)
    energy = np.matmul(states[:, :-1], biases)  # Ignore last column; gap column

    # Group states by threshold
    feasible_flag = energy <= pmodel.ground_energy
    feasible_states = states[feasible_flag]

    # Bin auxiliary configs with the same decision configs together
    step = 2**len(aux_variables)
    bins, bin_sizes = get_binned_indices(feasible_flag, step)

    if len(feasible_states) == 0:
        raise RuntimeError("no states with energies less than or equal to the"
                           " ground_energy found")

    # Check for balance
    if len(feasible_states) == len(pmodel.feasible_configurations):
        return pmodel

    # Cost function
    cost_weights = np.zeros((1, states.shape[1]))
    cost_weights[0, -1] = -1  # Only interested in maximizing the gap

    bounds = get_bounds(pmodel.ising_linear_ranges, pmodel.ising_quadratic_ranges,
                        labels, pmodel.min_classical_gap)

    # Attempt to find solution
    gap_threshold = max(pmodel.min_classical_gap - tol, 0)
    n_possibilities = functools.reduce(lambda a, b: a*b, bin_sizes)
    if n_possibilities <= n_tries:
        index_gen = itertools.product(*(range(x) for x in bin_sizes))
    else:
        index_gen = random_indices_generator(bins, n_tries)

    for row_indices in index_gen:
        index = [bin[i] for i, bin in zip(row_indices, bins)]

        is_unique = np.zeros(len(states))
        is_unique[index] = 1

        states[is_unique==1, -1] = 0
        states[is_unique==0, -1] = -1
        unique_feasible_states = states[is_unique==1, :]
        new_excited_states = -states[is_unique==0, :]

        # Returns a Scipy OptimizeResult
        #new_excited_states = -np.vstack((excited_states, duplicate_feasible_states))
        result = linprog(cost_weights.flatten(),
                         A_eq=unique_feasible_states,
                         b_eq=np.zeros((unique_feasible_states.shape[0], 1)),
                         A_ub=new_excited_states,
                         b_ub=np.zeros((new_excited_states.shape[0], 1)),
                         bounds=bounds,
                         method="revised simplex",
                         options={"tol": tol})

        # Break when we get desirable result
        result_gap = result.x[-1]
        if result.success and result_gap >= gap_threshold:
            break
    else:
        raise ValueError('Unable to balance this penaltymodel')

    # Parse result
    weights = result.x
    h = weights[:len(linear_labels)]
    j = weights[len(linear_labels):-2]
    offset = weights[-2]
    gap = weights[-1]

    # Create BQM
    new_bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    new_bqm.add_variables_from((v, bias) for v, bias in zip(linear_labels, h))
    new_bqm.add_interactions_from((u, v, bias) for (u, v), bias in
                                  zip(quadratic_labels, j))
    new_bqm.add_offset(offset)
    new_bqm = convert_to_original_vartype(new_bqm)

    # Copy and update
    new_pmodel = PenaltyModel(graph=pmodel.graph,
                              decision_variables=pmodel.decision_variables,
                              feasible_configurations=pmodel.feasible_configurations,
                              vartype=pmodel.vartype,
                              model=new_bqm,
                              classical_gap=gap,
                              ground_energy=pmodel.ground_energy,
                              ising_linear_ranges=pmodel.ising_linear_ranges,
                              ising_quadratic_ranges=pmodel.ising_quadratic_ranges)

    return new_pmodel
