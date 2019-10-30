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


def get_state_matrix(linear_labels, quadratic_labels):
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
    labels = list(linear_labels) + list(quadratic_labels)
    indices = {k: i for i, k in enumerate(labels)}  # map labels to column index
    for a, b in quadratic_labels:
        a_ind = indices[a]
        b_ind = indices[b]
        ab_ind = indices[(a, b)]
        states[:, ab_ind] = states[:, a_ind] * states[:, b_ind]

    return states, labels


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
    m_linear = len(bqm.linear)
    states, labels = get_state_matrix(bqm.linear.keys(), bqm.quadratic.keys())

    # Construct biases and energy vectors
    biases = [bqm.linear[label] for label in labels[:m_linear]]
    biases += [bqm.quadratic[label] for label in labels[m_linear:]]
    biases += [bqm.offset]
    biases = np.array(biases)
    energy = np.matmul(states[:, :-1], biases)  # Ignore last column; gap column
    # Group states by threshold
    excited_states = states[energy > pmodel.ground_energy]
    feasible_states = states[energy <= pmodel.ground_energy]

    if len(feasible_states) == 0:
        raise RuntimeError("no states with energies less than or equal to the"
                           " ground_energy found")

    # Check for balance
    if len(feasible_states) == len(pmodel.feasible_configurations):
        return pmodel

    # Cost function
    cost_weights = np.zeros((1, states.shape[1]))
    cost_weights[0, -1] = -1  # Only interested in maximizing the gap

    # Note: Since ising has {-1, 1}, the largest possible gap is [-largest_bias,
    #   largest_bias], hence that 2 * sum(largest_biases)
    l_ranges = pmodel.ising_linear_ranges
    q_ranges = pmodel.ising_quadratic_ranges
    bounds = [l_ranges.get(l, DEFAULT_LINEAR_RANGE) for l in labels[:m_linear]]
    bounds += [q_ranges.get(x, DEFAULT_QUADRATIC_RANGE).get(y, DEFAULT_QUADRATIC_RANGE)
               for x, y in labels[m_linear:]]
    max_gap = 2 * sum(max(abs(lbound), abs(ubound)) for lbound, ubound in bounds)
    bounds.append((None, None))  # Bound for offset
    bounds.append((pmodel.min_classical_gap, max_gap))  # Bound for gap.

    # Determine duplicate decision states
    # Note: we are forming a new matrix, decision_cols, which is made up of the
    #   decision variable columns. We use decision_cols to bin like-feasible
    #   states together (i.e. same decision state values, potentially different
    #   aux values).
    # Note2: using lexsort so that each row of decision_cols is treated as a
    #   single object with primary, secondary, tertiary, etc key orders
    # Note3: bins contains the index of the last item in each bin; these are the
    #   bin boundaries
    decision_indices = [i for i, label in enumerate(labels) if label in pmodel.decision_variables]
    decision_cols = feasible_states[:, decision_indices]
    sorted_indices = np.lexsort(decision_cols.T)
    decision_cols = decision_cols[sorted_indices, :]
    feasible_states = feasible_states[sorted_indices, :]
    bins = (decision_cols[:-1, :] != decision_cols[1:, :]).any(axis=1)
    bins = np.append(bins, True)  # Marking the end of the last bin
    bins = np.nonzero(bins)[0]

    # Get number of unique decision states and number of items in each bin
    n_uniques = bins.shape[0]
    bin_count = np.hstack((bins[0] + 1, bins[1:] - bins[:-1]))  # +1 for zero-index

    # pmodel is already balanced.
    if len(set(bin_count)) == 1:
        return pmodel

    # Attempt to find solution
    gap_threshold = max(pmodel.min_classical_gap - tol, 0)
    n_possibilities = functools.reduce(lambda a, b: a*b, bin_count)
    if n_possibilities <= n_tries:
        index_gen = itertools.product(*(range(x) for x in bin_count))
    else:
        index_gen = random_indices_generator(bins, n_tries)

    for random_indices in index_gen:
        random_indices = np.array(random_indices)
        random_indices[1:] += (bins[:-1] + 1)  # add bin offsets; +1 to negate bins' zero-index
        is_unique = np.zeros(feasible_states.shape[0], dtype=int)
        is_unique[random_indices] = 1

        # Select which feasible states are unique
        # Note: unique states do not have the 'gap' term in their linear
        #   equation, but duplicate states do. Hence the 0 for unique states'
        #   gap column and -1 for that of duplicates.
        feasible_states[is_unique == 1, -1] = 0  # unique states' gap column
        feasible_states[is_unique == 0, -1] = -1  # duplicate states' gap column
        unique_feasible_states = feasible_states[is_unique == 1]
        duplicate_feasible_states = feasible_states[is_unique == 0]

        # Returns a Scipy OptimizeResult
        new_excited_states = -np.vstack((excited_states, duplicate_feasible_states))
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
    h = weights[:m_linear]
    j = weights[m_linear:-2]
    offset = weights[-2]
    gap = weights[-1]

    # Create BQM
    new_bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    new_bqm.add_variables_from((v, bias) for v, bias in
                               zip(labels[:m_linear], h))
    new_bqm.add_interactions_from((u, v, bias) for (u, v), bias in
                                  zip(labels[m_linear:], j))
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
