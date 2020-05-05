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

import functools
import itertools
import random

import dimod
import numpy as np
from scipy.optimize import linprog

from penaltymodel.core import PenaltyModel
from penaltymodel.core.constants import (DEFAULT_LINEAR_RANGE,
                                         DEFAULT_QUADRATIC_RANGE)


def random_indices_generator(bin_sizes, n_tries):
    """Generate random indices such that there is one index picked from each bin.

    Args:
        bin_sizes():
        n_tries(integer): number of random tuples this generator will produce
    """
    for _ in range(n_tries):
        random_tuple = (random.randint(0, bin_size) for bin_size in bin_sizes)
        yield random_tuple


def get_ordered_state_matrix(linear_labels, quadratic_labels):
    """Returns a state matrix and its column labels.

    Specifically, the rows of the state matrix contains all possible states the
    linear and quadratic variables can take on. The columns of the state matrix
    follow the order [linear_labels, quadratic_labels, offset, gap]. As this is
    a state matrix, the offset and gap columns are held constant.

    Args:
        linear_labels: iterable of linear variables
        quadratic_labels: iterable of tuples representing quadratic variables
    """
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
    """Returns numpy vector of biases that follows the order of 'bias_order'.

    Args:
        linear_bias_dict(dict): key contains variable name, value contains bias
        quadratic_bias_dict(dict): key contains variable name, value contains bias
        bias_order(iterable): varible names indicating desired result order
        offset(number): the value of the offset
    """
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
    """Returns a list of tuple bounds that follow the variable ordering listed
    in 'order'. Each tuple indicates the (<lowerbound>, <upperbound>) of a
    variable.

    Args:
        linear_ranges(dict): key contains variable name; value contains tuple bound
        quadratic_ranges(dict): key contains variable name; value contains tuple bound
        order(iterable): list of variable names ordered in the desired order of the result
        min_classical_gap(number): a number used as the lowerbound of the gap
        default_linear_range(tuple): a tuple of bounds used when a variable does not
          exist in 'linear_ranges' dict
        default_quadratic_range(tuple): a tuple of bounds used when a variable does
          not exist in 'quadratic_ranges' dict
    """
    # Note: Since ising has {-1, 1}, the largest possible gap is [-largest_bias,
    #   largest_bias], hence that 2 * sum(largest_biases)
    lr = linear_ranges
    qr = quadratic_ranges
    default_lr = default_linear_range
    default_qr = default_quadratic_range

    bounds = []
    for k in order:
        if isinstance(k, tuple) and len(k) == 2 and k[0] != k[1]:
            try:
                quadratic_bias_bound = qr[k[0]][k[1]]
            except KeyError:
                quadratic_bias_bound = default_qr

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

        # Reached last item of current bin
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

    # Determine feasible states with a flag
    feasible_flag = energy <= pmodel.ground_energy
    if sum(feasible_flag) == 0:
        raise RuntimeError("no states with energies less than or equal to the"
                           " ground_energy found; no feasible states found")

    # Check for uniqueness, which is trivially balanced
    if sum(feasible_flag) == len(pmodel.feasible_configurations):
        return pmodel

    # Cost function
    cost_weights = np.zeros((1, states.shape[1]))
    cost_weights[0, -1] = -1  # Only interested in maximizing the gap

    bounds = get_bounds(pmodel.ising_linear_ranges, pmodel.ising_quadratic_ranges,
                        labels, pmodel.min_classical_gap)

    # Bin auxiliary configs with the same decision configs together
    step = 2**len(aux_variables)
    bins, bin_sizes = get_binned_indices(feasible_flag, step)

    # Determine index generator
    # Either try all possible feasible combinations or try a random subset of
    # combinations
    n_possibilities = functools.reduce(lambda a, b: a*b, bin_sizes)
    if n_possibilities <= n_tries:
        index_gen = itertools.product(*(range(x) for x in bin_sizes))
    else:
        index_gen = random_indices_generator(bins, n_tries)

    # Attempt to find solution
    gap_threshold = max(pmodel.min_classical_gap - tol, 0)
    for row_indices in index_gen:
        index = [bin[i] for i, bin in zip(row_indices, bins)]

        is_unique = np.zeros(len(states), dtype=bool)
        is_unique[index] = True

        states[is_unique, -1] = 0
        states[np.logical_not(is_unique), -1] = -1
        unique_feasible_states = states[is_unique, :]
        new_excited_states = -states[np.logical_not(is_unique), :]

        # Returns a Scipy OptimizeResult
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

    # Remove floating point rounding errors
    # Note: I am not checking that the weight is within a certain epsilon
    #   beyond the bound because if it passed the linear program (i.e. obeyed
    #   the constraints), I am assuming the error is a small rounding issue.
    weights = result.x
    for i, (weight, bound) in enumerate(zip(weights, bounds)):
        lbound, ubound = bound

        if lbound is not None:
            weights[i] = max(weight, lbound)
        if ubound is not None:
            weights[i] = min(weight, ubound)

    # Parse result
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

    # Copy pmodel with updated bqm and gap
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
