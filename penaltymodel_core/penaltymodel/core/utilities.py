import dimod
import itertools
import numpy as np
from scipy.optimize import linprog


#TODO: potentially duplicate code with LP; will refactor once this code is more stable
def _get_lp_matrix(spin_states, nodes, edges, offset_weight, gap_weight):
    """Creates an linear programming matrix based on the spin states, graph, and scalars provided.
    LP matrix:
        [spin_states, corresponding states of edges, offset_weight, gap_weight]

    Args:
        spin_states: Numpy array of spin states
        nodes: Iterable
        edges: Iterable of tuples
        offset_weight: Numpy 1-D array or number
        gap_weight: Numpy 1-D array or a number
    """
    if len(spin_states) == 0:
        return None

    # Set up an empty matrix
    n_states = len(spin_states)
    m_linear = len(nodes)
    m_quadratic = len(edges)
    matrix = np.empty((n_states, m_linear + m_quadratic + 2))  # +2 columns for offset and gap

    # Populate linear terms (i.e. spin states)
    if spin_states.ndim == 1:
        spin_states = np.expand_dims(spin_states, 1)
    matrix[:, :m_linear] = spin_states

    # Populate quadratic terms
    node_indices = dict(zip(nodes, range(m_linear)))
    for j, (u, v) in enumerate(edges):
        u_ind = node_indices[u]
        v_ind = node_indices[v]
        matrix[:, j + m_linear] = np.multiply(matrix[:, u_ind], matrix[:, v_ind])

    # Populate offset and gap columns, respectively
    matrix[:, -2] = offset_weight
    matrix[:, -1] = gap_weight
    return matrix


def balance(pmodel, n_tries=100):
    # TODO: Provide QUBO support
    # TODO: could probably put the matrix construction in its own function
    # TODO: multiple ground states
    if not pmodel.model:
        raise ValueError("There is no model to balance")

    # Set up
    bqm = pmodel.model
    m_linear = len(bqm.linear)
    m_quadratic = len(bqm.quadratic)
    labels = list(bqm.linear.keys()) + list(bqm.quadratic.keys())
    indices = {k: i for i, k in enumerate(labels)}  # map labels to column index

    # Construct the states matrix
    # Construct linear portion of states matrix
    states = np.empty((2 ** m_linear, m_linear + m_quadratic + 2), dtype=int)  # +2 for offset and gap cols
    states[:, :m_linear] = np.array([list(x) for x in
                                     itertools.product({-1, 1}, repeat=m_linear)])
    states[:, -2] = 1  # column for offset
    states[:, -1] = -1  # column for gap

    # Construct quadratic portion of states matrix
    for node0, node1 in bqm.quadratic.keys():
        edge_ind = indices[(node0, node1)]
        node0_ind = indices[node0]
        node1_ind = indices[node1]
        states[:, edge_ind] = states[:, node0_ind] * states[:, node1_ind]

    # Construct biases and energy vectors
    biases = [bqm.linear[label] for label in labels[:m_linear]]
    biases += [bqm.quadratic[label] for label in labels[m_linear:]]
    biases += [bqm.offset]
    biases = np.array(biases)
    energy = np.matmul(states[:, :-1], biases)  # Ignore last column; gap column

    # Group states by threshold
    excited_states = states[energy > pmodel.ground_energy]
    feasible_states = states[energy <= pmodel.ground_energy]

    # Check for balance
    if len(feasible_states) == len(pmodel.feasible_configurations):
        return

    # Cost function
    cost_weights = np.zeros((1, states.shape[1]))
    cost_weights[0, -1] = -1  # Only interested in maximizing the gap

    # Note: Since ising has {-1, 1}, the largest possible gap is [-largest_bias, largest_bias],
    #   hence that 2 * sum(largest_biases)
    # TODO remove default hardcoded bounds
    bounds = [pmodel.ising_linear_ranges.get(label, (-2, 2)) for label in labels[:m_linear]]
    bounds += [pmodel.ising_quadratic_ranges.get(label, (-1, 1)) for label in labels[m_linear:]]
    max_gap = 2 * sum(max(abs(lbound), abs(ubound)) for lbound, ubound in bounds)
    bounds.append((None, None))  # Bound for offset
    bounds.append((0, max_gap))  # Bound for gap.

    # Determine duplicate decision states
    # Note: we are forming a new matrix, decision_cols, which is made up of the decision
    #   variable columns. We use decision_cols to bin like-feasible_states together (i.e. same
    #   decision state values, potentially different aux values).
    # Note2: using lexsort so that each row of decision_cols is treated as a single object with
    #   primary, secondary, tertiary, etc key orders
    # Note3: bins contains the index of the last item in each bin; these are the bin boundaries
    decision_indices = [indices[label] for label in pmodel.decision_variables]
    decision_cols = feasible_states[:, decision_indices]
    sorted_indices = np.lexsort(decision_cols.T)
    decision_cols = decision_cols[sorted_indices, :]
    feasible_states = feasible_states[sorted_indices, :]
    bins = (decision_cols[:-1, :] != decision_cols[1:, :]).any(axis=1)
    bins = np.append(bins, True)  # Marking the end of the last bin
    bins = np.nonzero(bins)[0]

    # Get number of unique decision states and number of items in each bin
    n_uniques = bins.shape[0]
    bin_count = np.hstack((bins[0] + 1, bins[1:] - bins[:-1]))  # +1 to account for zero-index

    # Store solution with largest gap
    best_gap = 0
    best_result = None
    for _ in range(n_tries):
        # Generate random indices such that there is one index picked from each bin
        random_indices = np.random.rand(n_uniques) * bin_count
        random_indices = np.floor(random_indices).astype(np.int)
        random_indices[1:] += (bins[:-1] + 1)  # add bin offsets; +1 to negate bins' zero-index
        is_unique = np.zeros(feasible_states.shape[0], dtype=int)
        is_unique[random_indices] = 1

        # Select which feasible states are unique
        # Note: unique states do not have the 'gap' term in their linear equation, but duplicate
        #   states do. Hence the 0 for unique states' gap column and -1 for that of duplicates.
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
                         method="simplex")

        if not result.success:
            continue

        # Store best result
        gap = result.x[-1]
        if gap > best_gap:
            best_result = result
            best_gap = gap

    # Parse result
    weights = best_result.x
    h = weights[:m_linear]
    j = weights[m_linear:-2]
    offset = weights[-2]
    gap = weights[-1]

    # TODO: Test that gap meets user's gap requirements
    if gap <= 0:
        raise ValueError('Unable to balance this penaltymodel, hence no changes will be made.')

    # Create BQM
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, bias) for v, bias in zip(labels[:m_linear], h))
    bqm.add_interactions_from((u, v, bias) for (u, v), bias in zip(labels[m_linear:], j))
    bqm.add_offset(offset)

    return bqm