from bisect import bisect_left
import dimod
from itertools import product
import numpy as np
import penaltymodel.core as pm
from scipy.optimize import linprog

#TODO: would be nice to have a file for default linear energy ranges (currently, [-2, 2]); quad energy [-1, 1]

def _get_lp_matrix(spin_states, nodes, edges, offset_weight, gap_weight):
    n_states = len(spin_states)
    n_nodes = len(nodes)
    n_edges = len(edges)

    matrix = np.empty((n_states, n_nodes + n_edges + 2))
    matrix[:, :n_nodes] = spin_states      # Populate linear spins

    for j, (u, v) in enumerate(edges):
        u_ind = bisect_left(nodes, u)
        v_ind = bisect_left(nodes, v)
        matrix[:, j + n_nodes] = np.multiply(matrix[:, u_ind], matrix[:, v_ind])

    matrix[:, -2] = offset_weight     # Column associated with offset
    matrix[:, -1] = gap_weight     # Column associated with gap
    return matrix


def generate_bqm(graph, table, decision_variables,
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 precision=7, max_decision=8, max_variables=10,
                 return_auxiliary=False):

    # TODO: better way of saying that toy does not deal with auxiliary
    # Check for auxiliary variables in the graph
    if len(graph) != len(decision_variables):
        return

    # Sort graph information
    nodes = sorted(decision_variables)
    edges = graph.edges

    # Set variable names for lengths
    m_linear = len(decision_variables)      # Number of linear biases
    m_quadratic = len(graph.edges)          # Number of quadratic biases
    n_valid = len(table)                    # Number of valid spin combinations
    n_invalid = 2**(m_linear) - n_valid     # Number of invalid spin combinations

    # Determining valid and invalid states
    invalid_table = set(product([-1, 1], repeat=m_linear)) - set(table.keys())
    invalid_linear = np.array(list(invalid_table))
    valid_linear = np.array(list(table.keys()))

    # Valid states
    valid_states = _get_lp_matrix(valid_linear, nodes, edges, 1, 0)

    # Invalid states
    invalid_states = _get_lp_matrix(invalid_linear, nodes, edges, 1, -1)
    invalid_states = -1 * invalid_states # Taking negative in order to flip the inequality

    # Cost function
    cost_weights = np.zeros((1, m_linear + m_quadratic + 2))
    cost_weights[0, -1] = -1     # Only interested in maximizing the gap

    # Bounds
    bounds = []
    for node in nodes:
        try:
            bounds.append(linear_energy_ranges[node])
        except KeyError:
            bounds.append((-2, 2))

    for edge in edges:
        try:
            bounds.append(quadratic_energy_ranges[edge])
        except KeyError:
            bounds.append((-1, 1))

    bounds.append((None, None))     # for offset
    bounds.append((None, None))     # for gap

    # Returns a Scipy OptimizeResult
    result = linprog(cost_weights.flatten(), A_eq=valid_states, b_eq=np.zeros((n_valid, 1)), A_ub=invalid_states,
                     b_ub=np.zeros((n_invalid, 1)), bounds=bounds)

    x = result.x  # x = [h biases, J biases, offset, gap]
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, round(bias, precision)) for v, bias in zip(nodes, x[:m_linear]))     # h bias
    bqm.add_interactions_from((u, v, round(bias, precision)) for (u, v), bias in zip(edges, x[m_linear:-2])) # J bias
    bqm.add_offset(round(x[-2], precision)) # offset
    return bqm, x[-1] # bqm, gap
