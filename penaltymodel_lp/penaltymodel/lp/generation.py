from bisect import bisect_left
import dimod
from itertools import product
import numpy as np
import penaltymodel.core as pm
from scipy.optimize import linprog


#TODO: would be nice to have a file for default linear energy ranges (currently, [-2, 2]); quad energy [-1, 1]
def _get_lp_matrix(spin_states, nodes, edges, offset_weight, gap_weight):
    # Set up an empty matrix
    n_states = len(spin_states)
    m_linear = len(nodes)
    m_quadratic = len(edges)
    matrix = np.empty((n_states, m_linear + m_quadratic + 2))   # +2 columns to account for offset and gap

    # Populate linear terms (i.e. spin states)
    matrix[:, :m_linear] = spin_states

    # Populate quadratic terms
    for j, (u, v) in enumerate(edges):
        u_ind = bisect_left(nodes, u)
        v_ind = bisect_left(nodes, v)
        matrix[:, j + m_linear] = np.multiply(matrix[:, u_ind], matrix[:, v_ind])

    # Populate offset and gap columns, respectively
    matrix[:, -2] = offset_weight
    matrix[:, -1] = gap_weight
    return matrix


def generate_bqm(graph, table, decision_variables,
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 precision=7, max_decision=8, max_variables=10,
                 return_auxiliary=False):

    # TODO: better way of saying that toy does not deal with auxiliary
    # Check for auxiliary variables in the graph
    if len(graph) != len(decision_variables):
        raise ValueError('Penaltymodel-lp does not handle problems with auxiliary variables')

    # Sort graph information
    # Note: nodes' and edges' order determine the column order of the LP
    nodes = sorted(decision_variables)
    edges = graph.edges

    # Set variable names for lengths
    m_linear = len(nodes)                   # Number of linear biases
    m_quadratic = len(edges)                # Number of quadratic biases
    n_valid = len(table)                    # Number of valid spin combinations
    n_invalid = 2**(m_linear) - n_valid     # Number of invalid spin combinations

    # Determining valid and invalid spin states
    invalid_table = set(product([-1, 1], repeat=m_linear)) - set(table.keys())
    invalid_linear = np.array(list(invalid_table))
    valid_linear = np.array(list(table.keys()))

    # Linear programming matrix for valid spins
    valid_states = _get_lp_matrix(valid_linear, nodes, edges, 1, 0)

    # Linear programming matrix for invalid spins
    invalid_states = _get_lp_matrix(invalid_linear, nodes, edges, 1, -1)
    invalid_states = -1 * invalid_states    # Taking negative in order to flip the inequality

    # Bounds
    bounds = [linear_energy_ranges.get(node, (-2, 2)) for node in nodes]
    bounds += [quadratic_energy_ranges.get(edge, (-1, 1)) for edge in edges]
    bounds.append((None, None))     # for offset
    bounds.append((None, None))     # for gap

    # Cost function
    cost_weights = np.zeros((1, m_linear + m_quadratic + 2))
    cost_weights[0, -1] = -1     # Only interested in maximizing the gap

    # Returns a Scipy OptimizeResult
    result = linprog(cost_weights.flatten(), A_eq=valid_states, b_eq=np.zeros((n_valid, 1)), A_ub=invalid_states,
                     b_ub=np.zeros((n_invalid, 1)), bounds=bounds)

    # Split result
    x = result.x
    h = x[:m_linear]
    J = x[m_linear:-2]
    offset = x[-2]
    gap = x[-1]

    if gap <= 0:
        raise ValueError('Gap is not a positive value') # Should compare with min gap

    # Create BQM
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, round(bias, precision)) for v, bias in zip(nodes, h))
    bqm.add_interactions_from((u, v, round(bias, precision)) for (u, v), bias in zip(edges, J))
    bqm.add_offset(round(offset, precision))

    return bqm, gap
