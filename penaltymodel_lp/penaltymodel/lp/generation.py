import dimod
from itertools import product
import numpy as np
from operator import itemgetter
from scipy.optimize import linprog


def _get_lp_matrix(spin_states, nodes, edges, offset_weight, gap_weight):
    # Set up an empty matrix
    n_states = len(spin_states)
    m_linear = len(nodes)
    m_quadratic = len(edges)
    matrix = np.empty((n_states, m_linear + m_quadratic + 2))   # +2 columns to account for offset and gap

    # Populate linear terms (i.e. spin states)
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


def generate_bqm(graph, table, decision_variables,
                 linear_energy_ranges=None, quadratic_energy_ranges=None):

    #TODO: case where table is an iterable rather than dict
    # Check for auxiliary variables in the graph
    if len(graph) != len(decision_variables):
        raise ValueError('Penaltymodel-lp does not handle problems with auxiliary variables')

    if not linear_energy_ranges:
        linear_energy_ranges = {}

    if not quadratic_energy_ranges:
        quadratic_energy_ranges = {}

    # Simplify graph naming
    # Note: nodes' and edges' order determine the column order of the LP
    nodes = decision_variables
    edges = graph.edges

    # Set variable names for lengths
    m_linear = len(nodes)                   # Number of linear biases
    m_quadratic = len(edges)                # Number of quadratic biases
    n_noted = len(table)                    # Number of spin combinations specified in the table
    n_unnoted = 2**m_linear - n_noted       # Number of spin combinations of length `m_linear` that were not specified

    # Determining noted and unnoted spin states
    spin_states = product([-1, 1], repeat=m_linear)
    noted_linear = list(table.keys())
    unnoted_linear = [state for state in spin_states if state not in noted_linear]

    # Linear programming matrix for specified spins
    gap_weight = np.asarray([-table[state] for state in noted_linear])
    noted_states = _get_lp_matrix(np.asarray(noted_linear), nodes, edges, 1, gap_weight)

    # Linear programming matrix for spins that were not specified
    unnoted_states = _get_lp_matrix(np.asarray(unnoted_linear), nodes, edges, 1, -1)
    unnoted_states *= -1   # Taking negative in order to flip the inequality

    # Bounds
    bounds = [linear_energy_ranges.get(node, (-2, 2)) for node in nodes]
    bounds += [quadratic_energy_ranges.get(edge, (-1, 1)) for edge in edges]
    bounds.append((None, None))     # for offset
    bounds.append((None, None))     # for gap

    # Cost function
    cost_weights = np.zeros((1, m_linear + m_quadratic + 2))
    cost_weights[0, -1] = -1     # Only interested in maximizing the gap

    # Returns a Scipy OptimizeResult
    result = linprog(cost_weights.flatten(), A_eq=noted_states, b_eq=np.zeros((n_noted, 1)), A_ub=unnoted_states,
                     b_ub=np.zeros((n_unnoted, 1)), bounds=bounds)

    # Split result
    x = result.x
    h = x[:m_linear]
    J = x[m_linear:-2]
    offset = x[-2]
    gap = x[-1]

    if gap <= 0:
        raise ValueError('Gap is not a positive value') # TODO: Should compare with min gap

    # Create BQM
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, bias) for v, bias in zip(nodes, h))
    bqm.add_interactions_from((u, v, bias) for (u, v), bias in zip(edges, J))
    bqm.add_offset(offset)

    return bqm, gap
