from bisect import bisect_left
import dimod
from itertools import product
import numpy as np
import penaltymodel.core as pm
from scipy.optimize import linprog

#TODO: would be nice to have a file for default linear energy ranges (currently, [-2, 2]); quad energy [-1, 1]

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
    all_linear = set(product([-1, 1], repeat=m_linear))
    valid_linear = set(table.keys())
    invalid_linear = all_linear - valid_linear

    # Valid states
    valid_states = np.empty((n_valid, m_linear + m_quadratic + 2))
    valid_states[:, :m_linear] = np.asarray(list(valid_linear))      # Populate linear spins

    for j, (u, v) in enumerate(edges):
        u_ind = bisect_left(nodes, u)
        v_ind = bisect_left(nodes, v)

        #TODO: math is so simple, may not need to multiply; could do pattern matching
        valid_states[:, j + m_linear] = np.multiply(valid_states[:, u_ind], valid_states[:, v_ind])

    valid_states[:, -2] = 1     # Column associated with offset
    valid_states[:, -1] = 0     # Column associated with gap

    # Invalid states
    invalid_states = np.empty((n_invalid, m_linear + m_quadratic + 2))
    invalid_states[:, :m_linear] = np.asarray(list(invalid_linear))      # Populate linear spins

    for j, (u, v) in enumerate(edges):
        u_ind = bisect_left(nodes, u)
        v_ind = bisect_left(nodes, v)

        #TODO: math is so simple, may not need to multiply; could do pattern matching
        invalid_states[:, j + m_linear] = np.multiply(invalid_states[:, u_ind], invalid_states[:, v_ind])

    invalid_states[:, -2] = 1     # Column associated with offset
    invalid_states[:, -1] = -1     # Column associated with gap
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
    result = linprog(cost_weights.flatten(),
                                            A_eq=valid_states, b_eq=np.zeros((n_valid, 1)),
                                            A_ub=invalid_states, b_ub=np.zeros((n_invalid, 1)),
                                            bounds=bounds)

    x = result.x
    h = x[:m_linear]
    J = x[m_linear:-2]
    offset = x[-2]
    gap = x[-1]
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    bqm.add_variables_from((v, round(bias, precision)) for v, bias in zip(nodes, h))
    bqm.add_interactions_from((u, v, round(bias, precision)) for (u, v), bias in zip(edges, J))
    bqm.add_offset(round(offset, precision))
    return bqm, gap
