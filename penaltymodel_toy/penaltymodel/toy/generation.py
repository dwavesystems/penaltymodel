import numpy as np
import penaltymodel.core as pm
from scipy.optimize import linprog

#TODO: would be nice to have a file for default linear energy ranges (currently, [-2, 2]); quad energy [-1, 1]

def generate_bqm(graph, table, decision_variables,
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 precision=7, max_decision=8, max_variables=10,
                 return_auxiliary=False):

    # Valid states

    # Invalid states

    # Bounds
    #TODO: assumes order of variables does not change
    #TODO: aux probably needs energy range too; I'm giving aux linear biases for now
    bounds = []
    for node in graph.nodes():
        try:
            bounds.append(linear_energy_ranges[node])
        except KeyError:
            bounds.append((-2, 2))

    for edge in graph.edges():
        try:
            bounds.append(quadratic_energy_ranges[edge])
        except KeyError:
            bounds.append((-1, 1))

    print("Inside toy generate bqm!")