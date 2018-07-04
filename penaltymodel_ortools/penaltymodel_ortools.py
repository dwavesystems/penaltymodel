import itertools
import logging
import time
import random
import timeit

import dimod
import networkx as nx

from ortools.linear_solver import pywraplp

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()


def make_pm_complete(table, decision, graph, precision=7):

    auxiliary = [v for v in graph if v not in decision]
    variables = decision + auxiliary

    solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    h = {v: solver.NumVar(-2.0, 2.0, 'h_%s' % v) for v in graph.nodes}
    J = {(u, v): solver.NumVar(-1.0, 1.0, 'J_%s,%s' % (u, v)) for u, v, in graph.edges}
    offset = solver.NumVar(-solver.infinity(), solver.infinity(), 'offset')
    gap = solver.NumVar(0, solver.infinity(), 'classical_gap')

    # Let x, a be the decision, auxiliary variables respectively
    # Let E(x, a) be the energy of x and a
    # Let F be the feasible configurations of x
    # Let g be the classical gap
    # Let a*(x) be argmin_a E(x, a) - the config of aux variables that minimizes the energy with x fixed

    # We want:
    #   E(x, a) >= 0  forall x in F, forall a
    #   E(x, a) - g >= 0  forall x not in F, forall a
    for config in itertools.product((-1, 1), repeat=len(variables)):
        spins = dict(zip(variables, config))

        const = solver.Constraint(0.0, solver.infinity())

        if tuple(spins[v] for v in decision) not in table:
            # we want energy greater than gap for decision configs not in feasible
            const.SetCoefficient(gap, -1)

        # add the energy for the configuration
        for v, bias in h.items():
            const.SetCoefficient(bias, spins[v])
        for (u, v), bias in J.items():
            const.SetCoefficient(bias, spins[u] * spins[v])
        const.SetCoefficient(offset, 1)

    if not auxiliary:
        # We have no auxiliary variables. We want:
        #   E(x) <= 0 forall x in F
        for decision_config in table:
            spins = dict(zip(decision, decision_config))

            const = solver.Constraint(-solver.infinity(), 0.0)

            # add the energy for the configuration
            for v, bias in h.items():
                const.SetCoefficient(bias, spins[v])
            for (u, v), bias in J.items():
                const.SetCoefficient(bias, spins[u] * spins[v])
            const.SetCoefficient(offset, 1)

    else:
        # We have auxiliary variables. So that each feasible config has at least one ground we want:
        #   E(x, a) - 100*|| a - a*(x) || <= 0  forall x in F, forall a

        # we need a*(x) forall x in F
        a_star = {config: {v: solver.IntVar(0, 1, 'a*(%s)_%s' % (config, v)) for v in auxiliary} for config in table}

        for decision_config in table:
            for aux_config in itertools.product((-1, 1), repeat=len(variables) - len(decision)):
                spins = dict(zip(variables, decision_config+aux_config))

                ub = 0

                # the E(x, a) term
                coefficients = {bias: spins[v] for v, bias in h.items()}
                coefficients.update({bias: spins[u] * spins[v] for (u, v), bias in J.items()})
                coefficients[offset] = 1

                # # the -100*|| a - a*(x) || term
                auxiliary_coefficients = {}
                for v in auxiliary:
                    # we don't have absolute value, so we check what a is and order the subtraction accordingly
                    if spins[v] == -1:
                        # a*(x)_v - a_v
                        coefficients[a_star[decision_config][v]] = -200
                    else:
                        # a_v - a*(x)_v
                        assert spins[v] == 1  # sanity check
                        coefficients[a_star[decision_config][v]] = +200
                        ub += 200

                const = solver.Constraint(-solver.infinity(), ub)
                for var, coef in coefficients.items():
                    const.SetCoefficient(var, coef)

        # without loss of generality we can fix the auxiliary variables associated with
        # one of the feasible configurations. Do so randomly.
        for var in next(iter(a_star.values())).values():
            val = random.randint(0, 1)
            const = solver.Constraint(val, val)  # equality constrait
            const.SetCoefficient(var, 1)

    objective = solver.Objective()
    objective.SetCoefficient(gap, 1)
    objective.SetMaximization()

    result_status = solver.Solve()

    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

    for v, bias in h.items():
        bqm.add_variable(v, round(bias.solution_value(), precision))
    for (u, v), bias in J.items():
        bqm.add_interaction(u, v, round(bias.solution_value(), precision))
    bqm.add_offset(round(offset.solution_value(), precision))

    return bqm, round(gap.solution_value(), precision)

# EQ = {(-1, -1), (1, 1)}
# decision_variables = ['x', 'y']
# auxiliary_variables = []

# make_pm_complete(EQ, decision_variables, nx.complete_graph(decision_variables+['a']))
# make_pm_complete(EQ, decision_variables, nx.complete_graph(decision_variables))


# AND = {(-1, -1, -1),
#        (-1, +1, -1),
#        (+1, -1, -1),
#        (+1, +1, +1)}
# decision_variables = ['in0', 'in1', 'out']
# # auxiliary_variables = list(range(1))
# auxiliary_variables = ['aux']

# bqm, gap = make_pm_complete(AND, decision_variables, nx.complete_graph(decision_variables+auxiliary_variables))

# print(bqm)
# print(gap)

# resp = dimod.ExactSolver().sample(bqm)

# for sample, en in resp.data(['sample', 'energy']):
#     print([sample[v] for v in decision_variables], en)


# def xor(n):
#     return {config+(+1,) if sum(config) not in (-n, n) else config+(-1,)
#             for config in itertools.product((-1, 1), repeat=n)}


# for n in range(1, 5):
#     table = xor(n)

#     decision = ['v%s' % d for d in range(n)]
#     decision.append('out')

#     variables = decision.copy()

#     for __ in range(5):
#         t = time.time()
#         bqm, gap = make_pm_complete(table, decision, nx.complete_graph(variables))
#         t = time.time() - t
#         print('XOR({}), aux: {}, gap: {}, aux: True, time: {}'.format(n, 0, round(gap, 3), t))

#     for __ in range(5):
#         t = time.time()
#         bqm, gap = make_pm_complete(table, decision, nx.complete_graph(variables), _aux=False)
#         t = time.time() - t
#         print('XOR({}), aux: {}, gap: {}, aux: False, time: {}'.format(n, 0, round(gap, 3), t))

#     while abs(gap - 0) < .0001:
#         variables.append('aux%s' % (len(variables) - n))

#         for __ in range(5):
#             t = time.time()
#             bqm, gap = make_pm_complete(table, decision, nx.complete_graph(variables), _aux=True)
#             t = time.time() - t

#             print('XOR({}), aux: {}, gap: {}, aux: True, time: {}'.format(n, len(variables) - len(decision), gap, t))

#         for __ in range(5):
#             t = time.time()
#             bqm, gap = make_pm_complete(table, decision, nx.complete_graph(variables), _aux=False)
#             t = time.time() - t

#             print('XOR({}), aux: {}, gap: {}, aux: False, time: {}'.format(n, len(variables) - len(decision), gap, t))

#     # should have a positive gap so check it
#     response = dimod.ExactSolver().sample(bqm)

#     seen = set()
#     for sample, energy in response.data(['sample', 'energy']):

#         if energy > .00001:
#             break

#         seen.add(tuple(int(sample[v]) for v in decision))

#     assert seen == table
