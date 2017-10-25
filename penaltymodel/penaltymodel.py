from numbers import Number

from six import itervalues, iteritems

SPIN = -1
BINARY = 0
VARTYPES = {SPIN, BINARY}


class PenaltyModel(object):
    def __init__(self, graph, decision_variables, constraint, model, infeasible_gap):
        self.graph = graph
        self.decision_variables = decision_variables
        self.constraint = constraint
        self.model = model
        self.infeasible_gap = infeasible_gap

        # it would be good to add the model to the networkx graph


# class BQP(object):
#     def __init__(self, linear, quadratic, offset, vartype):
#         if not isinstance(linear, dict):
#             raise TypeError("expected input 'linear' to be a dict")
#         # for now we are going to be agnostic about the keys and values
#         self.linear = linear

#         if not isinstance(quadratic, dict):
#             raise TypeError("expected input 'quadratic' to be a dict")
#         # for now we are going to be agnostic about the values
#         try:
#             if not all(u in linear and v in linear for u, v in quadratic):
#                 raise ValueError("variables in 'quadratic' must also be in linear")
#         except ValueError:
#             raise TypeError("keys of quadratic must be 2-tuples")
#         self.quadratic = quadratic
#         self.adj = adj = {}
#         for (u, v), bias in iteritems(quadratic):
#             if u == v:
#                 linear[u] += bias
#                 continue

#             if u in adj:
#                 if v in adj[u]:
#                     adj[u][v] += bias
#                     adj[v][u] += bias
#                 else:
#                     adj[u][v] = bias
#                     adj[v][u] = bias
#             else:
#                 adj[u] = {v: bias}
#                 adj[v] = {u: bias}

#         # we will also be agnostic to the offset type, the user can determine what makes sense
#         self.offset = offset

#         if vartype not in VARTYPES:
#             raise ValueError("unknown vartype")
#         self.vartype = vartype


# class Ising(BQP):
#     def __init__(self, linear, quadratic, offset=0.0):
#         BQP.__init__(self, linear, quadratic, offset, SPIN)


# class QUBO(object):
#     def __init__(self, linear, quadratic, offset=0.0):
#         # for qubo, quadratic and linear are equivelent
#         BQP.__init__(self, linear, quadratic, offset, BINARY)
