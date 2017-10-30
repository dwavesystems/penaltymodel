from numbers import Number

from six import itervalues, iteritems

SPIN = -1
BINARY = 0
VARTYPES = {SPIN, BINARY}


class Specification(object):
    def __init__(self, graph, decision_variables, feasible_configurations,
                 vartype=SPIN,
                 linear_energy_range=None, quadratic_energy_range=None):
        pass


# class PenaltyModel(object):
#     def __init__(self, graph, decision_variables, constraint, model, infeasible_gap):
#         self.graph = graph
#         self.decision_variables = decision_variables
#         self.constraint = constraint
#         self.model = model
#         self.infeasible_gap = infeasible_gap

#         # it would be good to add the model to the networkx graph


class Ising(object):
    def __init__(self, linear, quadratic={}, offset=0.0):

        if isinstance(linear, QUBO):
            if quadratic or offset:
                raise ValueError("if input is a QUBO, no other args can be set")
            model = linear
            linear, quadratic, offset = dimod.qubo_to_ising(model.quadratic)
            offset += model.offset

        self.linear = linear
        self.quadratic = quadratic
        self.offset = offset


class QUBO():
    def __init__(self, model, offset=0.0):
        """

        model = QUBO(Q, offset)
        model = QUBO(Q)
        model = QUBO(ising_model)

        """

        # input parsing
        if isinstance(model, Ising):
            # if the model is an Ising object, we need to convert it to a QUBO.

            quadratic, off = ising_to_qubo(model.linear, model.quadratic, model.offset)

            # we have three potential offset sources, the optional parameter,
            # the model and the offset induced by the conversion
            offset += off

        elif isinstance(model, dict):
            # on the other hand model might be the quadratic biases
            if not all(len(edge) == 2 for edge in model):
                raise ValueError("edges in 'model' should be 2-tuples")

            # we are agnostic about the form of the biases
            quadratic = model
        else:
            raise TypeError("input 'model' should be an Ising or a dict")

        self.quadratic = quadratic
        self.offset = offset


def ising_to_qubo(h, J, offset=0.0):
    """Converts an Ising problem to a QUBO problem.

    Map an Ising model defined over -1/+1 variables to a binary quadratic
    program x' * Q * x defined over 0/1 variables. We return the Q defining
    the BQP model as well as the offset in energy between the two problem
    formulations, i.e. s' * J * s + h' * s = offset + x' * Q * x. The linear term
    of the BQP is contained along the diagonal of Q.

    See qubo_to_ising(Q) for the inverse function.

    Args:
        h (dict): A dict of the linear coefficients of the Ising problem.
        J (dict): A dict of the quadratic coefficients of the Ising problem.
        offset (float): An energy offset to be applied

    Returns:
        (dict, float): A dict of the QUBO coefficients. The energy offset.

    """
    # the linear biases are the easiest
    q = {(v, v): 2. * bias for v, bias in iteritems(h)}

    # next the quadratic biases
    for (u, v), bias in iteritems(J):
        if bias == 0.0:
            continue
        q[(u, v)] = 4. * bias
        q[(u, u)] -= 2. * bias
        q[(v, v)] -= 2. * bias

    # finally calculate the offset
    offset += sum(itervalues(J)) - sum(itervalues(h))

    return q, offset


def qubo_to_ising(Q, offset=0.0):
    """Converts a QUBO problem to an Ising problem.

    Map a binary quadratic program x' * Q * x defined over 0/1 variables to
    an Ising model defined over -1/+1 variables. We return the h and J
    defining the Ising model as well as the offset in energy between the
    two problem formulations, i.e. x' * Q * x = offset + s' * J * s + h' * s. The
    linear term of the QUBO is contained along the diagonal of Q.

    See ising_to_qubo(h, J) for the inverse function.

    Args:
        Q: A dict of the QUBO coefficients.
        offset (float): An energy offset to be applied

    Returns:
        (dict, dict, float):
        A dict of the linear coefficients of the Ising problem.
        A dict of the quadratic coefficients of the Ising problem.
        The energy offset.

    """
    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    for (u, v), bias in iteritems(Q):
        if u == v:
            if u in h:
                h[u] += .5 * bias
            else:
                h[u] = .5 * bias
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = .25 * bias

            if u in h:
                h[u] += .25 * bias
            else:
                h[u] = .25 * bias

            if v in h:
                h[v] += .25 * bias
            else:
                h[v] = .25 * bias

            quadratic_offset += bias

    offset += .5 * linear_offset + .25 * quadratic_offset

    return h, J, offset


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





# class QUBO(object):
#     def __init__(self, linear, quadratic, offset=0.0):
#         # for qubo, quadratic and linear are equivelent
#         BQP.__init__(self, linear, quadratic, offset, BINARY)
