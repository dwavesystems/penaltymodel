"""
BinaryQuadraticModel
--------------------
"""
from __future__ import absolute_import

from six import itervalues, iteritems

from penaltymodel.classes.vartypes import Vartype

__all__ = ['BinaryQuadraticModel']


class BinaryQuadraticModel(object):
    """Encodes a binary quadratic model.

    Args:
        linear (dict): The linear biases as a dict. The keys should be the
            variables of the binary quadratic model. The values should be
            the linear bias associated with each variable.
        quadratic (dict): The quadratic biases as a dict. The keys should
            be 2-tuples of variables. The values should be the quadratic
            bias associated with each pair of variables. `quadratic`
            should be upper triangular, that is if (u, v) in `quadratic`
            then (v, u) should not be in `quadratic`. `quadratic` also
            should not have self loops, that is (u, u) is not a valid
            quadratic bias.
        offset: The energy offset associated with the model.
        vartype (:class:`.Vartype`): The variable type.

    Notes:
        The BinaryQuadraticModel does not specify the type of the biases
        and offset, but many

    Examples:
        >>> model = pm.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                 {(0, 1): .5, (1, 2): 1.5},
        ...                                 1.4,
        ...                                 pm.BinaryQuadraticModel.SPIN)
        >>> for u, v in model.quadratic:
        ...     assert model.quadratic[(u, v)] == model.adj[u][v]
        ...     assert model.quadratic[(u, v)] == model.adj[v][u]

    Attributes:
        linear (dict): The linear biases as a dict. The keys are the
            variables of the binary quadratic model. The values are
            the linear biases associated with each variable.
        quadratic (dict): The quadratic biases as a dict. The keys are
            2-tuples of variables. The values are the quadratic
            biases associated with each pair of variables.
        offset: The energy offset associated with the model. Same type as given
            on instantiation.
        vartype (:class:`.Vartype`): The variable type. `BinaryQuadraticModel.SPIN` or
            `BinaryQuadraticModel.BINARY`.
        adj (dict): The adjacency dict of the model. See examples.
        Vartype (:class:`.Vartype`): An alias for :class:`.Vartype` for easier access.
        SPIN (:class:`.Vartype`): An alias for :class:`.SPIN` for easier access.
        BINARY (:class:`.Vartype`): An alias for :class:`.BINARY` for easier access.

    """

    SPIN = Vartype.SPIN
    BINARY = Vartype.BINARY
    Vartype = Vartype

    def __init__(self, linear, quadratic, offset, vartype):
        # make sure that we are dealing with a known vartype.
        try:
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)
        except (ValueError, KeyError):
            raise TypeError("unexpected `vartype`. See BinaryQuadraticModel.Vartype for known types.")
        self.vartype = vartype

        # We want the linear terms to be a dict.
        # The keys are the variables and the values are the linear biases.
        # Model is deliberately agnostic to the type of the variable names
        # and the biases.
        if not isinstance(linear, dict):
            raise TypeError("expected `linear` to be a dict")
        self.linear = linear

        # We want quadratic to be a dict.
        # The keys should be 2-tuples of the form (u, v) where both u and v
        # are in linear.
        # We are agnostic to the type of the bias.
        if not isinstance(quadratic, dict):
            raise TypeError("expected `quadratic` to be a dict")
        try:
            if not all(u in linear and v in linear for u, v in quadratic):
                raise ValueError("each u, v in `quadratic` must also be in `linear`")
        except ValueError:
            raise ValueError("keys of `quadratic` must be 2-tuples")
        self.quadratic = quadratic

        # Build the adjacency. For each (u, v), bias in quadratic, we want:
        #    adj[u][v] == adj[v][u] == bias
        self.adj = adj = {}
        for (u, v), bias in iteritems(quadratic):
            if u == v:
                raise ValueError("bias ({}, {}) in `quadratic` is a linear bias".format(u, v))

            if u in adj:
                if v in adj[u]:
                    raise ValueError(("`quadratic` must be upper triangular. "
                                      "That is if (u, v) in `quadratic`, (v, u) not in quadratic"))
                else:
                    adj[u][v] = bias
            else:
                adj[u] = {v: bias}

            if v in adj:
                if u in adj[v]:
                    raise ValueError(("`quadratic` must be upper triangular. "
                                      "That is if (u, v) in `quadratic`, (v, u) not in quadratic"))
                else:
                    adj[v][u] = bias
            else:
                adj[v] = {u: bias}

        # we will also be agnostic to the offset type, the user can determine what makes sense
        self.offset = offset

    def __repr__(self):
        return 'BinaryQuadraticModel({}, {}, {}, BinaryQuadraticModel.{})'.format(self.linear, self.quadratic,
                                                                                  self.offset, self.vartype)

    def __eq__(self, model):
        """Model is equal if linear, quadratic, offset and vartype are all equal."""
        if not isinstance(model, BinaryQuadraticModel):
            return False

        if self.vartype == model.vartype:
            return all([self.linear == model.linear,
                        self.quadratic == model.quadratic,
                        self.offset == model.offset])
        else:
            # different vartypes are not equal
            return False

    def __len__(self):
        """The length is number of variables."""
        return len(self.linear)

    def as_ising(self):
        """Converts the model into the (h, J, offset) Ising format.

        If the model type is not spin, it is converted.

        Returns:
            dict: The linear biases.
            dict: The quadratic biases.
            The offset.

        """
        if self.vartype == self.SPIN:
            # can just return the model as-is.
            return self.linear, self.quadratic, self.offset

        if self.vartype != self.BINARY:
            raise RuntimeError('converting from unknown vartype')

        h = {}
        J = {}
        linear_offset = 0.0
        quadratic_offset = 0.0

        linear = self.linear
        quadratic = self.quadratic

        for u, bias in iteritems(linear):
            h[u] = .5 * bias
            linear_offset += bias

        for (u, v), bias in iteritems(quadratic):

            if bias != 0.0:
                J[(u, v)] = .25 * bias

            h[u] += .25 * bias
            h[v] += .25 * bias

            quadratic_offset += bias

        offset = self.offset + .5 * linear_offset + .25 * quadratic_offset

        return h, J, offset

    def as_qubo(self):
        """Converts the model into the (Q, offset) QUBO format.

        If the model type is not binary, it is converted.

        Returns:
            dict: The qubo biases as an edge dict.

            The offset.

        """
        if self.vartype == self.BINARY:
            # need to dump the linear biases into quadratic
            qubo = {}
            for v, bias in iteritems(self.linear):
                qubo[(v, v)] = bias
            for edge, bias in iteritems(self.quadratic):
                qubo[edge] = bias
            return qubo, self.offset

        if self.vartype != self.SPIN:
            raise RuntimeError('converting from unknown vartype')

        linear = self.linear
        quadratic = self.quadratic

        # the linear biases are the easiest
        qubo = {(v, v): 2. * bias for v, bias in iteritems(linear)}

        # next the quadratic biases
        for (u, v), bias in iteritems(quadratic):
            if bias == 0.0:
                continue
            qubo[(u, v)] = 4. * bias
            qubo[(u, u)] -= 2. * bias
            qubo[(v, v)] -= 2. * bias

        # finally calculate the offset
        offset = self.offset + sum(itervalues(quadratic)) - sum(itervalues(linear))

        return qubo, offset

    def energy(self, sample):
        """Determines the energy of the given sample.

        Args:
            sample (dict): The sample. The keys should be the variables and
                the values should be the value associated with each variable.

        Returns:
            float: The energy.

        """
        linear = self.linear
        quadratic = self.quadratic

        en = self.offset
        en += sum(linear[v] * sample[v] for v in linear)
        en += sum(quadratic[(u, v)] * sample[u] * sample[v] for u, v in quadratic)
        return en

    def relabel_variables(self, mapping):
        """Relabel the variables according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable labels
                to new ones.

        Notes:
            Acts on model in place.

        """
        try:
            new_linear = {mapping[v]: bias for v, bias in iteritems(self.linear)}
            new_quadratic = {(mapping[u], mapping[v]): bias for (u, v), bias in iteritems(self.quadratic)}
            new_adj = {mapping[u]: {mapping[v] for v in neighbours} for u, neighbours in iteritems(self.adj)}
        except KeyError as e:
            raise ValueError("no mapping for variable {}".format(e))
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        if len(new_linear) != len(self.linear):
            raise ValueError("mapping does not contain unique keys")

        self.linear = new_linear
        self.quadratic = new_quadratic
        self.adj = new_adj
