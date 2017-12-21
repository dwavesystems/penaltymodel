"""
BinaryQuadraticModel
--------------------
"""
from __future__ import absolute_import

import itertools

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
        vartype (:class:`.Vartype`/str/set, optional): Default :class:`.Vartype.SPIN`.
            The variable type desired for the penalty model. Accepted input values:
            :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

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
            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError
        except (ValueError, KeyError):
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))
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
                        self._quadratic_equality(self.quadratic, self.quadratic),
                        self.offset == model.offset])
        else:
            # different vartypes are not equal
            return False

    @staticmethod
    def _quadratic_equality(quadratic0, quadratic1):
        for (u, v), bias in iteritems(quadratic0):
            if (u, v) in quadratic1:
                if quadratic1[(u, v)] != bias:
                    return False
            elif (v, u) in quadratic1:
                if quadratic1[(u, v)] != bias:
                    return False
            else:
                return False
        return True

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

        return self._binary_to_spin()

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

        linear, quadratic, offset = self._spin_to_binary()

        quadratic.update({(v, v): bias for v, bias in iteritems(linear)})

        return quadratic, offset

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

    def relabel_variables(self, mapping, copy=True):
        """Relabel the variables according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable labels
                to new ones. If an incomplete mapping is provided,
                variables will keep their labels
            copy (bool, default): If True, return a copy of BinaryQuadraticModel
                with the variables relabeled, otherwise apply the relabeling in
                place.

        Returns:
            :class:`.BinaryQuadraticModel`: A BinaryQuadraticModel with the
            variables relabelled. If copy=False, returns itself.

        Examples:
            >>> model = pm.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=pm.SPIN)
            >>> new_model = model.relabel_variables({0: 'a'})
            >>> new_model.quadratic
            {('a', 1): -1}
            >>> new_model = model.relabel_variables({0: 'a', 1: 'b'}, copy=False)
            >>> model.quadratic
            {('a', 'b'): -1}
            >>> new_model is model
            True

        """
        try:
            old_labels = set(mapping.keys())
            new_labels = set(mapping.values())
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        if copy:
            return BinaryQuadraticModel({mapping.get(v, v): bias for v, bias in iteritems(self.linear)},
                                        {(mapping.get(u, u), mapping.get(v, v)): bias
                                         for (u, v), bias in iteritems(self.quadratic)},
                                        self.offset, self.vartype)
        else:
            shared = old_labels & new_labels
            if shared:
                # in this case relabel to a new intermediate labeling, then map from the intermediate
                # labeling to the desired labeling

                # counter will be used to generate the intermediate labels, as an easy optimization
                # we start the counter with a high number because often variables are labeled by
                # integers starting from 0
                counter = itertools.count(2 * len(self))

                old_to_intermediate = {}
                intermediate_to_new = {}

                for old, new in iteritems(mapping):
                    if old == new:
                        # we can remove self-labels
                        continue

                    if old in new_labels or new in old_labels:

                        # try to get a new unique label
                        lbl = next(counter)
                        while lbl in new_labels or lbl in old_labels:
                            lbl = next(counter)

                        # add it to the mapping
                        old_to_intermediate[old] = lbl
                        intermediate_to_new[lbl] = new

                    else:
                        old_to_intermediate[old] = new
                        # don't need to add it to intermediate_to_new because it is a self-label

                self.relabel_variables(old_to_intermediate, copy=False)
                self.relabel_variables(intermediate_to_new, copy=False)
                return self

            linear = self.linear
            quadratic = self.quadratic
            adj = self.adj

            # rebuild linear and adj with the new labels
            for old in list(linear):
                if old not in mapping:
                    continue

                new = mapping[old]

                # acting on all of these in-place
                linear[new] = linear[old]
                adj[new] = adj[old]
                for u in adj[old]:
                    adj[u][new] = adj[u][old]
                    del adj[u][old]

                del linear[old]
                del adj[old]

            # now rebuild quadratic
            for old_u, old_v in list(quadratic):
                if old_u not in mapping:
                    if old_v not in mapping:
                        # no remap needed
                        continue
                    new_u = old_u
                else:
                    new_u = mapping[old_u]
                if old_v not in mapping:
                    new_v = old_v
                else:
                    new_v = mapping[old_v]

                if (old_v, old_u) in quadratic:
                    quadratic[(new_v, new_u)] = quadratic[(old_v, old_u)]
                    del quadratic[(old_v, old_u)]
                elif (old_u, old_v) in quadratic:
                    quadratic[(new_u, new_v)] = quadratic[(old_u, old_v)]
                    del quadratic[(old_u, old_v)]
                else:
                    raise RuntimeError("something went wrong in relabel")

            return self

    def copy(self):
        """Create a copy of the BinaryQuadraticModel.

        Returns:
            :class:`.BinaryQuadraticModel`

        """
        return BinaryQuadraticModel(self.linear.copy(), self.quadratic.copy(), self.offset, vartype=self.vartype)

    def change_vartype(self, vartype):
        """Creates a new BinaryQuadraticModel with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional): Default :class:`.Vartype.SPIN`.
                The variable type desired for the penalty model. Accepted input values:
                :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        Returns:
            :class:`.BinaryQuadraticModel`. A new BinaryQuadraticModel with
                vartype matching input 'vartype'.

        """
        try:
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)
            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError
        except (ValueError, KeyError):
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))

        # vartype matches so we are done
        if vartype is self.vartype:
            return self.copy()

        if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
            linear, quadratic, offset = self._spin_to_binary()
            return BinaryQuadraticModel(linear, quadratic, offset, vartype=Vartype.BINARY)
        elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
            linear, quadratic, offset = self._binary_to_spin()
            return BinaryQuadraticModel(linear, quadratic, offset, vartype=Vartype.SPIN)
        else:
            raise RuntimeError("something has gone wrong. unknown vartype conversion.")  # pragma: no cover

    def _spin_to_binary(self):
        """convert linear, quadratic, and offset from spin to binary.
        Does no checking of vartype. Copies all of the values into new objects.
        """
        linear = self.linear
        quadratic = self.quadratic

        # the linear biases are the easiest
        new_linear = {v: 2. * bias for v, bias in iteritems(linear)}

        # next the quadratic biases
        new_quadratic = {}
        for (u, v), bias in iteritems(quadratic):
            if bias == 0.0:
                continue
            new_quadratic[(u, v)] = 4. * bias
            new_linear[u] -= 2. * bias
            new_linear[v] -= 2. * bias

        # finally calculate the offset
        offset = self.offset + sum(itervalues(quadratic)) - sum(itervalues(linear))

        return new_linear, new_quadratic, offset

    def _binary_to_spin(self):
        """convert linear, quadratic and offset from binary to spin.
        Does no checking of vartype. Copies all of the values into new objects.
        """
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

    def to_networkx_graph(self, node_attribute_name='bias', edge_attribute_name='bias'):
        """Return the BinaryQuadraticModel as a NetworkX graph.

        Args:
            node_attribute_name (hashable): The attribute name for the
                linear biases.
            edge_attribute_name (hashable): The attribute name for the
                quadratic biases.

        Returns:
            :class:`networkx.Graph`: A NetworkX with the biases stored as
                node/edge attributes.

        Examples:
            >>> import networkx as nx
            >>>

        """
        import networkx as nx

        BQM = nx.Graph()

        # add the linear biases
        BQM.add_nodes_from(((v, {node_attribute_name: bias, 'vartype': self.vartype})
                            for v, bias in iteritems(self.linear)))

        # add the quadratic biases
        BQM.add_edges_from(((u, v, {edge_attribute_name: bias}) for (u, v), bias in iteritems(self.quadratic)))

        # set the offset
        BQM.offset = self.offset

        return BQM