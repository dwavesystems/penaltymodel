# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PenaltyModel
------------
"""
from __future__ import absolute_import

from numbers import Number

import itertools
import networkx as nx
import numpy as np
from scipy.optimize import linprog
from six import iteritems

from dimod import BinaryQuadraticModel, Vartype
import dimod
from penaltymodel.core.classes.specification import Specification


__all__ = ['PenaltyModel']


class PenaltyModel(Specification):
    """Container class for the components that make up a penalty model.

    A penalty model is a small Ising problem or QUBO that has ground
    states that match the feasible configurations and excited states
    that have a classical energy greater than the ground energy by
    at least the classical gap.

    PenaltyModel is a subclass of :class:`.Specification`.

    Args:
        graph (:class:`networkx.Graph`/iterable[edge]):
            Defines the structure of the desired binary quadratic model. Each
            node in the graph represents a variable and each edge defines an
            interaction between two variables.
            If given as an iterable of edges, the graph will be constructed
            by adding each edge to an (initially) empty graph.

        decision_variables (iterable):
            The labels of the penalty model's decision variables. Each variable label
            in `decision_variables` must correspond to a node in `graph`.
            Should be an ordered iterable of hashable labels.

        feasible_configurations (dict[tuple[int], number]/iterable[tuple[int]]):
            The set of feasible configurations. Defines the allowed configurations
            of the decision variables allowed by the constraint.
            Each feasible configuration should be a tuple, each element of which
            must be of a value matching `vartype`. If given as a dict, the key
            is the feasible configuration and the value is the desired relative
            energy. If given as an iterable, it will be case to a dict where
            the relative energies are all 0.

        vartype (:class:`.Vartype`/str/set):
            The variable type desired for the penalty model.
            Accepted input values:
            :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        model (:class:`dimod.BinaryQuadraticModel`): A binary quadratic model
            that has ground states that match the feasible_configurations.

        classical_gap (numeric): The difference in classical energy between the ground
            state and the first excited state. Must be positive.

        ground_energy (numeric): The minimum energy of all possible configurations.

        ising_linear_ranges (dict[node, [number, number]], optional, default=None):
            When the penalty model is spin-valued, specifies the allowed range
            for each of the linear biases.
            If a dict, should be of the form {v: [min, max], ...} where v is
            a variable in the desired penalty model and (min, max) defines
            the acceptable range for the linear bias associated with v.
            If None, the default will be set to {v: [-1, 1], ...} for each
            v in graph.
            A partial assignment is allowed.

        ising_quadratic_ranges (dict[node, dict[node, [number, number]], optional, default=None):
            When the penalty model is spin-valued, specifies the allowed range
            for each of the quadratic biases.
            If a dict, should be of the form {v: {u: [min, max], ...}, ...} where
            u and v are variables in the desired penalty model and u, v have an
            interaction - there is an edge between nodes u, v in `graph`. (min, max)
            the acceptable range for the quadratic bias associated with u, v.
            If None, the default will be set to
            {v: {u: [min, max], ...}, u: {v: [min, max], ...}, ...} for each
            edge u, v in graph.
            A partial assignment is allowed.

    Examples:
        The penalty model can be created from its component parts:

        >>> import networkx as nx
        >>> import dimod
        >>> graph = nx.path_graph(3)
        >>> decision_variables = (0, 2)  # the ends of the path
        >>> feasible_configurations = {(-1, -1), (1, 1)}  # we want the ends of the path to agree
        >>> model = dimod.BinaryQuadraticModel({0: 0, 1: 0, 2: 0}, {(0, 1): -1, (1, 2): -1}, 0.0, dimod.SPIN)
        >>> classical_gap = 2.0
        >>> ground_energy = -2.0
        >>> widget = pm.PenaltyModel(graph, decision_variables, feasible_configurations, dimod.SPIN,
        ...                          model, classical_gap, ground_energy)

        Or it can be created from a specification:

        >>> spec = pm.Specification(graph, decision_variables, feasible_configurations, dimod.SPIN)
        >>> widget = pm.PenaltyModel.from_specification(spec, model, classical_gap, ground_energy)

    Attributes:
        decision_variables (tuple): Maps the feasible configurations
            to the graph.
        classical_gap (numeric): The difference in classical energy between the ground
            state and the first excited state. Must be positive.
        feasible_configurations (dict[tuple[int], number]):
            The set of feasible configurations. The value is the (relative)
            energy of each of the feasible configurations.
        graph (:class:`networkx.Graph`): The graph that defines the relation
            between variables in the penaltymodel.
            The node labels will be used as the variable labels in the
            binary quadratic model.
        ground_energy (numeric): The minimum energy of all possible configurations.
        ising_linear_ranges (dict[node, (number, number)]):
            Defines the energy ranges available for the linear
            biases of the penalty model.
        model (:class:`dimod.BinaryQuadraticModel`): A binary quadratic model
            that has ground states that match the feasible_configurations.
        ising_quadratic_ranges (dict[edge, (number, number)]):
            Defines the energy ranges available for the quadratic
            biases of the penalty model.
        vartype (:class:`dimod.Vartype`): The variable type.

    """
    def __init__(self, graph, decision_variables, feasible_configurations, vartype,
                 model, classical_gap, ground_energy,
                 ising_linear_ranges=None, ising_quadratic_ranges=None):

        Specification.__init__(self, graph, decision_variables, feasible_configurations,
                               vartype=vartype,
                               min_classical_gap=classical_gap,
                               ising_linear_ranges=ising_linear_ranges,
                               ising_quadratic_ranges=ising_quadratic_ranges)

        if self.vartype != model.vartype:
            model = model.change_vartype(self.vartype)

        # check the energy ranges
        ising_linear_ranges = self.ising_linear_ranges
        ising_quadratic_ranges = self.ising_quadratic_ranges
        if self.vartype is Vartype.SPIN:
            # check the ising energy ranges
            for v, bias in iteritems(model.linear):
                min_, max_ = ising_linear_ranges[v]
                if bias < min_ or bias > max_:
                    raise ValueError(("variable {} has bias {} outside of the specified range [{}, {}]"
                                      ).format(v, bias, min_, max_))
            for (u, v), bias in iteritems(model.quadratic):
                min_, max_ = ising_quadratic_ranges[u][v]
                if bias < min_ or bias > max_:
                    raise ValueError(("interaction {}, {} has bias {} outside of the specified range [{}, {}]"
                                      ).format(u, v, bias, min_, max_))

        if not isinstance(model, BinaryQuadraticModel):
            raise TypeError("expected 'model' to be a BinaryQuadraticModel")
        if set(model.variables).symmetric_difference(graph.nodes):
            raise ValueError("model labels must match graph node labels")
        self.model = model

        if not isinstance(classical_gap, Number):
            raise TypeError("expected classical_gap to be numeric")
        if classical_gap <= 0.0:
            raise ValueError("classical_gap must be positive")
        self.classical_gap = classical_gap

        if not isinstance(ground_energy, Number):
            raise TypeError("expected ground_energy to be numeric")
        self.ground_energy = ground_energy

    @classmethod
    def from_specification(cls, specification, model, classical_gap, ground_energy):
        """Construct a PenaltyModel from a Specification.

        Args:
            specification (:class:`.Specification`): A specification that was used
                to generate the model.
            model (:class:`dimod.BinaryQuadraticModel`): A binary quadratic model
                that has ground states that match the feasible_configurations.
            classical_gap (numeric): The difference in classical energy between the ground
                state and the first excited state. Must be positive.
            ground_energy (numeric): The minimum energy of all possible configurations.

        Returns:
            :class:`.PenaltyModel`

        """

        # Author note: there might be a way that avoids rechecking all of the values without
        # side-effects or lots of repeated code, but this seems simpler and more explicit
        return cls(specification.graph,
                   specification.decision_variables,
                   specification.feasible_configurations,
                   specification.vartype,
                   model,
                   classical_gap,
                   ground_energy,
                   ising_linear_ranges=specification.ising_linear_ranges,
                   ising_quadratic_ranges=specification.ising_quadratic_ranges)

    def __eq__(self, penalty_model):
        # other values are derived
        return (isinstance(penalty_model, PenaltyModel) and
                Specification.__eq__(self, penalty_model) and
                self.model == penalty_model.model)

    def __ne__(self, penalty_model):
        return not self.__eq__(penalty_model)

    def relabel_variables(self, mapping, inplace=True):
        """Relabel the variables and nodes according to the given mapping.

        Args:
            mapping (dict[hashable, hashable]): A dict with the current
                variable labels as keys and new labels as values. A
                partial mapping is allowed.

            inplace (bool, optional, default=True):
                If True, the penalty model is updated in-place; otherwise, a new penalty model
                is returned.

        Returns:
            :class:`.PenaltyModel`: A PenaltyModel with the variables relabeled according to
            mapping.

        Examples:
            >>> spec = pm.Specification(nx.path_graph(3), (0, 2), {(-1, -1), (1, 1)}, dimod.SPIN)
            >>> model = dimod.BinaryQuadraticModel({0: 0, 1: 0, 2: 0}, {(0, 1): -1, (1, 2): -1}, 0.0, dimod.SPIN)
            >>> penalty_model = pm.PenaltyModel.from_specification(spec, model, 2., -2.)
            >>> relabeled_penalty_model = penalty_model.relabel_variables({0: 'a'}, inplace=False)
            >>> relabeled_penalty_model.decision_variables
            ('a', 2)

            >>> spec = pm.Specification(nx.path_graph(3), (0, 2), {(-1, -1), (1, 1)}, dimod.SPIN)
            >>> model = dimod.BinaryQuadraticModel({0: 0, 1: 0, 2: 0}, {(0, 1): -1, (1, 2): -1}, 0.0, dimod.SPIN)
            >>> penalty_model = pm.PenaltyModel.from_specification(spec, model, 2., -2.)
            >>> __ = penalty_model.relabel_variables({0: 'a'}, inplace=True)
            >>> penalty_model.decision_variables
            ('a', 2)

        """
        # just use the relabeling of each component
        if inplace:
            Specification.relabel_variables(self, mapping, inplace=True)
            self.model.relabel_variables(mapping, inplace=True)
            return self
        else:
            spec = Specification.relabel_variables(self, mapping, inplace=False)
            model = self.model.relabel_variables(mapping, inplace=False)
            return PenaltyModel.from_specification(spec, model, self.classical_gap, self.ground_energy)

    def _get_lp_matrix(self, spin_states, nodes, edges, offset_weight, gap_weight):
        """Creates an linear programming matrix based on the spin states, graph, and scalars provided.
        LP matrix:
            [spin_states, corresponding states of edges, offset_weight, gap_weight]

        Args:
            spin_states: Numpy array of spin states
            nodes: Iterable
            edges: Iterable of tuples
            offset_weight: Numpy 1-D array or number
            gap_weight: Numpy 1-D array or a number
        """
        if len(spin_states) == 0:
            return None

        # Set up an empty matrix
        n_states = len(spin_states)
        m_linear = len(nodes)
        m_quadratic = len(edges)
        matrix = np.empty((n_states, m_linear + m_quadratic + 2))  # +2 columns for offset and gap

        # Populate linear terms (i.e. spin states)
        if spin_states.ndim == 1:
            spin_states = np.expand_dims(spin_states, 1)
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

    def balance_penaltymodel(self, n_tries=100):
        #TODO: Do I want to edit in QUBO? Or should I just translate it all to Ising
        #TODO: Assume I'm only getting Ising for now (assuming order of method operations)
        #TODO: convert state matrix to use ints rather than floats
        #TODO: what about empty BQM?
        #TODO: could probably put the matrix construction in its own function
        if not self.model:
            raise ValueError("There is no model to balance")

        # Set up
        bqm = self.model
        m_linear = len(bqm.linear)
        m_quadratic = len(bqm.quadratic)
        labels = list(bqm.linear.keys()) + list(bqm.quadratic.keys())
        indices = {k: i for i, k in enumerate(labels)}  # map labels to column index

        # Construct the states matrix
        # Construct linear portion of states matrix
        states = np.empty((2**m_linear, m_linear + m_quadratic + 2))  # +2 for offset and gap cols
        states[:, :m_linear] = np.array([list(x) for x in
                                         itertools.product({-1, 1}, repeat=m_linear)])
        states[:, -2] = 1       # column for offset
        states[:, -1] = -1      # column for gap

        # Construct quadratic portion of states matrix
        for node0, node1 in bqm.quadratic.keys():
            edge_ind = indices[(node0, node1)]
            node0_ind = indices[node0]
            node1_ind = indices[node1]
            states[:, edge_ind] = states[:, node0_ind] * states[:, node1_ind]

        # Construct biases and energy vectors
        biases = [bqm.linear[label] for label in labels[:m_linear]]
        biases += [bqm.quadratic[label] for label in labels[m_linear:]]
        biases += [bqm.offset]
        biases = np.array(biases)
        energy = np.matmul(states[:, :-1], biases)  # Ignore last column; gap column

        # Group states by threshold
        excited_states = states[energy > self.ground_energy]
        feasible_states = states[energy <= self.ground_energy]

        # Check for balance
        if len(feasible_states) == len(self.feasible_configurations):
            return

        # Cost function
        cost_weights = np.zeros((1, states.shape[1]))
        cost_weights[0, -1] = -1  # Only interested in maximizing the gap

        # Note: Since ising has {-1, 1}, the largest possible gap is [-largest_bias, largest_bias],
        #   hence that 2 * sum(largest_biases)
        #TODO remove default hardcoded bounds
        bounds = [self.ising_linear_ranges.get(label, (-2, 2)) for label in labels[:m_linear]]
        bounds += [self.ising_quadratic_ranges.get(label, (-1, 1)) for label in labels[m_linear:]]
        max_gap = 2 * sum(max(abs(lbound), abs(ubound)) for lbound, ubound in bounds)
        bounds.append((None, None))  # Bound for offset
        bounds.append((0, max_gap))  # Bound for gap.

        # Determine duplicate decision states
        # Note: we are forming a new matrix, decision_cols, which is made up of the decision
        #   variable columns. We use decision_cols to bin like-feasible_states together (i.e. same
        #   decision state values, potentially different aux values).
        # Note2: using lexsort so that each row of decision_cols is treated as a single object with
        #   primary, secondary, tertiary, etc key orders
        # Note3: bins contains the index of the last item in each bin; these are the bin boundaries
        decision_indices = [indices[label] for label in self.decision_variables]
        decision_cols = feasible_states[:, decision_indices]
        sorted_indices = np.lexsort(decision_cols.T)
        decision_cols = decision_cols[sorted_indices, :]
        feasible_states = feasible_states[sorted_indices, :]
        bins = (decision_cols[:-1, :] != decision_cols[1:, :]).any(axis=1)
        bins = np.append(bins, True)   # Marking the end of the last bin
        bins = np.nonzero(bins)[0]

        # Get number of unique decision states and number of items in each bin
        n_uniques = bins.shape[0]
        bin_count = np.hstack((bins[0] + 1, bins[1:] - bins[:-1]))  # +1 to account for zero-index

        # Store solution with largest gap
        best_gap = 0
        best_result = None
        for _ in range(n_tries):
            # Generate random indices such that there is one index picked from each bin
            random_indices = np.random.rand(n_uniques) * bin_count
            random_indices = np.floor(random_indices).astype(np.int)
            random_indices[1:] += (bins[:-1] + 1)   # add bin offsets; +1 to negate bins' zero-index
            is_unique = np.zeros(feasible_states.shape[0], dtype=int)
            is_unique[random_indices] = 1

            # Select which feasible states are unique
            # Note: unique states do not have the 'gap' term in their linear equation, but duplicate
            #   states do. Hence the 0 for unique states' gap column and -1 for that of duplicates.
            feasible_states[is_unique==1, -1] = 0     # unique states' gap column
            feasible_states[is_unique==0, -1] = -1    # duplicate states' gap column
            unique_feasible_states = feasible_states[is_unique==1]
            duplicate_feasible_states = feasible_states[is_unique==0]

            # Returns a Scipy OptimizeResult
            new_excited_states = -np.vstack((excited_states, duplicate_feasible_states))
            result = linprog(cost_weights.flatten(),
                             A_eq=unique_feasible_states,
                             b_eq=np.zeros((unique_feasible_states.shape[0], 1)),
                             A_ub=new_excited_states,
                             b_ub=np.zeros((new_excited_states.shape[0], 1)),
                             bounds=bounds,
                             method="simplex")

            if not result.success:
                continue

            # Store best result
            gap = result.x[-1]
            if gap > best_gap:
                best_result = result
                best_gap = gap

        # Parse result
        weights = best_result.x
        h = weights[:m_linear]
        j = weights[m_linear:-2]
        offset = weights[-2]
        gap = weights[-1]

        #TODO: Test that gap meets user's gap requirements
        if gap <= 0:
            raise ValueError('Unable to balance this penaltymodel, hence no changes will be made.')

        # Create BQM
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        bqm.add_variables_from((v, bias) for v, bias in zip(labels[:m_linear], h))
        bqm.add_interactions_from((u, v, bias) for (u, v), bias in zip(labels[m_linear:], j))
        bqm.add_offset(offset)

        self.model = bqm