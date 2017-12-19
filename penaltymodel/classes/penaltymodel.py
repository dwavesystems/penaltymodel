"""
Specification and PenaltyModel
------------------------------

"""
from __future__ import absolute_import

from numbers import Number
import itertools

from six import itervalues, iteritems
import networkx as nx

from penaltymodel.classes.vartypes import Vartype
from penaltymodel.classes.binary_quadratic_model import BinaryQuadraticModel


__all__ = ['Specification', 'PenaltyModel']


class Specification(object):
    """Container class for the properties desired of a PenaltyModel.

    Args:
        graph (:class:`networkx.Graph`/iterable[edge]): The graph that
            defines the relation between variables in the penalty model.
            The node labels will be used as the variable labels in the
            binary quadratic model.
        decision_variables (tuple/iterable): Maps the feasible configurations
            to the graph. Must be the same length as each configuration
            in feasible_configurations. Any iterable will be case to
            a tuple.
        feasible_configurations (dict[tuple[int], number]/iterable[tuple[int]]):
            The set of feasible configurations. Each feasible configuration
            should be a tuple of variable assignments. See examples.
        vartype (:class:`.Vartype`/str/set, optional): Default :class:`.Vartype.SPIN`.
            The variable type desired for the penalty model. Accepted input values:
            :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
        linear_energy_ranges (dict[node, (number, number)], optional): If
            not provided, defaults to {v: (-2, 2), ...} for each variable v.
            Defines the energy ranges available for the linear
            biases of the penalty model.
        quadratic_energy_ranges (dict[edge, (number, number)], optional): If
            not provided, defaults to {edge: (-1, 1), ...} for each edge in
            graph. Defines the energy ranges available for the quadratic
            biases of the penalty model.

    Examples:
        >>> graph = nx.path_graph(5)
        >>> decision_variables = (0, 4)  # the ends of the path
        >>> feasible_configurations = {(-1, -1), (1, 1)}  # we want the ends of the path to agree
        >>> spec = pm.Specification(graph, decision_variables, feasible_configurations)
        >>> spec.vartype  # infers the vartype from the feasible_configurations
        <Vartype.SPIN: frozenset([1, -1])>

    Attributes:
        decision_variables (tuple): Maps the feasible configurations
            to the graph.
        feasible_configurations (dict[tuple[int], number]):
            The set of feasible configurations. The value is the (relative)
            energy of each of the feasible configurations.
        graph (:class:`networkx.Graph`): The graph that defines the relation
            between variables in the penalty model.
            The node labels will be used as the variable labels in the
            binary quadratic model.
        linear_energy_ranges (dict[node, (number, number)]):
            Defines the energy ranges available for the linear
            biases of the penalty model.
        quadratic_energy_ranges (dict[edge, (number, number)]):
            Defines the energy ranges available for the quadratic
            biases of the penalty model.
        vartype (:class:`.Vartype`): The variable type. If unknown or
            unspecified will be :class:`.Vartype.UNDEFINED`.

    """
    def __init__(self, graph, decision_variables, feasible_configurations, vartype=Vartype.SPIN,
                 linear_energy_ranges=None, quadratic_energy_ranges=None):

        #
        # graph
        #
        if not isinstance(graph, nx.Graph):
            try:
                edges = graph
                graph = nx.Graph()
                graph.add_edges_from(edges)
            except:
                TypeError("expected graph to be a networkx Graph or an iterable of edges")
        self.graph = graph

        #
        # decision_variables
        #
        try:
            if not isinstance(decision_variables, tuple):
                decision_variables = tuple(decision_variables)
        except TypeError:
            raise TypeError("expected decision_variables to be an iterable")
        if not all(v in graph for v in decision_variables):
            raise ValueError("some vars in decision decision_variables do not have a corresponding node in graph")
        self.decision_variables = decision_variables
        num_dv = len(decision_variables)

        #
        # feasible_configurations
        #
        try:
            if not isinstance(feasible_configurations, dict):
                feasible_configurations = {config: 0.0 for config in feasible_configurations}
            else:
                if not all(isinstance(en, Number) for en in itervalues(feasible_configurations)):
                    raise ValueError("the energy fo each configuration should be numeric")
        except TypeError:
            raise TypeError("expected decision_variables to be an iterable")
        if not all(len(config) == num_dv for config in feasible_configurations):
            raise ValueError("the feasible configurations should all match the length of decision_variables")
        self.feasible_configurations = feasible_configurations

        #
        # energy ranges
        #
        if linear_energy_ranges is None:
            self.linear_energy_ranges = {v: (-2, 2) for v in graph}
        elif not isinstance(linear_energy_ranges, dict):
            raise TypeError("linear_energy_ranges should be a dict")
        else:
            self.linear_energy_ranges = linear_energy_ranges

        if quadratic_energy_ranges is None:
            quadratic_energy_ranges = {edge: (-1, 1) for edge in graph.edges}
        elif not isinstance(quadratic_energy_ranges, dict):
            raise TypeError("quadratic_energy_ranges should be a dict")
        self.quadratic_energy_ranges = quadratic_energy_ranges

        # we also want quadratic energy ranges to be bi-directional
        other_direction = {}
        for (u, v), range_ in iteritems(quadratic_energy_ranges):
            if (v, u) in quadratic_energy_ranges:
                if quadratic_energy_ranges[(v, u)] == range_:
                    pass
                else:
                    raise ValueError("conflicting energy ranges for {}, {}".format((v, u), (u, v)))
            else:
                other_direction[(v, u)] = range_
        quadratic_energy_ranges.update(other_direction)

        #
        # vartype
        #
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
        # check that our feasible configurations match
        seen_variable_types = set().union(*feasible_configurations)
        if not seen_variable_types.issubset(vartype.value):
            raise ValueError(("feasible_configurations type must match vartype. "
                              "feasible_configurations have values {}, "
                              "values permitted by vartype are {}.").format(seen_variable_types, vartype.value))
        self.vartype = vartype

    def __len__(self):
        return len(self.graph)

    def __eq__(self, specification):
        """Implemented equality checking. """

        # for specification, graph is considered equal if it has the same nodes
        # and edges
        return (isinstance(specification, Specification) and
                self.graph.edges == specification.graph.edges and
                self.graph.nodes == specification.graph.nodes and
                self.decision_variables == specification.decision_variables and
                self.feasible_configurations == specification.feasible_configurations)

    def relabel_variables(self, mapping, copy=True):
        """Relabel the variables and nodes according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable/node labels
                to new ones.
            copy (bool, default): If True, return a copy of Specification
                with the variables relabeled, otherwise apply the relabeling in
                place.

        Returns:
            :class:`.Specification`: A Specification with the variables
            relabeled according to mapping. If copy=False returns itself,
            if copy=True returns a new Specification.

        """
        graph = self.graph
        linear_energy_ranges = self.linear_energy_ranges
        quadratic_energy_ranges = self.quadratic_energy_ranges

        if copy:
            return Specification(nx.relabel_nodes(graph, mapping, copy=True),  # also checks the mapping
                                 tuple(mapping.get(v, v) for v in self.decision_variables),
                                 self.feasible_configurations,  # does not change
                                 vartype=self.vartype,  # does not change
                                 linear_energy_ranges={mapping.get(v, v): linear_energy_ranges[v] for v in graph},
                                 quadratic_energy_ranges={(mapping.get(u, u), mapping.get(v, v)):
                                                          quadratic_energy_ranges[(u, v)]
                                                          for u, v in graph.edges})
        else:
            # now we need the linear_energy_ranges and quadratic_energy_ranges
            old_labels = set(mapping.keys())
            new_labels = set(mapping.values())
            shared = old_labels & new_labels

            if shared:
                # in this case we need to transform to an intermediate state
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

            # modifies graph in place
            nx.relabel_nodes(self.graph, mapping, copy=False)

            # this is always a new object
            self.decision_variables = tuple(mapping.get(v, v) for v in self.decision_variables)

            # we can just relabel in-place without worrying about conflict
            for v in old_labels:
                linear_energy_ranges[mapping.get(v, v)] = linear_energy_ranges[v]
                del linear_energy_ranges[v]

            for u, v in list(quadratic_energy_ranges):
                new_edge = (mapping.get(u, u), mapping.get(v, v))
                quadratic_energy_ranges[new_edge] = quadratic_energy_ranges[(u, v)]
                del quadratic_energy_ranges[(u, v)]

            return self


class PenaltyModel(Specification):
    """Container class for the components that make up a penalty model.

    A penalty model is a small Ising problem or QUBO that has ground
    states that match the feasible configurations and excited states
    that have a classical energy greater than the ground energy by
    at least the classical gap.

    PenaltyModel is a subclass of :class:`.Specification`.

    Args:
        graph (:class:`networkx.Graph`/iterable[edge]): The graph that
            defines the relation between variables in the penalty model.
            The node labels will be used as the variable labels in the
            binary quadratic model.
        decision_variables (tuple/iterable): Maps the feasible configurations
            to the graph. Must be the same length as each configuration
            in feasible_configurations. Any iterable will be case to
            a tuple.
        feasible_configurations (dict[tuple[int], number]/iterable[tuple[int]]):
            The set of feasible configurations. Each feasible configuration
            should be a tuple of variable assignments. See examples.
        model (:class:`.BinaryQuadraticModel`): A binary quadratic model
            that has ground states that match the feasible_configurations.
        classical_gap (numeric): The difference in classical energy between the ground
            state and the first excited state. Must be positive.
        ground_energy (numeric): The minimum energy of all possible configurations.
        linear_energy_ranges (dict[node, (number, number)], optional): If
            not provided, defaults to {v: (-2, 2), ...} for each variable v.
            Defines the energy ranges available for the linear
            biases of the penalty model.
        quadratic_energy_ranges (dict[edge, (number, number)], optional): If
            not provided, defaults to {edge: (-1, 1), ...} for each edge in
            graph. Defines the energy ranges available for the quadratic
            biases of the penalty model.
        vartype (:class:`.Vartype`, optional): The variable type. If not
            provided, tried to infer the vartype from the feasible_configurations.
            If Specification cannot determine the vartype then set to
            :class:`.Vartype.UNDEFINED`.

    Examples:
        The penalty model can be created from its component parts:

        >>> graph = nx.path_graph(3)
        >>> decision_variables = (0, 2)  # the ends of the path
        >>> feasible_configurations = {(-1, -1), (1, 1)}  # we want the ends of the path to agree
        >>> model = pm.BinaryQuadraticModel({0: 0, 1: 0, 2: 0}, {(0, 1): -1, (1, 2): -1}, 0.0, pm.SPIN)
        >>> classical_gap = 2.0
        >>> ground_energy = -2.0
        >>> widget = pm.PenaltyModel(graph, decision_variables, feasible_configurations,
        ...                          model, classical_gap, ground_energy)

        Or it can be created from a specification:

        >>> spec = pm.Specification(graph, decision_variables, feasible_configurations)
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
        linear_energy_ranges (dict[node, (number, number)]):
            Defines the energy ranges available for the linear
            biases of the penalty model.
        model (:class:`.BinaryQuadraticModel`): A binary quadratic model
            that has ground states that match the feasible_configurations.
        quadratic_energy_ranges (dict[edge, (number, number)]):
            Defines the energy ranges available for the quadratic
            biases of the penalty model.
        vartype (:class:`.Vartype`): The variable type. If unknown or
            unspecified will be :class:`.Vartype.UNDEFINED`.

    """
    def __init__(self, graph, decision_variables, feasible_configurations,
                 model, classical_gap, ground_energy,
                 vartype=Vartype.SPIN,
                 linear_energy_ranges=None, quadratic_energy_ranges=None):

        Specification.__init__(self, graph, decision_variables, feasible_configurations,
                               vartype=vartype,
                               linear_energy_ranges=linear_energy_ranges,
                               quadratic_energy_ranges=quadratic_energy_ranges)

        if self.vartype != model.vartype:
            model = model.change_vartype(self.vartype)

        if not isinstance(model, BinaryQuadraticModel):
            raise TypeError("expected 'model' to be a BinaryQuadraticModel")
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
            model (:class:`.BinaryQuadraticModel`): A binary quadratic model
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
                   model,
                   classical_gap,
                   ground_energy,
                   linear_energy_ranges=specification.linear_energy_ranges,
                   quadratic_energy_ranges=specification.quadratic_energy_ranges,
                   vartype=specification.vartype)

    def __eq__(self, penalty_model):
        # other values are derived
        return (isinstance(penalty_model, PenaltyModel) and
                Specification.__eq__(self, penalty_model) and
                self.model == penalty_model.model)

    def relabel_variables(self, mapping, copy=True):
        """Relabel the variables and nodes according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable/node labels
                to new ones.
            copy (bool, default): If True, return a copy of PenaltyModel
                with the variables relabeled, otherwise apply the relabeling in
                place.

        Returns:
            :class:`.PenaltyModel`: A PenaltyModel with the variables
            relabeled according to mapping. If copy=False returns itself,
            if copy=True returns a new PenaltyModel.

        """
        # just use the relabeling of each component
        Specification.relabel_variables(self, mapping, copy=copy)
        self.model.relabel_variables(mapping, copy=copy)
