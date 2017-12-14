"""
Specification and PenaltyModel
------------------------------

"""
from __future__ import absolute_import

from numbers import Number
from collections import defaultdict

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
    def __init__(self, graph, decision_variables, feasible_configurations,
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 vartype=None):

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
            self.linear_energy_ranges = defaultdict(lambda: (-2., 2.))
        elif not isinstance(linear_energy_ranges, dict):
            raise TypeError("linear_energy_ranges should be a dict")
        else:
            self.linear_energy_ranges = linear_energy_ranges
        if quadratic_energy_ranges is None:
            self.quadratic_energy_ranges = defaultdict(lambda: (-1., 1.))
        elif not isinstance(quadratic_energy_ranges, dict):
            raise TypeError("quadratic_energy_ranges should be a dict")
        else:
            self.quadratic_energy_ranges = quadratic_energy_ranges

        #
        # vartype
        #

        # see what we can determine from the feasible_configurations
        seen_variable_types = set().union(*feasible_configurations)
        if vartype is None or vartype is Vartype.UNDEFINED:
            # the vartype is not provided or is undefined, so see if we can determine from
            # the feasible_configurations input
            if len(seen_variable_types) >= 2:
                try:
                    vartype = Vartype(seen_variable_types)
                except ValueError:
                    vartype = Vartype.UNDEFINED
            elif not seen_variable_types:
                vartype = Vartype.UNDEFINED
            else:
                candidate_vartypes = [vt for vt in Vartype if vt.value is not None and seen_variable_types.issubset(vt.value)]
                if len(candidate_vartypes) == 1:
                    vartype, = candidate_vartypes
                else:
                    vartype = Vartype.UNDEFINED

        else:
            # the vartype has been specified, so check that it matches the given inputs
            if not isinstance(vartype, Vartype):
                # try to cast to vartype
                try:
                    if isinstance(vartype, str):
                        vartype = Vartype[vartype]
                    else:
                        vartype = Vartype(vartype)
                except (ValueError, KeyError):
                    raise TypeError("unexpected `vartype`. See BinaryQuadraticModel.Vartype for known types.")
            # check the values
            if not seen_variable_types.issubset(vartype.value):
                            raise ValueError(("the variable types in feasible_configurations ({}) "
                                              "are not of type {}").format(feasible_configurations, vartype))
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
                 linear_energy_ranges=None, quadratic_energy_ranges=None,
                 vartype=None):

        if vartype is not None:
            if vartype != model.vartype:
                raise ValueError("mismatched vartype")
        else:
            vartype = model.vartype

        Specification.__init__(self, graph, decision_variables, feasible_configurations,
                               linear_energy_ranges, quadratic_energy_ranges, vartype)

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
