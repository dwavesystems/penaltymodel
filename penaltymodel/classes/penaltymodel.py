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


__all__ = ['Specification', 'PenaltyModel']


class Specification(object):
    """Container class for the properties desired of a PenaltyModel.

    Args:
        graph (:class:`networkx.Graph`/iterable[edge]): The graph that
            defines the relation between variables in the penaltymodel.
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
        graph (:class:`networkx.Graph`): The graph that defines the relation
            between variables in the penaltymodel.
            The node labels will be used as the variable labels in the
            binary quadratic model.
        decision_variables (tuple): Maps the feasible configurations
            to the graph.
        feasible_configurations (dict[tuple[int], number]):
            The set of feasible configurations. The value is the (relative)
            energy of each of the feasible configurations.
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
    def __init__(self, specification, model, classical_gap, ground_energy):

        # there might be a more clever way to do this but this will work
        # for now.
        self.graph = specification.graph
        self.decision_variables = specification.decision_variables
        self.feasible_configurations = specification.feasible_configurations
        self.linear_energy_ranges = specification.linear_energy_ranges
        self.quadratic_energy_ranges = specification.quadratic_energy_ranges

        if not isinstance(model, BinaryQuadraticModel):
            raise TypeError("expected model to be a Model")
        self.model = model

        self.classical_gap = classical_gap
        self.ground_energy = ground_energy

    def __eq__(self, penalty_model):

        # other values are derived
        return (isinstance(penalty_model, PenaltyModel) and
                Specification.__eq__(self, penalty_model) and
                self.model == penalty_model.model)
