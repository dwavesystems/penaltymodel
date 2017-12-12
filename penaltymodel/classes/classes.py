"""

"""

from collections import defaultdict

from six import itervalues, iteritems
import networkx as nx

from penaltymodel.serialization import *

__all__ = ['BinaryQuadraticModel', 'PenaltyModel', 'Specification']


class Specification(object):
    """Specifies that properties desired of the PenaltyModel.

    Args:
        TODO

    Parameters:
        TODO

    """
    def __init__(self, graph, decision_variables, feasible_configurations,
                 linear_energy_ranges=None, quadratic_energy_ranges=None):

        self.graph = graph
        self.decision_variables = decision_variables
        self.feasible_configurations = feasible_configurations

        self.linear_energy_ranges = linear_energy_ranges
        self.quadratic_energy_ranges = quadratic_energy_ranges

        # need to check correctness

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("expected graph to be a networkx Graph object")
        self._graph = graph

    @property
    def linear_energy_ranges(self):
        return self._linear_ranges

    @linear_energy_ranges.setter
    def linear_energy_ranges(self, linear_energy_ranges):
        if linear_energy_ranges is None:
            self._linear_ranges = defaultdict(lambda: (-2., 2.))
        elif not isinstance(linear_energy_ranges, dict):
            raise TyperError("linear_energy_ranges should be a dict")
        else:
            self._linear_ranges = linear_energy_ranges

    @property
    def quadratic_energy_ranges(self):
        return self._quadratic_ranges

    @quadratic_energy_ranges.setter
    def quadratic_energy_ranges(self, quadratic_energy_ranges):
        if quadratic_energy_ranges is None:
            self._quadratic_ranges = defaultdict(lambda: (-1., 1.))
        elif not isinstance(quadratic_energy_ranges, dict):
            raise TyperError("quadratic_energy_ranges should be a dict")
        else:
            self._quadratic_ranges = quadratic_energy_ranges

    def serialize(self, nodelist=None, edgelist=None):
        """TODO: dump to dict, each object in dict must be serializable."""

        serial = {}

        if nodelist is None or edgelist is None:
            graph = self.graph
            if not all(isinstance(v, int) for v in graph):
                raise NotImplementedError("cannot currently serialize arbitrarily named variables.")
            nodelist = sorted(graph.nodes)
            edgelist = sorted(sorted(edge) for edge in graph.edges)
        serial['num_nodes'], serial['num_edges'], serial['edges'] = serialize_graph(nodelist, edgelist)

        # next config
        serial['num_variables'], serial['num_feasible_configurations'], serial['feasible_configurations'], serial['energies'] = serialize_configurations(self.feasible_configurations)

        # decision variables
        serial['decision_variables'] = serialize_decision_variables(self.decision_variables)

        # encode the energy ranges
        serial['min_quadratic_bias'] = min(self.quadratic_energy_ranges[edge][0] for edge in self.graph.edges)
        serial['max_quadratic_bias'] = max(self.quadratic_energy_ranges[edge][1] for edge in self.graph.edges)
        serial['min_linear_bias'] = min(self.linear_energy_ranges[v][0] for v in self.graph)
        serial['max_linear_bias'] = max(self.linear_energy_ranges[v][1] for v in self.graph)

        return serial

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

    def serialize(self):

        # graph first
        graph = self.graph
        if not all(isinstance(v, int) for v in graph):
            raise NotImplementedError("cannot currently serialize arbitrarily named variables.")
        nodelist = sorted(graph.nodes)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        serial = Specification.serialize(self, nodelist, edgelist)

        # add the model, this overwrites min_linear_bias, max_quadratic_bias, etc
        serial.update(self.model.serialize(nodelist, edgelist))

        # finally the gap and ground energy
        serial['classical_gap'] = self.classical_gap
        serial['ground_energy'] = self.ground_energy

        return serial

    def __eq__(self, penalty_model):

        # other values are derived
        return (isinstance(penalty_model, PenaltyModel) and
                Specification.__eq__(self, penalty_model) and
                self.model == penalty_model.model)
