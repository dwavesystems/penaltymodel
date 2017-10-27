import json

import networkx as nx


def serialize_graph(graph):
    """Converts a graph into a 3-tuple of serializable objects.

    Args:
        graph: an integer-labeled networkx Graph.

    Returns:
        (int, int, str): A 3-tuple of the form (num_nodes, num_edges, edges_string).
            edges_string is a json-encoded edgelist. Each edge in the list is sorted
            and the list of edges is sorted.

    """
    num_nodes = len(graph)

    # we need to be sure that the edges are always in the same form, because the nx.Graph
    # does not keep them ordered. Luckily they are index labeled, so they are orderable
    # in Python3
    edges = sorted(sorted(edge) for edge in graph.edges)

    num_edges = len(edges)

    edges_str = json.dumps(edges, separators=(',', ':'))

    return num_nodes, num_edges, edges_str


def decode_graph(num_nodes, num_edges, edges_string):
    """Inverse of serialize_graph.

    Args:
        num_nodes: The number of nodes in the graph.
        num_edges: The number of edges in the graph.
        edges_string: A json-encoded edge list.

    Returns:
        nx.Graph: a networkx graph.

    """
    # num_edges is not used, but it is kept as an input for consistancy.
    graph = nx.Graph()
    graph.add_edges_from(json.loads(edges_string))
    graph.add_nodes_from(range(num_nodes))
    return graph


def _serialize_config(config):
    """Turns a config into an integer treating each of the variables as spins.

    Examples:
        >>> _serialize_config((0, 0, 1))
        1
        >>> _serialize_config((1, 1))
        3
        >>> _serialize_config((1, 0, 0))
        4

    """
    out = 0
    for bit in config:
        out = (out << 1) | (bit > 0)

    return out


def _decode_config(c, num_variables):
    """inverse of _serialize_config, always converts to spin."""
    def bits(c):
        n = 2 ** (num_variables - 1)
        for __ in range(num_variables):
            yield 1 if c & n else -1
            n >>= 1
    return tuple(bits(c))


def serialize_configurations(configurations):

    num_variables = len(next(iter(configurations)))
    num_configurations = len(configurations)

    if isinstance(configurations, dict) and any(configurations.values()):
        encoded = {_serialize_config(config): en for config, en in configurations.items()}

        configs, energies = zip(*sorted(encoded.items()))
        configurations_str = json.dumps(configs, separators=(',', ':'))
        energies_str = json.dumps(energies, separators=(',', ':'))
    else:
        configurations_str = json.dumps(sorted(_serialize_config(config)
                                               for config in configurations),
                                        separators=(',', ':'))
        energies_str = '[]'

    return num_variables, num_configurations, configurations_str, energies_str


def decode_configurations(num_variables, num_configurations, configurations_str, energies_str):
    configs = json.loads(configurations_str)
    energies = json.loads(energies_str)

    if energies:
        configurations = {_decode_config(config, num_variables): energy
                          for config, energy in zip(configs, energies)}
    else:
        configurations = {_decode_config(config, num_variables) for config in configs}

    return configurations
