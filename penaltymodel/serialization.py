import json
import struct
import base64


def serialize_graph(nodelist, edgelist):
    """Converts a graph into a 3-tuple of serializable objects.

    Args:
        nodelist (list): a list of the nodes in the graph. Should be the integers
            0..n-1 inclusive where n-1 is the number of nodes in the graph.
        edgelist (list): a list of 2-tuples where each 2-tuple is an edge in the
            graph.

    Returns:
        (int, int, str): A 3-tuple of the form (num_nodes, num_edges, edges_string).
            edges_string is a json-encoded edgelist with no whitespace.

    Examples:
        >>> G = nx.complete_graph(3)
        >>> nodelist = sorted(G.nodes)
        >>> edgelist = sorted(sorted(edge) for edge in G.edges)
        >>> serialize_graph(nodelist, edgelist)
        (3, 3, '[[0,1],[0,2],[1,2]]')

    """
    num_nodes = len(nodelist)
    num_edges = len(edgelist)
    edges_str = json.dumps(edgelist, separators=(',', ':'))
    return num_nodes, num_edges, edges_str


def decode_graph(num_nodes, num_edges, edgelist_string):
    """Inverse of serialize_graph.

    Args:
        num_nodes: The number of nodes in the graph.
        num_edges: The number of edges in the graph.
        edgelist_string: A json-encoded edge list.

    Returns:
        list: a list of nodes in the graph
        list: a list of edges in the graph

    """
    return list(range(num_nodes)), [tuple(edge) for edge in json.loads(edgelist_string)]


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
        n = 1 << (num_variables - 1)
        for __ in range(num_variables):
            yield 1 if c & n else -1
            n >>= 1
    return tuple(bits(c))


def serialize_configurations(configurations):
    """Converts a set of configurations into a 4-tuple of serializable objects.

    Args:
        configurations (set/dict): A set of feasible configurations. For a set should
            each be a tuple of spins. For a dict, the keys should be the feasible
            configurations, the values should be the target energy.

    Returns:
        int: the number of variables (the length of each configurations)
        int: the number of configurations
        str: json-encoded list of configurations, sorted by integer.
        str: json-encoded list of energies. If all are 0, then empty list.

    Notes:
        Each configuration is stored as an integer. Where the spins are treated
            as the bits of the integer.

    """
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
    """Inverse of serialize_configurations.

    Args:
        int: the number of variables (the length of each configurations)
        int: the number of configurations
        str: json-encoded list of configurations, sorted by integer.
        str: json-encoded list of energies. If all are 0, then empty list.

    Returns:
        set/dict: A set of feasible configurations. For a set should
            each be a tuple of spins. For a dict, the keys should be the feasible
            configurations, the values should be the target energy.

    """
    configs = json.loads(configurations_str)
    energies = json.loads(energies_str)

    if energies:
        configurations = {_decode_config(config, num_variables): energy
                          for config, energy in zip(configs, energies)}
    else:
        configurations = {_decode_config(config, num_variables) for config in configs}

    return configurations


def _serialize_linear_biases(linear, nodelist):
    """Serializes the linear biases.

    Args:
        linear: a interable object where linear[v] is the bias
            associated with v.
        nodelist (list): an ordered iterable containing the nodes.

    Returns:
        str: base 64 encoded string of little endian 8 byte floats,
            one for each of the biases in linear. Ordered according
            to nodelist.

    Examples:
        >>> _serialize_linear_biases({1: -1, 2: 1, 3: 0}, [1, 2, 3])
        'AAAAAAAA8L8AAAAAAADwPwAAAAAAAAAA'
        >>> _serialize_linear_biases({1: -1, 2: 1, 3: 0}, [3, 2, 1])
        'AAAAAAAAAAAAAAAAAADwPwAAAAAAAPC/'

    """
    linear_bytes = struct.pack('<' + 'd' * len(linear), *[linear[i] for i in nodelist])
    return base64.b64encode(linear_bytes).decode('utf-8')


def _decode_linear_biases(linear_string, nodelist):
    """Inverse of _serialize_linear_biases.

    Args:
        linear_string (str): base 64 encoded string of little endian
            8 byte floats, one for each of the nodes in nodelist.
        nodelist (list): list of the form [node1, node2, ...].

    Returns:
        dict: linear biases in a dict.

    Examples:
        >>> _decode_linear_biases('AAAAAAAA8L8AAAAAAADwPwAAAAAAAAAA', [1, 2, 3])
        {1: -1.0, 2: 1.0, 3: 0.0}
        >>> _decode_linear_biases('AAAAAAAA8L8AAAAAAADwPwAAAAAAAAAA', [3, 2, 1])
        {1: 0.0, 2: 1.0, 3: -1.0}

    """
    linear_bytes = base64.b64decode(linear_string)
    return dict(zip(nodelist, struct.unpack('<' + 'd' * (len(linear_bytes) // 8), linear_bytes)))


def _serialize_quadratic_biases(quadratic, edgelist):
    """Serializes the quadratic biases.

    Args:
        quadratic (dict): a dict of the form {edge1: bias1, ...} where
            each edge is of the form (node1, node2).
        edgelist (list): a list of the form [(node1, node2), ...].

    Returns:
        str: base 64 encoded string of little endian 8 byte floats,
            one for each of the edges in quadratic. Ordered by edgelist.

    Example:
        >>> _serialize_quadratic_biases({(0, 1): -1, (1, 2): 1, (0, 2): .4},
        ...                             [(0, 1), (1, 2), (0, 2)])
        'AAAAAAAA8L8AAAAAAADwP5qZmZmZmdk/'

    """
    # assumes quadratic is upper-triangular or reflected in edgelist
    quadratic_list = [quadratic[(u, v)] if (u, v) in quadratic else quadratic[(v, u)]
                      for u, v in edgelist]
    quadratic_bytes = struct.pack('<' + 'd' * len(quadratic), *quadratic_list)
    return base64.b64encode(quadratic_bytes).decode('utf-8')


def _decode_quadratic_biases(quadratic_string, edgelist):
    """Inverse of _serialize_quadratic_biases

    Args:
        quadratic_string (str) : base 64 encoded string of little
            endian 8 byte floats, one for each of the edges.
        edgelist (list): a list of edges of the form [(node1, node2), ...].

    Returns:
        dict: J. A dict of the form {edge1: bias1, ...} where each
            edge is of the form (node1, node2).

    Example:
        >>> _decode_quadratic_biases('AAAAAAAA8L8AAAAAAADwP5qZmZmZmdk/',
        ...                          [(0, 1), (1, 2), (0, 2)])
        {(0, 1): -1.0, (0, 2): 0.4, (1, 2): 1.0}

    """
    quadratic_bytes = base64.b64decode(quadratic_string)
    return dict(zip(edgelist,
                    struct.unpack('<' + 'd' * (len(quadratic_bytes) // 8), quadratic_bytes)))


def serialize_biases(linear, quadratic, offset, nodelist, edgelist):
    """Serializes the linear and quadratic biases as well as the offset.

    Args:
        linear: a interable object where linear[v] is the bias
            associated with v.
        quadratic (dict): a dict of the form {edge1: bias1, ...} where
            each edge is of the form (node1, node2).
        offset (float): The real-valued offset
        nodelist (list): list of the form [node1, node2, ...].
        edgelist (list): a list of the form [(node1, node2), ...].

    Returns:
        str: base 64 encoded string of little endian 8 byte floats,
            one for each of the biases in linear. Ordered according
            to nodelist.
        str: base 64 encoded string of little endian 8 byte floats,
            one for each of the edges in quadratic. Ordered by edgelist.
        str: base 64 encoded string of little endian 8 byte float
            for the offset.

    """
    linear_string = _serialize_linear_biases(linear, nodelist)
    quadratic_string = _serialize_quadratic_biases(quadratic, edgelist)
    # offset = base64.b64encode(struct.pack('<d', offset)).decode('utf-8')

    # offset is a float and therefor serializable
    return linear_string, quadratic_string, offset


def decode_biases(linear_string, quadratic_string, offset, nodelist, edgelist):
    """Inverse of serialize_biases

    Args:
        linear_string (str): base 64 encoded string of little endian
            8 byte floats, one for each of the nodes in nodelist.
        offset (str): base 64 encoded string of little endian 8 byte float
            for the offset.
        quadratic_string (str) : base 64 encoded string of little
            endian 8 byte floats, one for each of the edges.
        nodelist (list): list of the form [node1, node2, ...].
        edgelist (list): a list of edges of the form [(node1, node2), ...].

    Returns:
        dict: linear biases in a dict of the form {v: bias,... }
        dict: quadratic biases in a dict of the form {(u, v): bias, ...}
        float: The offset.

    """
    linear = _decode_linear_biases(linear_string, nodelist)
    quadratic = _decode_quadratic_biases(quadratic_string, edgelist)
    # offset, = struct.unpack('<d', base64.b64decode(offset))
    return linear, quadratic, offset


def serialize_decision_variables(decision_variables):
    """Serializes the decision variables.

    Args:
        decision_variables (list/tuple): A list of the decision
            variables.

    Returns:
        str: json-encoded list of decision_variables.

    """
    return json.dumps(decision_variables, separators=(',', ':'))


def decode_decision_variables(decision_variables_string):
    """Inverse of serializable_decision_variables.

    Args:
        decision_variables_string (str): json-encoded list of
            decision_variables.

    Returns:
        list: A list of the decision variables.

    """
    return json.loads(decision_variables_string)
