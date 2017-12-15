"""This module includes all of the functions that interact with the sqlite
database directly.

Encoding
--------
num_nodes (int): the number of nodes in a graph
num_edges (int): the number of edges in a graph
edges (str): The json-encoded list of edges.
"""
import sqlite3
import os
import json
import struct
import base64

from six import itervalues
import penaltymodel as pm
import networkx as nx

from penaltymodel_cache.schema import schema
from penaltymodel_cache.cache_manager import cache_file

__all__ = ["cache_connect",
           "insert_graph", "iter_graph",
           "insert_feasible_configurations", "iter_feasible_configurations",
           "insert_ising_model", "iter_ising_model",
           "insert_penalty_model", "iter_penalty_model_from_specification"]


def cache_connect(database=cache_file()):
    """Returns a connection object to a sqlite database.

    Args:
        database (str, optional): The path to the database the user wishes
            to connect to. If not specified, a default is chosen.

    Returns:
        :class:`sqlite3.Connection`

    """
    if os.path.isfile(database):
        # just connect to the database as-is
        conn = sqlite3.connect(database)
    else:
        # we need to populate the database
        conn = sqlite3.connect(database)
        conn.executescript(schema)

    with conn as cur:
        # turn on foreign keys, allows deletes to cascade.
        cur.execute("PRAGMA foreign_keys = ON;")

    conn.row_factory = sqlite3.Row

    return conn


def insert_graph(cur, nodelist, edgelist, encoded_data=None):
    """Insert a graph into the cache.

    Args:
        nodelist (list): A list of nodes in the graph.
        edgelist (list): A list of edges in the graph.
        encoded_data (dict, optional): A dictionary with encoded
            data. If nodelist or edgelist was previously encoded
            then it can speed up execution, otherwise the encoding
            is added to the provided encoded_data dict.

    Notes:
        The cache will not recognize different orderings of
        nodelist or edgelist, so for efficiency it is best
        to be consistent.

    """
    if encoded_data is None:
        encoded_data = {}

    if 'num_nodes' not in encoded_data:
        encoded_data['num_nodes'] = len(nodelist)
    if 'num_edges' not in encoded_data:
        encoded_data['num_edges'] = len(edgelist)
    if 'edges' not in encoded_data:
        encoded_data['edges'] = json.dumps(edgelist, separators=(',', ':'))

    insert = \
        """
        INSERT OR IGNORE INTO graph(num_nodes, num_edges, edges)
        VALUES (:num_nodes, :num_edges, :edges);
        """

    cur.execute(insert, encoded_data)


def iter_graph(cur):
    """Iterate over all graphs in the cache.

    Yields:
        tuple: A 2-tuple containing:

            list: The nodelist for the graph.

            list: the edgelist for the graph.

    """
    select = """SELECT num_nodes, num_edges, edges from graph;"""
    for num_nodes, num_edges, edges in cur.execute(select):
        yield list(range(num_nodes)), json.loads(edges)


def insert_feasible_configurations(cur, feasible_configurations, encoded_data=None):
    """todo
    """
    if encoded_data is None:
        encoded_data = {}

    if 'num_variables' not in encoded_data:
        encoded_data['num_variables'] = len(next(iter(feasible_configurations)))
    if 'num_feasible_configurations' not in encoded_data:
        encoded_data['num_feasible_configurations'] = len(feasible_configurations)
    if 'feasible_configurations' not in encoded_data or 'energies' not in encoded_data:
        encoded = {_serialize_config(config): en for config, en in feasible_configurations.items()}

        configs, energies = zip(*sorted(encoded.items()))
        encoded_data['feasible_configurations'] = json.dumps(configs, separators=(',', ':'))
        encoded_data['energies'] = json.dumps(energies, separators=(',', ':'))

    insert = """
            INSERT OR IGNORE INTO feasible_configurations(
                num_variables,
                num_feasible_configurations,
                feasible_configurations,
                energies)
            VALUES (
                :num_variables,
                :num_feasible_configurations,
                :feasible_configurations,
                :energies);
            """

    cur.execute(insert, encoded_data)


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


def iter_feasible_configurations(cur):
    """todo"""
    select = \
        """
        SELECT num_variables, feasible_configurations, energies
        FROM feasible_configurations
        """
    for num_variables, feasible_configurations, energies in cur.execute(select):
        configs = json.loads(feasible_configurations)
        energies = json.loads(energies)

        yield {_decode_config(config, num_variables): energy
               for config, energy in zip(configs, energies)}


def _decode_config(c, num_variables):
    """inverse of _serialize_config, always converts to spin."""
    def bits(c):
        n = 1 << (num_variables - 1)
        for __ in range(num_variables):
            yield 1 if c & n else -1
            n >>= 1
    return tuple(bits(c))


def insert_ising_model(cur, nodelist, edgelist, linear, quadratic, offset, encoded_data=None):
    """todo"""
    if encoded_data is None:
        encoded_data = {}

    # insert graph and partially populate encoded_data with graph info
    insert_graph(cur, nodelist, edgelist, encoded_data=encoded_data)

    # need to encode the biases
    if 'linear_biases' not in encoded_data:
        encoded_data['linear_biases'] = _serialize_linear_biases(linear, nodelist)
    if 'quadratic_biases' not in encoded_data:
        encoded_data['quadratic_biases'] = _serialize_quadratic_biases(quadratic, edgelist)
    if 'offset' not in encoded_data:
        encoded_data['offset'] = offset
    if 'max_quadratic_bias' not in encoded_data:
        encoded_data['max_quadratic_bias'] = max(itervalues(quadratic))
    if 'min_quadratic_bias' not in encoded_data:
        encoded_data['min_quadratic_bias'] = min(itervalues(quadratic))
    if 'max_linear_bias' not in encoded_data:
        encoded_data['max_linear_bias'] = max(itervalues(linear))
    if 'min_linear_bias' not in encoded_data:
        encoded_data['min_linear_bias'] = min(itervalues(linear))

    insert = \
        """
        INSERT OR IGNORE INTO ising_model(
            linear_biases,
            quadratic_biases,
            offset,
            max_quadratic_bias,
            min_quadratic_bias,
            max_linear_bias,
            min_linear_bias,
            graph_id)
        SELECT
            :linear_biases,
            :quadratic_biases,
            :offset,
            :max_quadratic_bias,
            :min_quadratic_bias,
            :max_linear_bias,
            :min_linear_bias,
            graph.id
        FROM graph WHERE
            num_nodes = :num_nodes AND
            num_edges = :num_edges AND
            edges = :edges;
        """

    cur.execute(insert, encoded_data)


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


def iter_ising_model(cur):
    select = \
        """
        SELECT linear_biases, quadratic_biases, num_nodes, edges
        FROM ising_model, graph
        WHERE graph.id = ising_model.graph_id;
        """

    for linear_biases, quadratic_biases, num_nodes, edges in cur.execute(select):
        nodelist = list(range(num_nodes))
        edgelist = json.loads(edges)
        yield (nodelist, edgelist,
               _decode_linear_biases(linear_biases, nodelist),
               _decode_quadratic_biases(quadratic_biases, edgelist))


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
    return {tuple(edge): bias for edge, bias in zip(edgelist,
            struct.unpack('<' + 'd' * (len(quadratic_bytes) // 8), quadratic_bytes))}


def insert_penalty_model(cur, penalty_model, encoded_data=None):
    """todo"""
    if encoded_data is None:
        encoded_data = {}

    nodelist = sorted(penalty_model.graph)
    edgelist = sorted(sorted(penalty_model.graph.edges))
    linear, quadratic, offset = penalty_model.model.as_ising()

    insert_graph(cur, nodelist, edgelist, encoded_data)
    insert_feasible_configurations(cur, penalty_model.feasible_configurations, encoded_data)
    insert_ising_model(cur, nodelist, edgelist, linear, quadratic, offset, encoded_data)

    if 'decision_variables' not in encoded_data:
        encoded_data['decision_variables'] = json.dumps(penalty_model.decision_variables, separators=(',', ':'))
    if 'classical_gap' not in encoded_data:
        encoded_data['classical_gap'] = penalty_model.classical_gap
    if 'ground_energy' not in encoded_data:
        encoded_data['ground_energy'] = penalty_model.ground_energy

    insert = \
        """
        INSERT OR IGNORE INTO penalty_model(
            decision_variables,
            classical_gap,
            ground_energy,
            feasible_configurations_id,
            ising_model_id)
        SELECT
            :decision_variables,
            :classical_gap,
            :ground_energy,
            feasible_configurations.id,
            ising_model.id
        FROM feasible_configurations, ising_model, graph
        WHERE
            graph.edges = :edges AND
            graph.num_nodes = :num_nodes AND
            ising_model.graph_id = graph.id AND
            ising_model.linear_biases = :linear_biases AND
            ising_model.quadratic_biases = :quadratic_biases AND
            ising_model.offset = :offset AND
            feasible_configurations.num_variables = :num_variables AND
            feasible_configurations.num_feasible_configurations = :num_feasible_configurations AND
            feasible_configurations.feasible_configurations = :feasible_configurations AND
            feasible_configurations.energies = :energies;
        """

    cur.execute(insert, encoded_data)


def iter_penalty_model_from_specification(cur, specification, encoded_data=None):
    """todo"""
    if encoded_data is None:
        encoded_data = {}

    nodelist = sorted(specification.graph)
    edgelist = sorted(sorted(edge) for edge in specification.graph.edges)
    if 'num_nodes' not in encoded_data:
        encoded_data['num_nodes'] = len(nodelist)
    if 'num_edges' not in encoded_data:
        encoded_data['num_edges'] = len(edgelist)
    if 'edges' not in encoded_data:
        encoded_data['edges'] = json.dumps(edgelist, separators=(',', ':'))
    if 'num_variables' not in encoded_data:
        encoded_data['num_variables'] = len(next(iter(specification.feasible_configurations)))
    if 'num_feasible_configurations' not in encoded_data:
        encoded_data['num_feasible_configurations'] = len(specification.feasible_configurations)
    if 'feasible_configurations' not in encoded_data or 'energies' not in encoded_data:
        encoded = {_serialize_config(config): en for config, en in specification.feasible_configurations.items()}
        configs, energies = zip(*sorted(encoded.items()))
        encoded_data['feasible_configurations'] = json.dumps(configs, separators=(',', ':'))
        encoded_data['energies'] = json.dumps(energies, separators=(',', ':'))

    select = \
        """
        SELECT
            linear_biases,
            quadratic_biases,
            offset,
            decision_variables,
            classical_gap,
            ground_energy
        FROM penalty_model_view
        WHERE
            -- graph:
            num_nodes = :num_nodes AND
            num_edges = :num_edges AND
            edges = :edges AND
            -- feasible_configurations:
            num_variables = :num_variables AND
            num_feasible_configurations = :num_feasible_configurations AND
            feasible_configurations = :feasible_configurations AND
            energies = :energies
            -- we could apply filters based on the energy ranges but in practice this seems slower
        ORDER BY classical_gap DESC;
        """

    for row in cur.execute(select, encoded_data):
        # we need to build the model
        linear = _decode_linear_biases(row['linear_biases'], nodelist)
        quadratic = _decode_quadratic_biases(row['quadratic_biases'], edgelist)

        model = pm.BinaryQuadraticModel(linear, quadratic, row['offset'], pm.SPIN)

        yield pm.PenaltyModel.from_specification(specification, model, row['classical_gap'], row['ground_energy'])
