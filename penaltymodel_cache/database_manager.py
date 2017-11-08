import sqlite3
import tempfile
import os

from penaltymodel.serialization import serialize_graph, serialize_configurations, serialize_decision_variables

from penaltymodel_cache.schema import schema_statements
from penaltymodel_cache.cache_manager import cache_file


def cache_connect(database=None, directory=None):
    """TODO"""

    conn = sqlite3.connect(cache_file(database, directory))

    # let us go ahead and populate the database with the tables we need. Each table
    # is only created if it does not already exist
    with conn as cur:
        for statement in schema_statements:
            cur.execute(statement)

    return conn


def graph_id(conn, nodelist, edgelist):
    """Get the unique id associated with each graph in the cache.

    Args:
        conn (:class:`sqlite3.Connection`): a connection to the database.
        nodelist (list): the nodes in the graph.
        edgelist (list): the edges in the graph.

    Returns:
        int: the unique id associated with the given graph.

    Notes:
        Inserts the graph into the sqlite3 database if it is not already
        present. If the graph is not present, the database is locked
        between query and insert.

    """
    # For a graph G s.t. |G| = N, the nodelist should be the integers [0,..,N-1]
    # and the edges should be an ordered list of the edges where each edge is itself
    # ordered. These should be able to be turned off.
    assert all(idx == v for idx, v in enumerate(nodelist))
    assert all(isinstance(u, int) and isinstance(v, int) for u, v in edgelist)
    assert all(u >= 0 and u < len(nodelist) and v >= 0 and v < len(nodelist) for u, v in edgelist)
    assert edgelist == sorted(tuple(sorted(edge)) for edge in edgelist)

    # serialize the graph. Returns a tuple (num_nodes, num_edges, edges_string)
    serial_graph = serialize_graph(nodelist, edgelist)

    select = "SELECT graph_id from graph WHERE num_nodes = ? and num_edges = ? and edges = ?;"
    insert = "INSERT INTO graph(num_nodes, num_edges, edges) VALUES (?, ?, ?);"

    with conn as cur:
        # get the graph_id
        row = cur.execute(select, serial_graph).fetchone()

        # if it's not there, insert and re-query
        if row is None:
            cur.execute(insert, serial_graph)
            row = cur.execute(select, serial_graph).fetchone()

    # the row should only have the id in it.
    graph_id, = row

    return graph_id


def get_configurations_id(conn, feasible_configurations):
    """Get the unique id associated with the given configurations.

    Args:
        conn (:class:`sqlite3.Connection`): a connection to the database.
        feasible_configurations (dict/set): The feasible configurations
            of the decision variables.

    Returns:
        int: The configuration_id as stored in the database.

    """
    # these should be checked already but we'll leave them as asserts for now
    assert isinstance(feasible_configurations, (set, dict))
    assert all(len(next(iter(feasible_configurations))) == len(config) for config in feasible_configurations)
    assert all(isinstance(energy, (int, float)) for energy in feasible_configurations.values()) \
        if isinstance(feasible_configurations, dict) else True

    serial_config = serialize_configurations(feasible_configurations)

    select = """SELECT feasible_configurations_id FROM feasible_configurations WHERE
                    num_variables = ? and
                    num_feasible_configurations = ? and
                    feasible_configurations = ? and
                    energies = ?;"""
    insert = """INSERT INTO feasible_configurations(
                    num_variables,
                    num_feasible_configurations,
                    feasible_configurations,
                    energies)
                VALUES (?, ?, ?, ?);"""

    with conn as cur:

        row = cur.execute(select, serial_config).fetchone()

        if row is None:
            cur.execute(insert, serial_config)
            row = cur.execute(select, serial_config).fetchone()

    return row[0]


def query_penalty_model(conn, graph, decision_variables, feasible_configurations):

    # we need to get a nodelist and an edgelist from graph
    if all(v in graph for v in range(len(graph))):
        # in this case the graph has indexlabelled [0, .., n-1]
        nodelist = list(range(len(graph)))
        edgelist = sorted(sorted(edge) for edge in graph.edges)
    else:
        raise NotImplementedError

    serial_graph = serialize_graph(nodelist, edgelist)
    serial_feasible_configurations = serialize_configurations(feasible_configurations)
    serial_decision_variables = serialize_decision_variables(decision_variables)

    select = \
        """
        SELECT
            linear_biases,
            quadratic_biases,
            offset,
            classical_gap,
            model_id
        FROM penalty_model
        WHERE
            num_variables = ? AND
            num_feasible_configurations = ? AND
            feasible_configurations = ? AND
            energies = ? AND
            num_nodes = ? AND
            num_edges = ? AND
            edges = ? AND
            decision_variables = ?;
        """

    key = serial_graph + serial_feasible_configurations + (serial_decision_variables,)
    with conn as cur:
        row = cur.execute(select, key)

        for r in row:
            yield -1


def load_penalty_model(conn, graph, decision_variables, feasible_configurations,
                       linear_biases, quadratic_biases, offset, classical_gap):
    # NB: we also want to track number of rows
    raise NotImplementedError
