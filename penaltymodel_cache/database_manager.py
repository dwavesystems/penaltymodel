import sqlite3
import tempfile
import os

from penaltymodel_cache.schema import schema_statements
from penaltymodel_cache.serialization import serialize_graph, serialize_configurations


def instantiate_database(conn):
    """TODO: builds tables etc for an empty database"""

    with conn as cur:
        for statement in schema_statements:
            cur.execute(statement)


def connection(database=None, directory=None):

    if database is None:
        database = filename(directory)

    return sqlite3.connect(database)


def filename(directory=None):
    """returns the database file (with path)
    """

    # if there is no specified directory for the database file, assume that it is
    # in the temporary directory (as is given by tempfile)
    if directory is None:
        directory = tempfile.gettempdir()

    filename = 'penaltymodel_cache.db'

    return os.path.join(directory, filename)


def get_graph_id(conn, graph):
    """Get the unique id associated with each graph in the cache.

    Args:
        conn (sqlite3.Connection): A sqlite3 Connection object.
        graph (nx.Graph): A networkx graph. Should be integer labeled.

    Returns:
        int: the unique id associated with the given graph.

    Notes:
        Inserts the graph into the sqlite3 database if it is not already
        present. If the graph is not present, the database is locked
        between query and insert.

    """
    # serialize the graph
    num_nodes, num_edges, edges = serialize_graph(graph)

    select = "SELECT graph_id from graphs WHERE num_nodes = ? and num_edges = ? and edges = ?;"
    insert = "INSERT INTO graphs(num_nodes, num_edges, edges) VALUES (?, ?, ?);"

    with conn as cur:
        row = cur.execute(select, (num_nodes, num_edges, edges)).fetchone()

        # if it's not there, insert and query again
        if row is None:
            cur.execute(insert, (num_nodes, num_edges, edges))
            row = cur.execute(select, (num_nodes, num_edges, edges)).fetchone()

    return row[0]


def get_configurations_id(conn, configurations):
    # assume everything is in the correct form

    serial_config = serialize_configurations(configurations)

    select = """SELECT configurations_id from configurations WHERE
                    num_variables = ? and
                    num_configurations = ? and
                    configurations = ? and
                    energies = ?;"""
    insert = """INSERT INTO configurations(
                    num_variables,
                    num_configurations,
                    configurations,
                    energies)
                VALUES (?, ?, ?, ?);"""

    with conn as cur:

        row = cur.execute(select, serial_config).fetchone()

        if row is None:
            cur.execute(insert, serial_config)
            row = cur.execute(select, serial_config).fetchone()

    return row[0]
