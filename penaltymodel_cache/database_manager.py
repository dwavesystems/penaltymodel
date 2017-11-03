import sqlite3
import tempfile
import os

from penaltymodel.serialization import serialize_graph, serialize_configurations, serialize_decision_variables

from penaltymodel_cache.schema import schema_statements
from penaltymodel_cache.version import __version__


class OutdatedDatabaseException(Exception):
    """Database uses an outdated schema"""


def configure_database(conn):
    """For a given connection, adds the schema for the penalty model cache.

    If the database has already been configured, does nothing.

    Args:
        conn (:class:`sqlite3.Connection`): a connection to the database.

    Raises:
        OutdatedDatabaseException: If the version as stored in the version
            table in the connected database does not match the major and
            minor version number of the currently executing code.

    """

    # we want to know if the database has already been configured. That is it has
    # a matching version number. The schema can only change between major or minor
    # version increments
    version = __version__.split('.')

    table_select = "SELECT name FROM sqlite_master WHERE type='table' AND name='version';"
    version_select = "SELECT revision FROM version WHERE major = ? and minor = ?;"
    version_insert = "INSERT INTO version(major, minor, revision) VALUES (?, ?, ?);"

    with conn as cur:
        # see if there is a current version number
        tbl = conn.execute(table_select).fetchone()

        if tbl is not None:
            # there is a version number stored

            # get the value
            major, minor, __ = version
            row = conn.execute(version_select, (major, minor)).fetchone()

            if row is None:
                # we should wipe the database here
                raise OutdatedDatabaseException('given database matches old version')
            else:
                # we are done! We could check all of the schema but let's just assume that people
                # have not gone out of their way to mess with the database
                return

        # we assume that the database has not been set up or has just been wiped
        # so we can just add the tables
        for statement in schema_statements:
            cur.execute(statement)

        # record the current version
        conn.execute(version_insert, version)


def connection(database=None, directory=None):
    """Open a connection to a sqlite3 database file.

    Args:
        database (str, optional): The path to the desired database. When
            no database is provided, a new one is created. A special
            value ':memory:' can be given to only build the database in
            memory.
        directory (str, optional): If specified, database file is built
            in the given directory. The 'database' parameter takes
            precedence. If neither a database or directory parameter
            are provided, then a database is created in system temporary
            directory.

    Returns:
        :class:`sqlite3.Connection`: A connection to the database.

    """

    if database is None:
        database = _filename(directory)

    conn = sqlite3.connect(database)

    # make sure the database has all of the correct fields
    configure_database(conn)

    return conn


def _filename(directory=None):
    """returns the database file (with path) in the given directory. If no
    directory is given then puts it in the temp directory.
    """

    # if there is no specified directory for the database file, assume that it is
    # in the temporary directory (as is given by tempfile)
    if directory is None:
        directory = tempfile.gettempdir()

    filename = 'penaltymodel_cache.db'

    return os.path.join(directory, filename)


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
