"""A series of statements that can be executed to build the sqlite
database for the cache.
"""

version_table = \
    """CREATE TABLE IF NOT EXISTS version
        (
            major INTEGER NOT NULL,
            minor INTEGER NOT NULL,
            revision INTEGER NOT NULL
        );
    """

graph_table = \
    """
    CREATE TABLE IF NOT EXISTS graph
        (
            num_nodes INTEGER NOT NULL,
            num_edges INTEGER NOT NULL,
            edges TEXT NOT NULL,
            graph_id INTEGER PRIMARY KEY
        );
    """

# graph_index = """CREATE INDEX IF NOT EXISTS idx_graph ON graphs(
#                     num_nodes,
#                     num_edges,
#                     edges,
#                     graph_id);
#               """

feasible_configurations_table = \
    """
    CREATE TABLE IF NOT EXISTS feasible_configurations
        (
            num_variables INTEGER NOT NULL,
            num_feasible_configurations INTEGER NOT NULL,
            feasible_configurations TEXT NOT NULL,
            energies TEXT,
            feasible_configurations_id INTEGER PRIMARY KEY
        );
    """

# configurations_index = """CREATE INDEX IF NOT EXISTS idx_configurations ON configurations(
#                             num_variables,
#                             num_configurations,
#                             configurations,
#                             energies,
#                             configurations_id);"""

model_table = \
    """
    CREATE TABLE IF NOT EXISTS model
        (
            graph_id INTEGER NOT NULL,
            decision_variables TEXT NOT NULL,
            feasible_configurations_id INTEGER NOT NULL,
            linear_biases TEXT NOT NULL,
            quadratic_biases TEXT NOT NULL,
            offset TEXT NOT NULL,
            classical_gap REAL NOT NULL,
            model_id INTEGER PRIMARY KEY
        );
    """

penalty_model_view = \
    """
    CREATE VIEW IF NOT EXISTS  penalty_model AS
    SELECT
        num_variables,
        num_feasible_configurations,
        feasible_configurations,
        energies,
        num_nodes,
        num_edges,
        edges,
        decision_variables,
        linear_biases,
        quadratic_biases,
        offset,
        classical_gap,
        model_id
    FROM
        model,
        feasible_configurations,
        graph
    WHERE
        model.graph_id = graph.graph_id
        AND model.feasible_configurations_id = feasible_configurations.feasible_configurations_id;
    """

schema_statements = [version_table,
                     graph_table,
                     feasible_configurations_table,
                     model_table,
                     penalty_model_view]
