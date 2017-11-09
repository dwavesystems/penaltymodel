"""A series of statements that can be executed to build the sqlite
database for the cache.
"""
# name conventions from: https://launchbylunch.com/posts/2014/Feb/16/sql-naming-conventions/

graph = \
    """
    CREATE TABLE IF NOT EXISTS graph
        (
            num_nodes INTEGER NOT NULL,
            num_edges INTEGER NOT NULL,
            edges TEXT NOT NULL,
            id INTEGER PRIMARY KEY
        );
    """

# graph_ix_num_nodes_num_edges_edges_id = """CREATE INDEX IF NOT EXISTS idx_graph ON graphs(
#                     num_nodes,
#                     num_edges,
#                     edges,
#                     id);
#               """

feasible_configurations = \
    """
    CREATE TABLE IF NOT EXISTS feasible_configurations
        (
            num_variables INTEGER NOT NULL,
            num_feasible_configurations INTEGER NOT NULL,
            feasible_configurations TEXT NOT NULL,
            energies TEXT,
            id INTEGER PRIMARY KEY
        );
    """

# todo: name
# configurations_index = """CREATE INDEX IF NOT EXISTS idx_configurations ON configurations(
#                             num_variables,
#                             num_configurations,
#                             configurations,
#                             energies,
#                             configurations_id);"""

# table that encodes the ising model values. This table does not need an index created because
# SELECTS are done on id.
ising_model = \
    """
    CREATE TABLE IF NOT EXISTS ising_model
        (
            linear_biases TEXT NOT NULL,
            quadratic_biases TEXT NOT NULL,
            offset REAL NOT NULL,
            max_quadratic_bias REAL NOT NULL,
            min_quadratic_bias REAL NOT NULL,
            max_linear_bias REAL NOT NULL,
            min_linear_bias REAL NOT NULL,
            id INTEGER PRIMARY KEY
        );
    """

penalty_model = \
    """
    CREATE TABLE IF NOT EXISTS penalty_model
        (
            decision_variables TEXT NOT NULL,
            classical_gap REAL NOT NULL,
            ground_energy REAL NOT NULL,
            graph_id INT,
            feasible_configurations_id INT,
            ising_model_id INT,
            id INTEGER PRIMARY KEY,
                FOREIGN KEY (graph_id) REFERENCES graph(id) ON DELETE CASCADE,
                FOREIGN KEY (feasible_configurations_id) REFERENCES feasible_configurations(id) ON DELETE CASCADE,
                FOREIGN KEY (ising_model_id) REFERENCES ising_model(id) ON DELETE CASCADE
        );
    """

penalty_model_view = \
    """
    CREATE VIEW IF NOT EXISTS penalty_model_view AS
    SELECT
        num_variables,
        num_feasible_configurations,
        feasible_configurations,
        energies,

        num_nodes,
        num_edges,
        edges,

        linear_biases,
        quadratic_biases,
        offset,
        max_quadratic_bias,
        min_quadratic_bias,
        max_linear_bias,
        min_linear_bias,

        decision_variables,
        classical_gap,
        ground_energy,
        penalty_model.id
    FROM
        ising_model,
        feasible_configurations,
        graph,
        penalty_model
    WHERE
        penalty_model.ising_model_id = ising_model.id
        AND feasible_configurations.id = penalty_model.feasible_configurations_id
        AND graph.id = penalty_model.graph_id;
    """

schema_statements = [graph,
                     feasible_configurations,
                     ising_model,
                     penalty_model,
                     penalty_model_view
                     ]
