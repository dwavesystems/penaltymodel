graph_table = """CREATE TABLE graphs (
                    num_nodes INTEGER NOT NULL,
                    num_edges INTEGER NOT NULL,
                    edges TEXT NOT NULL,
                    graph_id INTEGER PRIMARY KEY);
              """

graph_index = """CREATE INDEX idx_graph ON graphs(
                    num_nodes,
                    num_edges,
                    edges,
                    graph_id);
              """

configurations_table = """CREATE TABLE configurations (
                            num_variables INTEGER NOT NULL,
                            num_configurations INTEGER NOT NULL,
                            configurations TEXT NOT NULL,
                            energies TEXT,
                            configurations_id INTEGER PRIMARY KEY);"""

# model_table = """"""

schema_statements = [graph_table, graph_index,
                     configurations_table]
