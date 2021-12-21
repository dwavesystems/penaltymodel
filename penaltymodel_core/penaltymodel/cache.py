# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import sqlite3
import os
import json
import struct

from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import dimod
import homebase
import networkx as nx

from penaltymodel.core.package_info import __version__  # todo: move
from penaltymodel.exceptions import MissingPenaltyModel
from penaltymodel.typing import PenaltyModel

__all__ = ['PenaltyModelCache']


class PenaltyModelCache(contextlib.AbstractContextManager):
    """

    Args:
        database (str, optional): The path to the database the user wishes
            to connect to. If not specified, a default is chosen using
            :func:`.cache_file`. If the special database name ':memory:'
            is given, then a temporary database is created in memory.

    """

    database_schema = \
        """
        CREATE TABLE IF NOT EXISTS graph(
            num_nodes INTEGER NOT NULL,
            num_edges INTEGER NOT NULL,
            edges TEXT NOT NULL,  -- json list of lists, should be sorted (with each edge sorted)
            id INTEGER PRIMARY KEY,
            CONSTRAINT graph UNIQUE (num_nodes, edges)
        );

        CREATE TABLE IF NOT EXISTS feasible_configurations(
            num_variables INTEGER NOT NULL,
            num_feasible_configurations INTEGER NOT NULL,
            feasible_configurations TEXT NOT NULL,
            energies BLOB NOT NULL,
            id INTEGER PRIMARY KEY,
            CONSTRAINT feasible_configurations UNIQUE (
                num_variables,
                feasible_configurations,
                energies)
        );

        CREATE TABLE IF NOT EXISTS binary_quadratic_model(
            bqm_data BLOB NOT NULL,
            max_quadratic_bias REAL NOT NULL,
            min_quadratic_bias REAL NOT NULL,
            max_linear_bias REAL NOT NULL,
            min_linear_bias REAL NOT NULL,
            graph_id INTEGER NOT NULL,
            id INTEGER PRIMARY KEY,
            CONSTRAINT ising_model UNIQUE (bqm_data, graph_id),
            FOREIGN KEY (graph_id) REFERENCES graph(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS penalty_model(
            decision_variables TEXT NOT NULL,
            classical_gap REAL NOT NULL,
            feasible_configurations_id INT,
            bqm_id INT,
            id INTEGER PRIMARY KEY,
            CONSTRAINT penalty_model UNIQUE (decision_variables, feasible_configurations_id, bqm_id),
            FOREIGN KEY (feasible_configurations_id) REFERENCES feasible_configurations(id) ON DELETE CASCADE,
            FOREIGN KEY (bqm_id) REFERENCES binary_quadratic_model(id) ON DELETE CASCADE
        );

        CREATE VIEW IF NOT EXISTS penalty_model_view AS
        SELECT
            num_variables,
            num_feasible_configurations,
            feasible_configurations,
            energies,

            num_nodes,
            num_edges,
            edges,

            bqm_data,
            max_quadratic_bias,
            min_quadratic_bias,
            max_linear_bias,
            min_linear_bias,

            decision_variables,
            classical_gap,
            penalty_model.id
        FROM
            binary_quadratic_model,
            feasible_configurations,
            graph,
            penalty_model
        WHERE
            penalty_model.bqm_id = binary_quadratic_model.id
            AND feasible_configurations.id = penalty_model.feasible_configurations_id
            AND graph.id = binary_quadratic_model.graph_id;
        """

    insert_bqm_statement = \
        """
        INSERT OR IGNORE INTO binary_quadratic_model(
            bqm_data,
            max_quadratic_bias,
            min_quadratic_bias,
            max_linear_bias,
            min_linear_bias,
            graph_id)
        SELECT
            :bqm_data,
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

    insert_graph_statement = \
        """
        INSERT OR IGNORE INTO graph(num_nodes, num_edges, edges)
        VALUES (:num_nodes, :num_edges, :edges);
        """

    insert_penalty_model_statement = \
        """
        INSERT OR IGNORE INTO penalty_model(
            decision_variables,
            classical_gap,
            feasible_configurations_id,
            bqm_id)
        SELECT
            :decision_variables,
            :classical_gap,
            feasible_configurations.id,
            binary_quadratic_model.id
        FROM feasible_configurations, binary_quadratic_model, graph
        WHERE
            graph.edges = :edges AND
            graph.num_nodes = :num_nodes AND
            binary_quadratic_model.graph_id = graph.id AND
            binary_quadratic_model.bqm_data = :bqm_data AND
            feasible_configurations.num_variables = :num_variables AND
            feasible_configurations.feasible_configurations = :feasible_configurations AND
            feasible_configurations.energies = :energies;
        """

    insert_table_statement = \
        """
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

    def __init__(self, database: Optional[Union[str, os.PathLike]] = None):
        if database is None:
            database = os.path.join(
                homebase.user_data_dir(
                    app_name='dwave-penaltymodel-cache',
                    app_author='dwave-systems',
                    create=True,
                    ),
                f'penaltymodel_v{__version__}.db'
                )

        if os.path.isfile(database):
            conn = sqlite3.connect(database)
        else:
            conn = sqlite3.connect(database)
            conn.executescript(self.database_schema)

        with conn as cur:
            # turn on foreign keys, allows deletes to cascade.
            cur.execute("PRAGMA foreign_keys = ON;")

        # give us mapping access to values returned by .execute
        conn.row_factory = sqlite3.Row

        self.conn = conn

    def __exit__(self, *args):
        # todo: make reentrant
        self.close()

    def close(self):
        self.conn.close()

    @staticmethod
    def encode_graph(graph: Union[nx.Graph, dimod.BinaryQuadraticModel]
                     ) -> Dict[str, Union[int, str]]:
        """Encode a NetworkX graph or BQM to be stored in the cache."""
        if isinstance(graph, nx.Graph):
            nodes = graph.nodes
            edges = graph.edges
        else:
            nodes = graph.linear
            edges = graph.quadratic

        return dict(
            num_nodes=len(nodes),
            num_edges=len(edges),
            edges=json.dumps(sorted(map(sorted, edges)), separators=(',', ':')),
            )

    @staticmethod
    def decode_graph(row: Dict[str, Union[int, str]]) -> nx.Graph:
        """Decode a row in the cache to a NetworkX graph."""
        graph = nx.Graph()
        graph.add_nodes_from(range(row['num_nodes']))
        graph.add_edges_from(json.loads(row['edges']))
        return graph

    def insert_graph(self, graph: nx.Graph):
        """Insert a graph into the cache.

        A graph is stored by number of nodes, number of edges and a
        json-encoded list of edges.

        Args:
            graph: a NetworkX graph.

        Notes:
            This function assumes that the nodes are index-labeled and range
            from 0 to num_nodes - 1.

        """

        if graph.nodes ^ range(len(graph.nodes)):
            raise ValueError("graph nodes must be exactly range(num_nodes)")

        with self.conn as cur:
            cur.execute(self.insert_graph_statement, self.encode_graph(graph))

    def iter_graphs(self) -> Iterator[nx.Graph]:
        """Iterate over all graphs in the cache.

        Yields:
            All the graphs in the cache, as NetworkX graphs.

        """
        yield from map(self.decode_graph, self.conn.execute("SELECT num_nodes, edges from graph;"))

    @staticmethod
    def encode_table(table: Dict[Tuple[int, ...], float]) -> Dict[str, Union[int, str, bytes]]:
        """Encode a table of feasible configurations to be stored in the cache."""

        variable_counts = set(map(len, table))
        if len(variable_counts) == 0:
            num_variables = 0
        elif len(variable_counts) == 1:
            num_variables, = variable_counts
        else:
            raise ValueError

        def config_to_int(config: Tuple[int, ...]) -> int:
            out = 0
            for bit in config:
                out = (out << 1) | (bit > 0)
            return out

        configs, energies = zip(*sorted((config_to_int(k), v) for k, v in table.items()))

        return dict(
            num_variables=num_variables,
            num_feasible_configurations=len(table),
            feasible_configurations=json.dumps(configs, separators=(',', ':')),
            energies=struct.pack('<' + 'd' * len(energies), *energies),
            )

    @staticmethod
    def decode_table(row: Dict[str, Union[int, str, bytes]]) -> Dict[Tuple[int, ...], float]:
        num_variables = row['num_variables']

        def bits(c):
            n = 1 << (num_variables - 1)
            for __ in range(num_variables):
                yield 1 if c & n else -1
                n >>= 1

        return dict(zip(
            (tuple(bits(c)) for c in json.loads(row['feasible_configurations'])),
            struct.unpack('<' + 'd' * (len(row['energies']) // 8), row['energies'])
            ))

    def insert_table(self, table: Dict[Tuple[int, ...], float]):
        """Insert a group of feasible configurations into the cache.

        Args:
            table: The set of feasible configurations. Each key should be a
                tuple of variable assignments. The values are the relative
                energies.

        """
        with self.conn as cur:
            cur.execute(self.insert_table_statement, self.encode_table(table))

    def iter_tables(self) -> Iterator[Dict[Tuple[int, ...], float]]:
        select = "SELECT num_variables, feasible_configurations, energies FROM feasible_configurations"
        yield from map(self.decode_table, self.conn.execute(select))

    @staticmethod
    def encode_bqm(bqm: dimod.BinaryQuadraticModel) -> Dict[str, Union[float, bytes]]:
        return dict(
            max_quadratic_bias=bqm.quadratic.max(),
            min_quadratic_bias=bqm.quadratic.min(),
            max_linear_bias=bqm.linear.max(),
            min_linear_bias=bqm.linear.min(),
            bqm_data=bqm.to_file().read(),
            )

    @staticmethod
    def decode_bqm(row: Dict[str, Union[bytes, str, int]]) -> dimod.BinaryQuadraticModel:
        return dimod.BinaryQuadraticModel.from_file(row['bqm_data'])

    def insert_binary_quadratic_model(self, bqm: dimod.BinaryQuadraticModel):
        """
        converted to SPIN
        must be integer labelled

        """
        bqm = dimod.as_bqm(bqm, dtype=float)

        if bqm.vartype is not dimod.SPIN:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

        if not bqm.variables.is_range:
            if bqm.variables ^ range(bqm.num_variables):
                raise ValueError("BQM variables must be a range of integers")
            else:
                new = dimod.BinaryQuadraticModel(bqm.num_variables, dimod.SPIN)
                new.add_linear_from(bqm.linear)
                new.add_quadratic_from(bqm.quadratic)
                new.offset = bqm.offset
                bqm = new

        parameters = self.encode_graph(bqm) | self.encode_bqm(bqm)

        with self.conn as cur:
            cur.execute(self.insert_graph_statement, parameters)
            cur.execute(self.insert_bqm_statement, parameters)

    def iter_binary_quadratic_models(self) -> Iterator[dimod.BinaryQuadraticModel]:
        for bqm_data in self.conn.execute("SELECT bqm_data FROM binary_quadratic_model;"):
            yield self.decode_bqm(bqm_data)

    def insert_penalty_model(
            self,
            bqm: dimod.BinaryQuadraticModel,
            table: Dict[Tuple[int, ...], float],
            decision: Sequence[int],
            classical_gap: float,
            ):
        """Does not check for correctness"""

        # todo: test decision subset of bqm

        parameters = self.encode_graph(bqm) | self.encode_bqm(bqm) | self.encode_table(table)

        parameters.update(
            decision_variables=json.dumps(decision, separators=(',', ':')),
            classical_gap=classical_gap,
            )

        with self.conn as cur:
            cur.execute(self.insert_graph_statement, parameters)
            cur.execute(self.insert_bqm_statement, parameters)
            cur.execute(self.insert_table_statement, parameters)
            cur.execute(self.insert_penalty_model_statement, parameters)

    def iter_penalty_models(self) -> Iterator[PenaltyModel]:
        for row in self.conn.execute("SELECT * FROM penalty_model_view;"):
            yield PenaltyModel(
                    self.decode_bqm(row),
                    self.decode_table(row),
                    json.loads(row['decision_variables']),
                    row['classical_gap']
                )

    def retrieve(self,
                 graph: nx.Graph,
                 table: Mapping[Tuple[int, ...], float],
                 decision: Sequence[int],
                 *,
                 linear_bound: Tuple[float, float] = (-2, 2),
                 quadratic_bound: Tuple[float, float] = (-1, 1),
                 min_classical_gap: float = 2,
                 ) -> Tuple[dimod.BinaryQuadraticModel, float]:

        parameters = self.encode_graph(graph)
        parameters.update(self.encode_table(table))
        parameters.update(
            decision_variables=json.dumps(decision, separators=(',', ':')),
            min_classical_gap=min_classical_gap,
            min_linear_bias=linear_bound[0],
            max_linear_bias=linear_bound[1],
            min_quadratic_bias=quadratic_bound[0],
            max_quadratic_bias=quadratic_bound[1],
            )

        # print(parameters)

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT bqm_data, classical_gap FROM penalty_model_view
            WHERE
                -- graph:
                num_nodes = :num_nodes AND
                num_edges = :num_edges AND
                edges = :edges AND
                -- feasible_configurations:
                num_variables = :num_variables AND
                feasible_configurations = :feasible_configurations AND
                energies = :energies AND
                -- decision variables:
                decision_variables = :decision_variables AND
                -- bounds
                min_linear_bias >= :min_linear_bias AND
                max_linear_bias <= :max_linear_bias AND
                min_quadratic_bias >= :min_quadratic_bias AND
                max_quadratic_bias <= :max_quadratic_bias AND
                -- gap
                classical_gap >= :min_classical_gap
            ORDER BY classical_gap DESC;
            """,
            parameters
            )

        row = cur.fetchone()
        cur.close()

        if row is None:
            raise MissingPenaltyModel(
                "no penalty model with the given specification found in cache")

        return self.decode_bqm(row), row['classical_gap']
