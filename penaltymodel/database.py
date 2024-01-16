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
import functools
import sqlite3
import os
import json
import struct
import tempfile
import threading

from typing import Dict, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import dimod
import homebase
import networkx as nx
import numpy as np

from penaltymodel import __version__
from penaltymodel.exceptions import MissingPenaltyModel
from penaltymodel.typing import GraphLike
from penaltymodel.utils import as_graph

__all__ = ['PenaltyModelCache']


# developer note: we could use sqlite's adaptor's methods
# but since those changes are applied globally and since we're using pretty
# standard types (NumPy arrays and NetworkX graphs) it makes sense to
# do it "by hand" rather than risk interfering with other's code.


class PenaltyModel(NamedTuple):
    bqm: dimod.BinaryQuadraticModel
    sampleset: dimod.SampleSet
    classical_gap: float


class PenaltyModelCache(contextlib.AbstractContextManager):
    """Manage a database of penalty models.

    Penalty models are stored in an :mod:`sqlite3` database.

    This class can be used as a context manager to automatically close
    the database connection on exit.

    Args:
        database:
            The path to the database the user wishes to connect to.
            The default path will depend on the operating system, certain
            environmental variables and whether it is being run inside a
            virtual environment.
            See `homebase <https://github.com/dwavesystems/homebase>`_.
            If the special database name ':memory:' is given, then a temporary
            database is created in memory.

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

        CREATE TABLE IF NOT EXISTS sampleset(
            num_variables INTEGER NOT NULL,
            num_samples INTEGER NOT NULL,
            samples TEXT NOT NULL,
            energies BLOB NOT NULL,
            id INTEGER PRIMARY KEY,
            CONSTRAINT sampleset UNIQUE (
                num_variables,
                samples,
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
            sampleset_id INT,
            bqm_id INT,
            id INTEGER PRIMARY KEY,
            CONSTRAINT penalty_model UNIQUE (decision_variables, sampleset_id, bqm_id),
            FOREIGN KEY (sampleset_id) REFERENCES sampleset(id) ON DELETE CASCADE,
            FOREIGN KEY (bqm_id) REFERENCES binary_quadratic_model(id) ON DELETE CASCADE
        );

        CREATE VIEW IF NOT EXISTS penalty_model_view AS
        SELECT
            num_variables,
            num_samples,
            samples,
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
            sampleset,
            graph,
            penalty_model
        WHERE
            penalty_model.bqm_id = binary_quadratic_model.id
            AND sampleset.id = penalty_model.sampleset_id
            AND graph.id = binary_quadratic_model.graph_id;

        PRAGMA foreign_keys = ON;
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
            sampleset_id,
            bqm_id)
        SELECT
            :decision_variables,
            :classical_gap,
            sampleset.id,
            binary_quadratic_model.id
        FROM sampleset, binary_quadratic_model, graph
        WHERE
            graph.edges = :edges AND
            graph.num_nodes = :num_nodes AND
            binary_quadratic_model.graph_id = graph.id AND
            binary_quadratic_model.bqm_data = :bqm_data AND
            sampleset.num_variables = :num_variables AND
            sampleset.samples = :samples AND
            sampleset.energies = :energies;
        """

    insert_sampleset_statement = \
        """
        INSERT OR IGNORE INTO sampleset(
            num_variables,
            num_samples,
            samples,
            energies)
        VALUES (
            :num_variables,
            :num_samples,
            :samples,
            :energies);
        """

    database_name = f'penaltymodel_v{__version__}.db'
    database_path = homebase.user_data_dir(app_name='dwave-penaltymodel-cache',
                                           app_author='dwave-systems',
                                           create=True,
                                           )

    def __init__(self, database: Optional[Union[str, os.PathLike]] = None):
        if database is None:
            database = os.path.join(self.database_path, self.database_name)
        self.conn = conn = sqlite3.connect(database)

        # add the main schema
        conn.executescript(self.database_schema)

        # give us mapping access to values returned by .execute
        conn.row_factory = sqlite3.Row

    def __exit__(self, *args):
        # todo: make reentrant
        self.close()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    @staticmethod
    def encode_graph(graph_like: Union[GraphLike, dimod.BinaryQuadraticModel]
                     ) -> Dict[str, Union[int, str]]:
        """Encode a NetworkX graph or BQM to be stored in the cache."""
        if isinstance(graph_like, dimod.BinaryQuadraticModel):
            nodes = graph_like.linear.keys()
            edges = graph_like.quadratic.keys()
        else:
            graph = as_graph(graph_like)
            nodes = graph.nodes
            edges = graph.edges

        if nodes ^ range(len(nodes)):
            raise ValueError("nodes must be index-labelled")

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

    def insert_graph(self, graph_like: GraphLike):
        """Insert a graph into the database.

        Args:
            graph: a NetworkX graph, an integer or a list of nodes.

        Raises:
            ValueError: If the nodes of the graph are not labelled `[0, n)`.

        """
        with self.conn as cur:
            cur.execute(self.insert_graph_statement, self.encode_graph(graph_like))

    def iter_graphs(self) -> Iterator[nx.Graph]:
        """Iterate over all of the graphs in the database.

        Yields:
            All the graphs in the database, as NetworkX graphs.

        """
        yield from map(self.decode_graph, self.conn.execute("SELECT num_nodes, edges from graph;"))

    @staticmethod
    def encode_sampleset(samples_like) -> Dict[str, Union[int, str, bytes]]:
        samples, labels = dimod.as_samples(samples_like)

        if not all(i == v for i, v in enumerate(labels)):
            # we could mess with re-ordering but for now let's just do the
            # simple thing
            raise ValueError("sample labels must be labelled [0, n)")

        num_samples, num_variables = samples.shape

        if num_variables > 32:
            raise ValueError("sample set must have 32 or fewer variables")

        if isinstance(samples_like, dimod.SampleSet):
            energies = samples_like.record.energy
        else:
            energies = np.zeros(num_samples)

        order = np.lexsort(samples.transpose(), axis=0)
        samples = samples[order, :]
        energies = energies[order]

        packed = dimod.serialization.utils.pack_samples(samples > 0).flatten().tolist()

        return dict(
            num_variables=num_variables,
            num_samples=num_samples,
            samples=json.dumps(packed, separators=(',', ':')),
            energies=struct.pack('<' + 'd' * len(energies), *energies),
            )

    @staticmethod
    def decode_sampleset(row: Dict[str, Union[int, str, bytes]]) -> dimod.SampleSet:
        num_variables = row['num_variables']

        packed = np.atleast_2d(json.loads(row['samples'])).transpose()
        samples = dimod.serialization.utils.unpack_samples(packed, num_variables, dtype=np.int8)

        # convert to SPIN
        samples = 2*samples-1

        energies = struct.unpack('<' + 'd' * (len(row['energies']) // 8), row['energies'])

        return dimod.SampleSet.from_samples(samples, vartype='SPIN', energy=energies)

    def insert_sampleset(self, samples_like):
        """Insert a sample set into the database.

        Args:
            samples_like:
                Samples to add to the database.
                'samples_like' is an extension of NumPy's array_like_.
                See :func:`dimod.as_samples`.

        Raises:
            ValueError: If the variables are not labelled `[0, n)`.

        .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

        """
        with self.conn as cur:
            cur.execute(self.insert_sampleset_statement, self.encode_sampleset(samples_like))

    def iter_samplesets(self):
        """Iterate over all of the sample sets in the database."""
        select = "SELECT num_variables, num_samples, samples, energies FROM sampleset"
        yield from map(self.decode_sampleset, self.conn.execute(select))

    @staticmethod
    def encode_bqm(bqm: dimod.BinaryQuadraticModel) -> Dict[str, Union[float, bytes]]:
        with bqm.to_file() as f:
            return dict(
                max_quadratic_bias=bqm.quadratic.max(),
                min_quadratic_bias=bqm.quadratic.min(),
                max_linear_bias=bqm.linear.max(),
                min_linear_bias=bqm.linear.min(),
                bqm_data=f.read(),
                )

    @staticmethod
    def decode_bqm(row: Dict[str, Union[bytes, str, int]]) -> dimod.BinaryQuadraticModel:
        return dimod.BinaryQuadraticModel.from_file(row['bqm_data'])

    def insert_binary_quadratic_model(self, bqm: dimod.BinaryQuadraticModel):
        """Insert a binary quadratic model into the database.

        Args:
            bqm: A binary quadratic model.

        Raises:
            ValueError: If the variables of the binary quadratic model are not
                labelled `[0, n)`.

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

        parameters = self.encode_graph(bqm)
        parameters.update(self.encode_bqm(bqm))

        with self.conn as cur:
            cur.execute(self.insert_graph_statement, parameters)
            cur.execute(self.insert_bqm_statement, parameters)

    def iter_binary_quadratic_models(self) -> Iterator[dimod.BinaryQuadraticModel]:
        """Iterate over all of the binary quadratic models in the database."""
        for bqm_data in self.conn.execute("SELECT bqm_data FROM binary_quadratic_model;"):
            yield self.decode_bqm(bqm_data)

    def insert_penalty_model(
            self,
            bqm: dimod.BinaryQuadraticModel,
            samples_like,
            classical_gap: float,
            ):
        """Insert a penalty model into the database.

        Args:
            bqm: A binary quadratic model.

            samples_like: Samples to add to the database.
                'samples_like' is an extension of NumPy's array_like_.
                See :func:`dimod.as_samples`.

            classical_gap: The classical gap. This is not checked for
                correctness.

        .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

        """

        samples, decision = samples_like = dimod.as_samples(samples_like)

        # do some input checking
        if not all(v in bqm.variables for v in decision):
            raise ValueError("bqm's variables must be a superset of the "
                             "samples_like's variables")

        # we need the variables to be labelled [0, n) and for the decision
        # variables to be sorted
        if bqm.variables ^ range(bqm.num_variables) or any(i != v for i, v in enumerate(decision)):
            mapping = {v: i for i, v in enumerate(decision)}
            mapping.update((v, i) for i, v in enumerate(bqm.variables ^ decision, len(mapping)))

            return self.insert_penalty_model(bqm.relabel_variables(mapping, inplace=False), samples, classical_gap)

        parameters = self.encode_graph(bqm)
        parameters.update(self.encode_bqm(bqm))
        parameters.update(self.encode_sampleset(samples_like))
        parameters.update(
            decision_variables=json.dumps(decision, separators=(',', ':')),
            classical_gap=classical_gap,
            )

        with self.conn as cur:
            cur.execute(self.insert_graph_statement, parameters)
            cur.execute(self.insert_bqm_statement, parameters)
            cur.execute(self.insert_sampleset_statement, parameters)
            cur.execute(self.insert_penalty_model_statement, parameters)

    def iter_penalty_models(self) -> Iterator[PenaltyModel]:
        """Iterate over all of the penalty models in the database."""
        for row in self.conn.execute("SELECT * FROM penalty_model_view;"):
            yield PenaltyModel(
                    self.decode_bqm(row),
                    self.decode_sampleset(row),
                    row['classical_gap']
                )

    def retrieve(self,
                 samples_like,
                 graph_like,
                 *,
                 linear_bound: Tuple[float, float] = (-2, 2),
                 quadratic_bound: Tuple[float, float] = (-1, 1),
                 min_classical_gap: float = 2,
                 ) -> Tuple[dimod.BinaryQuadraticModel, float]:
        """Retrieve a penalty model from the database.

        Args:
            samples_like:
                The set of feasible states that form the ground states of the
                generated binary quadratic model.

                'samples_like' is an extension of NumPy's array_like_.
                See :func:`dimod.as_samples`.

                .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

            graph_like:
                Defines the structure of the desired binary quadratic model. Each
                node in the graph represents a variable and each edge defines an
                interaction between two variables.
                Can be given as a :class:`networkx.Graph`, a :class:`int`, or as
                a sequence of variable labels.

                If given as a sequence of labels, the structure will be
                fully-connected, with the variables labelled according to the
                sequence.

                If given as an int, the structure will be
                fully-connected with the variables labelled ``range(n)``.

                The nodes of the graph must be a superset of the labels of
                ``samples_like``.

                If not provided, defaults to a fully connected graph with nodes
                that are the variables of ``samples_like``.

            linear_bound:
                The range allowed for the linear biases of the binary quadratic
                model.

            quadratic_bound:
                The range allowed for the quadratic biases of the binary quadratic
                model.

            min_classical_gap:
                This is a threshold value for the classical gap. It describes the
                minimum energy gap between the highest feasible state and the
                lowest infeasible state.

        Returns:
            A 2-tuple of the binary quadratic model and the classical gap. Note
            that the binary quadratic model always has vartype ``'SPIN'``.

        """
        samples, labels = dimod.as_samples(samples_like)
        graph = as_graph(graph_like)

        # do some input checking
        if not all(v in graph.nodes for v in labels):
            raise ValueError("graph_like's nodes must be a superset of the "
                             "samples_like's variables")

        # we need the nodes/variables to be labelled [0, n). The variables
        # also need to be sorted
        if graph.nodes ^ range(len(graph.nodes)) or any(i != v for i, v in enumerate(labels)):
            mapping = {v: i for i, v in enumerate(labels)}
            mapping.update((v, i) for i, v in enumerate(graph.nodes ^ labels, len(mapping)))

            bqm, gap = self.retrieve(samples, nx.relabel_nodes(graph, mapping, copy=True),
                                     linear_bound=linear_bound,
                                     quadratic_bound=quadratic_bound,
                                     min_classical_gap=min_classical_gap)

            inverse_mapping = dict((i, v) for v, i in mapping.items())
            return bqm.relabel_variables(inverse_mapping, inplace=True), gap

        parameters = self.encode_graph(graph)
        parameters.update(self.encode_sampleset(samples_like))
        parameters.update(
            decision_variables=json.dumps(labels, separators=(',', ':')),
            min_classical_gap=min_classical_gap,
            min_linear_bias=linear_bound[0],
            max_linear_bias=linear_bound[1],
            min_quadratic_bias=quadratic_bound[0],
            max_quadratic_bias=quadratic_bound[1],
            )

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
                samples = :samples AND
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


def patch_cache(database: Union[str, os.PathLike] = ':memory:'):
    """A function decorator that passes in a PenaltyModelCache as a new argument.

    Args:
        database: The database location to use. Defaults to use memory.

    """
    def _patch(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with PenaltyModelCache(database) as cache:
                return f(*args, cache, **kwargs)
        return wrapper
    return _patch


@contextlib.contextmanager
def isolated_cache(*args, **kwargs):
    """Temporarily isolate the cache.

    Can be used as a decorator or a context manager.

    This context manager is not reentrant.

    """
    import sys

    if sys.version_info[:2] >= (3, 10):
        # Added in 3.10
        # We need ignore_cleanup_errors for Windows, it will still make a "best effort"
        # to remove the directory.
        kwarg = dict(ignore_cleanup_errors=True)
    else:
        kwarg = dict()

    with threading.RLock():
        with tempfile.TemporaryDirectory(**kwarg) as d:
            current = PenaltyModelCache.database_path
            PenaltyModelCache.database_path = d
            try:
                yield
            finally:
                PenaltyModelCache.database_path = current
