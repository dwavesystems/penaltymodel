import unittest
import random
import itertools

import networkx as nx

import penaltymodel as pm


class TestBinaryQuadraticModel(unittest.TestCase):
    def test_construction_typical(self):

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.SPIN)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.BINARY)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

    def test_input_checking_vartype(self):
        """Check that exceptions get thrown for broken inputs"""

        # this biases values are themselves not important, so just choose them randomly
        linear = {v: random.uniform(-2, 2) for v in range(10)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()

        with self.assertRaises(TypeError):
            pm.BinaryQuadraticModel(linear, quadratic, offset, 147)

        with self.assertRaises(TypeError):
            pm.BinaryQuadraticModel(linear, quadratic, offset, 'my made up type')

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, pm.BINARY).vartype, pm.BINARY)

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, -1).vartype, pm.SPIN)

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, 'BINARY').vartype, pm.BINARY)

    def test_input_checking_linear(self):
        """linear should be a dict."""
        linear = {v: random.uniform(-2, 2) for v in range(10)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = pm.SPIN

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, pm.BINARY).linear, linear)

        with self.assertRaises(TypeError):
            pm.BinaryQuadraticModel(list(linear.values()), quadratic, offset, pm.BINARY)

    def test_input_checking_quadratic(self):
        linear = {v: random.uniform(-2, 2) for v in range(11)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = pm.SPIN

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, pm.BINARY).quadratic, quadratic)

        # quadratic should be a dict
        with self.assertRaises(TypeError):
            pm.BinaryQuadraticModel(linear, [], offset, pm.BINARY)

        # unknown varialbe (vars must be in linear)
        with self.assertRaises(ValueError):
            pm.BinaryQuadraticModel(linear, {('a', 1): .5}, offset, pm.BINARY)

        # not 2-tuple
        with self.assertRaises(ValueError):
            pm.BinaryQuadraticModel(linear, {'edge': .5}, offset, pm.BINARY)

        # not upper triangular
        with self.assertRaises(ValueError):
            pm.BinaryQuadraticModel(linear, {(0, 1): .5, (1, 0): -.5}, offset, pm.BINARY)

        # no self-loops
        with self.assertRaises(ValueError):
            pm.BinaryQuadraticModel(linear, {(0, 0): .5}, offset, pm.BINARY)

    def test__eq__(self):
        linear = {v: random.uniform(-2, 2) for v in range(11)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = pm.SPIN

        self.assertEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, vartype),
                         pm.BinaryQuadraticModel(linear, quadratic, offset, vartype))

        # mismatched type
        self.assertNotEqual(pm.BinaryQuadraticModel(linear, quadratic, offset, vartype), -1)

        self.assertNotEqual(pm.BinaryQuadraticModel({}, {}, 0.0, pm.SPIN),
                            pm.BinaryQuadraticModel({}, {}, 0.0, pm.BINARY))

    def test_as_qubo_binary_to_qubo(self):
        """Binary model's as_qubo method"""
        linear = {0: 0, 1: 0}
        quadratic = {(0, 1): 1}
        offset = 0.0
        vartype = pm.BINARY

        model = pm.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = model.as_qubo()

        self.assertEqual(off, offset)
        self.assertEqual({(0, 0): 0, (1, 1): 0, (0, 1): 1}, Q)

    def test_as_qubo_spin_to_qubo(self):
        """Spin model's as_qubo method"""
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = pm.SPIN

        model = pm.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = model.as_qubo()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in Q.items():
                energy += bin_sample[u] * bin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(spin_sample))

    def test_as_ising_spin_to_ising(self):
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = pm.SPIN

        model = pm.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        h, J, off = model.as_ising()

        self.assertEqual(off, offset)
        self.assertEqual(linear, h)
        self.assertEqual(quadratic, J)

    def test_as_ising_binary_to_ising(self):
        """binary model's as_ising method"""
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = pm.BINARY

        model = pm.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        h, J, off = model.as_ising()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in J.items():
                energy += spin_sample[u] * spin_sample[v] * bias
            for v, bias in h.items():
                energy += spin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(bin_sample))

    def test__repr__(self):
        """check that repr works correctly."""
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4

        m = pm.BinaryQuadraticModel(linear, quadratic, offset, pm.SPIN)

        # should recreate the model
        from penaltymodel import BinaryQuadraticModel
        m2 = eval(m.__repr__())

        self.assertEqual(m, m2)


class TestSpecification(unittest.TestCase):
    def test_serialize(self):
        spec = pm.Specification(nx.complete_graph(2), [0], {(1, 1), (-1, -1)})
        serial = spec.serialize()

        #
        # todo
        #


class TestPenaltyModel(unittest.TestCase):
    pass
