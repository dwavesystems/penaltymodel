from itertools import product
import dwavebinarycsp as dbc
import networkx as nx
import unittest
import penaltymodel.lp as lp


#TODO: need to run without truth table
#TODO: test with binary values
class TestLinearProgramming(unittest.TestCase):
    def test_empty(self):
        pass

    def test_dictionary_inputs(self):
        min_gap = 2
        or_gate_values = {(-1, 1, 1): 0, (1, -1, 1): 0, (1, 1, 1): 0, (-1, -1, -1): 0}

        # Make a BQM for an or-gate
        nodes = ['a', 'b', 'c']
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), or_gate_values, nodes)

        # Check that valid or-gate inputs are at ground; invalid values meet threshold requirement
        ground_energy = bqm.energy({'a': -1, 'b': -1, 'c': -1})    # Valid or-gate value
        for a, b, c in product([-1, 1], repeat=3):
            energy = bqm.energy({'a': a, 'b': b, 'c': c})

            if c == max(a, b):
                # Or-gate values
                self.assertEqual(energy, ground_energy, "Failed for a:{}, b:{}, c:{}".format(a, b, c))
            else:
                # Non-or-gate values
                self.assertGreaterEqual(energy, ground_energy + min_gap, "Failed for a:{}, b:{}, c:{}".format(a, b, c))

    def test_iterable_inputs(self):
        min_gap = 2
        and_gate_set = {(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1)}
        nodes = [1, 2, 3]
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), and_gate_set, nodes)

    def test_multi_energy_bqm(self):
        configurations = {(-1, -1): -.5, (-1, 1): 3.5, (1, -1): 1.5, (1, 1): 3.5}
        nodes = ['x', 'y']

        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), configurations, nodes)

        # Verify bqm
        for (x, y), expected_energy in configurations.items():
            energy = bqm.energy({'x': x, 'y': y})
            self.assertEqual(energy, expected_energy, "Failed for x:{}, y:{}".format(x, y))

    def not_test_xor_gate_without_aux(self):
        min_gap = 2
        xor_gate_values = {(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)}

        # Make a BQM for an xor-gate
        # Note: this should not be possible with an auxiliary variable
        nodes = ['a', 'b', 'c']
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), xor_gate_values, nodes)

        # Make a BQM for an and-gate
        csp = dbc.ConstraintSatisfactionProblem(dbc.SPIN)
        csp.add_constraint(xor_gate_values, ('a', 'b', 'c'))
        bqm = dbc.stitch(csp, min_classical_gap=min_gap)


if __name__ == "__main__":
    unittest.main()