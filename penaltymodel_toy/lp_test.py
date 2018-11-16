from itertools import product
import dwavebinarycsp as dbc
import unittest


#TODO: need to run without truth table
class TestLinearProgramming(unittest.TestCase):
    def test_or_gate_binary(self):
        min_gap = 2
        or_gate_values = {(0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0)}

        # Make a BQM for an or-gate
        csp = dbc.ConstraintSatisfactionProblem(dbc.BINARY)
        csp.add_constraint(or_gate_values, ('a', 'b', 'c'))
        bqm = dbc.stitch(csp, min_classical_gap=min_gap)

        # Check that valid or-gate inputs are at ground; invalid values meet threshold requirement
        ground_energy = bqm.energy({'a': 0, 'b': 0, 'c': 0})    # Valid or-gate value
        for a, b, c in product([0, 1], repeat=3):
            energy = bqm.energy({'a': a, 'b': b, 'c': c})

            if c == (a or b):
                # Or-gate values
                self.assertEqual(energy, ground_energy)
            else:
                # Non-or-gate values
                self.assertGreaterEqual(energy, ground_energy + min_gap, "Failed for a:{}, b:{}, c:{}".format(a, b, c))

    def test_and_gate_ising(self):
        min_gap = 2
        and_gate_values = {(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1)}

        # Make a BQM for an and-gate
        csp = dbc.ConstraintSatisfactionProblem(dbc.SPIN)
        csp.add_constraint(and_gate_values, ('a', 'b', 'c'))
        bqm = dbc.stitch(csp, min_classical_gap=min_gap)

        # Check that valid or-gate inputs are at ground; invalid values meet threshold requirement
        ground_energy = bqm.energy({'a': -1, 'b': -1, 'c': -1})    # Valid and-gate value
        for a, b, c in product([-1, 1], repeat=3):
            energy = bqm.energy({'a': a, 'b': b, 'c': c})

            if c == min(a, b):
                # and-gate values
                self.assertEqual(energy, ground_energy)
            else:
                # and-or-gate values
                self.assertGreaterEqual(energy, ground_energy + min_gap, "Failed for a:{}, b:{}, c:{}".format(a, b, c))

    def test_xor_gate_without_aux(self):
        min_gap = 2
        xor_gate_values = {(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)}

        # Make a BQM for an and-gate
        csp = dbc.ConstraintSatisfactionProblem(dbc.SPIN)
        csp.add_constraint(xor_gate_values, ('a', 'b', 'c'))
        bqm = dbc.stitch(csp, min_classical_gap=min_gap)



if __name__ == "__main__":
    unittest.main()
