from itertools import product
import dwavebinarycsp as dbc
import networkx as nx
import unittest
import penaltymodel.lp as lp


#TODO: need to run without truth table
#TODO: test with binary values
class TestPenaltyModelLinearProgramming(unittest.TestCase):
    def verify_gate_bqm(self, bqm, nodes, get_gate_output, min_gap=2):
        """Check that valid gate inputs are at ground; invalid values meet threshold (min_gap) requirement
        """
        for a, b, c in product([-1, 1], repeat=3):
            spin_state = {nodes[0]: a, nodes[1]: b, nodes[2]: c}
            energy = bqm.energy(spin_state)

            if c == get_gate_output(a, b):
                self.assertEqual(energy, 0, "Failed for {}".format(spin_state))
            else:
                self.assertGreaterEqual(energy, min_gap, "Failed for {}".format(spin_state))

    def test_empty(self):
        with self.assertRaises(ValueError):
            result = lp.generate_bqm(nx.complete_graph([]), [], [])

    def test_dictionary_input(self):
        # Make or-gate BQM
        nodes = ['a', 'b', 'c']
        or_gate_values = {(-1, 1, 1): 0, (1, -1, 1): 0, (1, 1, 1): 0, (-1, -1, -1): 0}
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), or_gate_values, nodes)

        self.verify_gate_bqm(bqm, nodes, max)

    def test_set_input(self):
        # Generate BQM for a set
        nodes = [1, 2, 3]
        and_gate_set = {(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1)}
        bqm_set, gap_set = lp.generate_bqm(nx.complete_graph(nodes), and_gate_set, nodes)

        self.verify_gate_bqm(bqm_set, nodes, min)

    def test_list_input(self):
        # Generate BQM for a list
        nodes = [1, 2, 3]
        nand_gate_list = [(-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        bqm_list, gap_list = lp.generate_bqm(nx.complete_graph(nodes), nand_gate_list, nodes)

        self.verify_gate_bqm(bqm_list, nodes, lambda x, y: -1 * min(x, y))

    def test_multi_energy_bqm(self):
        # Create BQM for fully determined configuration with no ground states
        configurations = {(-1, -1): -.5, (-1, 1): 3.5, (1, -1): 1.5, (1, 1): 3.5}
        nodes = ['x', 'y']
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), configurations, nodes)

        # Verify BQM
        for (x, y), expected_energy in configurations.items():
            energy = bqm.energy({'x': x, 'y': y})
            self.assertEqual(energy, expected_energy, "Failed for x:{}, y:{}".format(x, y))

    def not_test_xor_gate_without_aux(self):
        min_gap = 2
        xor_gate_values = {(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)}

        # Make a BQM for an xor-gate
        # Note: this should not be possible without an auxiliary variable
        nodes = ['a', 'b', 'c']
        bqm, gap = lp.generate_bqm(nx.complete_graph(nodes), xor_gate_values, nodes)

        # Make a BQM for an and-gate
        csp = dbc.ConstraintSatisfactionProblem(dbc.SPIN)
        csp.add_constraint(xor_gate_values, ('a', 'b', 'c'))
        bqm = dbc.stitch(csp, min_classical_gap=min_gap)


if __name__ == "__main__":
    unittest.main()