import unittest
import itertools

import networkx as nx
import penaltymodel as pm

import penaltymodel_maxgap as maxgap


class TestInterface(unittest.TestCase):
    """We assume that the generation code works correctly.
    Test that the interface gives a penalty model corresponding to the specification"""
    def test_typical(self):
        graph = nx.complete_graph(3)
        spec = pm.Specification(graph, [0, 1], {(-1, -1): 0, (+1, +1): 0}, pm.SPIN)

        widget = maxgap.get_penalty_model(spec)

        # some quick test to see that the penalty model propogated in
        for v in graph:
            self.assertIn(v, widget.model.linear)
        for (u, v) in graph.edges:
            self.assertIn(u, widget.model.adj[v])

    def test_binary_specification(self):
        graph = nx.Graph()
        for i in range(4):
            for j in range(4, 8):
                graph.add_edge(i, j)

        decision_variables = (0, 1)
        feasible_configurations = ((0, 0), (1, 1))  # equality

        spec = pm.Specification(graph, decision_variables, feasible_configurations, vartype=pm.BINARY)
        widget = maxgap.get_penalty_model(spec)

        self.assertIs(widget.model.vartype, pm.BINARY)

        # test the correctness of the widget
        energies = {}
        for decision_config in itertools.product((0, 1), repeat=2):
            energies[decision_config] = float('inf')

            for aux_config in itertools.product((0, 1), repeat=6):
                sample = dict(enumerate(decision_config + aux_config))
                energy = widget.model.energy(sample)

                energies[decision_config] = min(energies[decision_config], energy)

        for decision_config, energy in energies.items():
            if decision_config in feasible_configurations:
                self.assertAlmostEqual(energy, widget.ground_energy)
            else:
                self.assertGreaterEqual(energy, widget.ground_energy + widget.classical_gap - 10**-6)

    def test_and_on_k44(self):
        graph = nx.Graph()
        for i in range(4):
            for j in range(4, 8):
                graph.add_edge(i, j)

        decision_variables = (0, 2, 3)
        feasible_configurations = AND(2)

        mapping = {0: '0', 1: '1', 2: '2', 3: '3'}
        graph = nx.relabel_nodes(graph, mapping)
        decision_variables = tuple(mapping[x] for x in decision_variables)

        spin_configurations = tuple([tuple([2 * i - 1 for i in b]) for b in feasible_configurations])
        spec = pm.Specification(graph, decision_variables, spin_configurations, vartype=pm.SPIN)

        pm0 = maxgap.get_penalty_model(spec)
        self.check_generated_ising_model(pm0.feasible_configurations, pm0.decision_variables,
                                         pm0.model.linear, pm0.model.quadratic, pm0.ground_energy,
                                         pm0.classical_gap)

    def check_generated_ising_model(self, feasible_configurations, decision_variables,
                                    linear, quadratic, ground_energy, infeasible_gap):
        """Check that the given Ising model has the correct energy levels"""
        if not feasible_configurations:
            return

        from dimod import ExactSolver

        samples = ExactSolver().sample_ising(linear, quadratic)

        # samples are returned in order of energy
        sample, ground = next(iter(samples.items()))
        gap = float('inf')

        self.assertIn(tuple(sample[v] for v in decision_variables), feasible_configurations)

        seen_configs = set()

        for sample, energy in samples.items():
            config = tuple(sample[v] for v in decision_variables)

            # we want the minimum energy for each config of the decisison variables,
            # so once we've seen it once we can skip
            if config in seen_configs:
                continue

            if config in feasible_configurations:
                self.assertAlmostEqual(energy, ground)
                seen_configs.add(config)
            else:
                gap = min(gap, energy - ground)

        self.assertAlmostEqual(ground_energy, ground)
        self.assertAlmostEqual(gap, infeasible_gap)


def AND(n):
    # AND of n inputs
    feas = list()
    for x in itertools.product((0, 1), repeat=n):
        y = int(all(x))
        feas.append(x + (y,))
    return tuple(feas)
