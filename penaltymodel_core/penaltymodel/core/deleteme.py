from utilities import get_uniform_penaltymodel
from collections import defaultdict
import dimod
import dwavebinarycsp as dbc
import penaltymodel.core as pmc
import networkx as nx

def get_quadratic(linear_dict):
    linear_keys = list(linear_dict.keys())
    quadratic = defaultdict(int)

    for k0 in linear_keys:
        for k1 in linear_keys:
            a0, a1 = sorted([k0, k1])
            quadratic[(a0, a1)] += linear_dict[a0] * linear_dict[a1]
    return quadratic

terms = {"x1":1, "x2":1, "x3":1, "x4":1, "a1":-2, "a2":-2}
q = get_quadratic(terms)
print(q)
bqm = dimod.BinaryQuadraticModel.from_qubo(q)

g = nx.complete_graph(["x1", "x2", "x3", "x4", "a1", "a2"])
decision_variables = ["x1","x2","x3","x4"]
feasible_config = [(1,0,0,1),
                   (0,1,0,1),
                   (0,0,1,1),
                   (1,1,1,1)]
pm = pmc.PenaltyModel(g, decision_variables, feasible_config, dimod.BINARY, bqm, 2,0)
upm = get_uniform_penaltymodel(pm)

print(upm)


