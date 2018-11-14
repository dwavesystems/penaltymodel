import dwavebinarycsp as dbc

csp = dbc.ConstraintSatisfactionProblem(dbc.BINARY)
csp.add_constraint({(0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0)}, ('a', 'b', 'c'))  # or_gate

bqm = dbc.stitch(csp, min_classical_gap=2.0)
print("bqm: ", bqm)

