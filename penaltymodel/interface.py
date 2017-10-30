from penaltymodel.plugins import factories

def get_penalty_model(graph, decision_variables, constraint):
    for factory in factories:
        pm = factory(graph, decision_variables, constraint)

        print(pm)

