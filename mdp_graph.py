# bpsg: best partial solution graph
def get_unexpanded_states(mdp, bpsg):
    return list(
        filter(lambda x: mdp[x]["expanded"], bpsg.keys())
    )


def expand_state(s, mdp, explicit_graph):
    pass


def init_graph(graph):
    return {k: {"expanded": False, **v} for k, v in graph.items()}


def add_state_graph(s, mdp):
    pass

# "I.e., only include ancestors states from which
#   the expanded state can be reached by following the current best solution"


def find_ancestors(s, mdp):
    pass


def value_iteration(V, mdp):
    pass
