# bpsg: best partial solution graph
def get_unexpanded_states(mdp, bpsg):
    return list(
        filter(lambda x: not mdp[x]["expanded"], bpsg.keys())
    )


def expand_state(s, mdp, explicit_graph):

    # Ver estados vizinhos de 's' no mdp que não foram expandidos
    neighbour_states = map(lambda _s: _s["name"], mdp[s]['Adj'])
    unexpanded_neighbours = filter(
        lambda _s: not mdp[_s]['expanded'], neighbour_states)

    # - Adicionar novos estados vazios na lista de adjacências do vértice 's'
    new_explicit_graph = explicit_graph
    for n in unexpanded_neighbours:
        new_explicit_graph = add_state_graph(n, new_explicit_graph)
        mdp_n_obj = next(filter(lambda _s: _s["name"] == n, mdp[s]["Adj"]))
        new_explicit_graph[s]["Adj"].append({
            "name": n,
            "A": mdp_n_obj
        })

    return new_explicit_graph


def init_graph(graph):
    return {k: {"expanded": False, **v} for k, v in graph.items()}


def add_state_graph(s, graph):
    graph_ = graph.copy()
    graph_[str(s)] = {'Adj': []}

    return graph_

# "I.e., only include ancestors states from which
#   the expanded state can be reached by following the current best solution"


def find_ancestors(s, mdp):
    pass


def value_iteration(V, mdp):
    pass
