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


def find_ancestors(s, bpsg):
    # Find states in bpsg that have 's' in the adjacent list (except from 's' itself):
    direct_ancestors = list(
        filter(lambda s_: s_ != s and len(
            list(filter(lambda s__: s__['name'] == s, bpsg[s_]['Adj']))
        ) > 0, bpsg)
    )

    result = [] + direct_ancestors

    for a in direct_ancestors:
        result += find_ancestors(a, bpsg)

    return result


def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(
        lambda obj_s_: a in obj_s_['A'],
        all_reachable_from_s
    ))


def bellman(V, V_i,  A, Z, mdp, c=1):
    V_ = V.copy()
    for i, s in enumerate(Z):
        actions_results = []
        for a in A:
            reachable = find_reachable(s, a, mdp)
            actions_results.append(c + sum([
                V[V_i[s_['name']]] * s_['A'][a] for s_ in reachable]))
        V_[i] = min(actions_results)

    return V_


def value_iteration(V, A, Z, mdp, c=1, epsilon=1e-3):
    pass
