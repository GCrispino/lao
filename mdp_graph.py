import numpy as np

# bpsg: best partial solution graph


def flatten(l):
    return [x for l_i in l for x in l_i]


def get_actions(mdp):
    adjs = map(lambda s: s['Adj'], mdp.values())
    actions = map(lambda s: list(s['A'].keys()), flatten(adjs))
    return list(set(flatten(actions)))


def get_unexpanded_states(mdp, bpsg):
    return list(
        filter(lambda x: not mdp[x]["expanded"]
               and not mdp[x]["goal"], bpsg.keys())
    )


def expand_state(s, mdp, explicit_graph):
    if mdp[s]['goal']:
        raise ValueError(
            'State %d can\'t be expanded because it is a goal state' % int(s))

    # Get 's' neighbour states that were not expanded
    neighbour_states = map(lambda _s: _s["name"], mdp[s]['Adj'])
    unexpanded_neighbours = filter(
        lambda _s: not mdp[_s]['expanded'], neighbour_states)

    # Add new empty states to 's' adjacency list
    new_explicit_graph = explicit_graph
    for n in unexpanded_neighbours:
        new_explicit_graph = add_state_graph(n, new_explicit_graph)
        mdp_n_obj = next(filter(lambda _s: _s["name"] == n, mdp[s]["Adj"]))
        new_explicit_graph[s]["Adj"].append({
            "name": n,
            "A": mdp_n_obj
        })

    mdp_ = mdp.copy()

    mdp_[s]['expanded'] = True

    return new_explicit_graph, mdp_


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


def bellman(V, V_i, A, Z, mdp, c=1):
    V_ = np.array(V)

    pi = np.array([None] * len(Z))
    for i, s in enumerate(Z):
        actions_results = []
        for a in A:
            reachable = find_reachable(s, a, mdp)
            c_ = 0 if mdp[s]['goal'] else c
            actions_results.append(c_ + sum([
                V[V_i[s_['name']]] * s_['A'][a] for s_ in reachable]))
        i_min = np.argmin(actions_results)
        pi[V_i[s]] = A[i_min]
        V_[V_i[s]] = actions_results[i_min]

    return V_, pi


def value_iteration(V, V_i, A, Z, mdp, c=1, epsilon=1e-3, n_iter=1000):

    i = 1
    pi = None
    while(True):
        V_, pi = bellman(V, V_i, A, Z, mdp, c)
        if i == n_iter or np.linalg.norm(V_ - V, np.inf) < epsilon:
            break
        V = V_

        i += 1

    return V_, pi


def update_action_partial_solution(s, a, bpsg, mdp):
    """
        Updates partial solution given pair of state and action
    """
    bpsg_ = bpsg.copy()
    s_obj = bpsg_[s]
    s_obj['Adj'] = []
    reachable = find_reachable(s, a, mdp)
    for s_obj_ in reachable:
        s_ = s_obj_["name"]
        s_obj['Adj'].append({
            'name': s_,
            'A': {a: s_obj_['A'][a]}
        })
        if s_ not in bpsg:
            bpsg_ = add_state_graph(s_, bpsg_)
    return bpsg_


def update_partial_solution(pi, S, bpsg, mdp):
    bpsg_ = bpsg.copy()

    for s, a in zip(S, pi):
        if s not in bpsg:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a != None:
                bpsg_ = update_action_partial_solution(s, a, bpsg_, mdp)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a != None and best_current_action != a:
                bpsg_ = update_action_partial_solution(s, a, bpsg_, mdp)

    return bpsg_
