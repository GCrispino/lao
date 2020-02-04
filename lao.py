import mdp_graph as mg


def lao(s0, heuristic, V_i, S, A, mdp):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = bpsg.copy()

    i = 0
    unexpanded = mg.get_unexpanded_states(mdp, bpsg)
    V = heuristic
    while len(unexpanded) > 0:
        s = unexpanded[0]
        explicit_graph, mdp = mg.expand_state(s, mdp, explicit_graph)
        Z = mg.find_ancestors(s, bpsg) + [s]
        V, pi = mg.value_iteration(V, V_i, A, Z, mdp)
        bpsg = mg.update_partial_solution(pi, S, bpsg, mdp)
        unexpanded = mg.get_unexpanded_states(mdp, bpsg)
        i += 1
    return V, pi
