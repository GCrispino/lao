import mdp_graph as mg


def lao(s0, heuristic, V_i, pi, S, A, mdp, epsilon=1e-3):
    """
        TODO:
            - implement convergence test
    """
    bpsg = {s0: {"Adj": []}}
    explicit_graph = bpsg.copy()

    i = 0
    unexpanded = mg.get_unexpanded_states(mdp, bpsg)
    V = heuristic
    while len(unexpanded) > 0:
        s = unexpanded[0]
        explicit_graph, mdp = mg.expand_state(s, mdp, explicit_graph)
        Z = mg.find_ancestors(s, bpsg) + [s]
        V, pi = mg.value_iteration(V, V_i, pi, A, Z, mdp, epsilon=epsilon)
        bpsg = mg.update_partial_solution(pi, S, bpsg, mdp)
        unexpanded = mg.get_unexpanded_states(mdp, bpsg)
        i += 1
    return V, pi
