import numpy as np
import mdp_graph as mg
from copy import deepcopy


def lao(s0, heuristic, V_i, pi, S, A, mdp, epsilon=1e-3):
    """
        TODO:
            - implement convergence test
    """
    bpsg = {s0: {"Adj": []}}
    explicit_graph = deepcopy(bpsg)

    i = 0
    unexpanded = mg.get_unexpanded_states(mdp, bpsg)
    V = heuristic
    while True:
        while len(unexpanded) > 0:
            s = unexpanded[0]
            explicit_graph, mdp = mg.expand_state(s, mdp, explicit_graph)
            Z = mg.find_ancestors(s, bpsg) + [s]
            V, pi = mg.value_iteration(V, V_i, pi, A, Z, mdp, epsilon=epsilon)
            bpsg = mg.update_partial_solution(pi, S, bpsg, mdp)
            unexpanded = mg.get_unexpanded_states(mdp, bpsg)
            i += 1
        V, pi, converged = convergence_test(
            V, V_i, pi, A, Z, mdp, epsilon=epsilon)

        if converged:
            break
        else:
            bpsg = mg.update_partial_solution(pi, S, bpsg, mdp)
            unexpanded = mg.get_unexpanded_states(mdp, bpsg)
    return V, pi


def convergence_test(V, V_i, pi, A, Z, mdp, c=1, epsilon=1e-3):
    i = 0
    while(True):
        old_pi = np.array(pi)
        V_, pi = mg.bellman(V, V_i, pi, A, Z, mdp, c)
        pi_ = np.array(pi)
        different_actions = pi_[pi_ != old_pi]

        # converged
        if np.linalg.norm(V_ - V, np.inf) < epsilon:
            return V_, pi, True
        if len(different_actions) > 0:
            return V_, pi, False

        V = V_
        i += 1
