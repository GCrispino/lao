import unittest
from copy import deepcopy
import numpy as np
import pytest
import mdp_graph
from utils import read_json

graph = {
    "1": {
        "goal": False,
        "Adj": [
            {
                "name": "1",
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": "2",
                "A": {"E": 0.5}
            }
        ]
    },
    "2": {
        "goal": False,
        "Adj": [
            {
                "name": "2",
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": "3",
                "A": {"E": 0.5}
            }
        ]
    },
    "3": {
        "goal": True,
        "Adj": [{
            "name": "3",
            "A": {
                "N": 1,
                "S": 1,
                "E": 1,
            }
        }]
    },
}

bpsg = {
    "1": {"Adj": []},
}

bpsg_2 = {
    "1": {
        "Adj": [
            {
                "name": "1",
                "A": {
                    "E": 0.5,
                }
            },
            {
                "name": "2",
                "A": {"E": 0.5}
            }
        ]
    },
    "2": {
        "Adj": [
            {
                "name": "2",
                "A": {
                    "E": 0.5,
                }
            },
            {
                "name": "3",
                "A": {"E": 0.5}
            }
        ]
    },
    "3": {
        "Adj": [{
            "name": "3",
            "A": {
                "N": 1,
            }
        }]
    },
}

bpsg_3 = {
    "1": {
        "Adj": [
            {
                "name": "1",
                "A": {
                    "E": 0.5,
                }
            },
            {
                "name": "2",
                "A": {"E": 0.5}
            }
        ]
    },
    "2": {
        "Adj": [
            {
                "name": "2",
                "A": {
                    "E": 0.5,
                }
            }
        ]
    },
    "3": {"Adj": []}
}

A = ['N', 'S', 'E']
S = list(graph.keys())
V_i = {S[i]: i for i in range(len(S))}

graph_env_1 = mdp_graph.init_graph(read_json('./env1.json'))


class TestMDPGraph(unittest.TestCase):
    def test_init_graph(self):
        mdp_g = mdp_graph.init_graph(graph)
        success = True
        for k in mdp_g:
            success &= not mdp_g[k]['expanded']
            state = deepcopy(mdp_g[k])
            state.pop('expanded')
            success &= state == graph[k]

        assert success, "Initializes the graph correctly"

    def test_get_actions(self):
        actions = mdp_graph.get_actions(graph)
        self.assertSetEqual(set(actions), set(['N', 'S', 'E']))

    def test_unexpanded_states_1(self):
        mdp_g = mdp_graph.init_graph(graph)

        unexpanded = mdp_graph.get_unexpanded_states(mdp_g, bpsg)
        self.assertListEqual(unexpanded, ['1'])

    def test_unexpanded_states_2(self):
        mdp_g = mdp_graph.init_graph(graph)
        # Expand states '1' and '2':
        explicit, mdp_g = mdp_graph.expand_state('1', mdp_g, bpsg)
        bpsg_, mdp_g = mdp_graph.expand_state('2', mdp_g, explicit)

        unexpanded = mdp_graph.get_unexpanded_states(mdp_g, bpsg_2)
        self.assertListEqual(unexpanded, [])

    def test_add_state_graph(self):
        g_ = mdp_graph.add_state_graph('4', graph)
        g__ = mdp_graph.add_state_graph(4, graph)

        assert '4' in g_
        g_.pop('4')
        assert g_ == graph
        assert '4' in g__
        g__.pop('4')
        assert g__ == graph

    def test_expand_state(self):
        init_state = '1'
        explicit_graph = mdp_graph.add_state_graph(init_state, {})
        mdp_g = mdp_graph.init_graph(graph)
        init_state_neighbours = map(
            lambda _s: _s["name"], mdp_g[init_state]['Adj'])
        new_explicit_graph, mdp_g = mdp_graph.expand_state(
            init_state, mdp_g, explicit_graph)

        assert mdp_g[init_state]['expanded']

        for s in init_state_neighbours:
            assert s in new_explicit_graph

    def test_expand_state_goal(self):
        state = '3'
        mdp_g = mdp_graph.init_graph(graph)
        with pytest.raises(
                ValueError,
                match="State %d can't be expanded because it is a goal state" % int(state)):

            _, mdp_g = mdp_graph.expand_state(
                state, mdp_g, {})

    def test_find_ancestors(self):
        ancestors = mdp_graph.find_ancestors('1', bpsg)

        self.assertListEqual(ancestors, [])

    def test_find_ancestors_2(self):
        # Test for second bpsg example
        ancestors = set(mdp_graph.find_ancestors('3', bpsg_2))
        self.assertSetEqual(ancestors, set(['1', '2']))

    def test_find_ancestors_3(self):
        # Test for third bpsg example
        ancestors = set(mdp_graph.find_ancestors('2', bpsg_3))
        self.assertSetEqual(ancestors, set(['1']))

    def test_find_reachable(self):
        reachable_1_n = mdp_graph.find_reachable('1', 'N', graph)
        reachable_1_e = mdp_graph.find_reachable('1', 'E', graph)
        reachable_2_e = mdp_graph.find_reachable('2', 'E', graph)
        reachable_2_s = mdp_graph.find_reachable('2', 'S', graph)

        self.assertListEqual(reachable_1_n, [{
            'name': '1',
            "A": {
                "N": 1,
                "S": 1,
                "E": 0.5,
            }
        }])
        self.assertListEqual(reachable_1_e, [
            {
                "name": "1",
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": "2",
                "A": {"E": 0.5}
            }
        ])
        self.assertListEqual(reachable_2_e, [
            {
                "name": "2",
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": "3",
                "A": {"E": 0.5}
            }
        ])
        self.assertListEqual(reachable_2_s, [{
            "name": "2",
            "A": {
                "N": 1,
                "S": 1,
                "E": 0.5,
            }
        }])

    def test_bellman(self):
        # Test bellman equation after expanding state '1'
        Z = ['1']
        V = np.array([2.0, 1.0, 0.0])
        pi = np.array([None] * len(S))
        V_, pi = mdp_graph.bellman(V, V_i, pi, A, Z, graph)

        self.assertListEqual(pi[:1].tolist(), ['E'])
        self.assertListEqual(V_.tolist(), [2.5, 1, 0])

    def test_bellman_2(self):
        # Test bellman equation after expanding state '2'
        Z = ['1', '2']

        V = np.array([3.0, 1.0, 0.0])
        pi = np.array([None] * len(S))
        V_, pi = mdp_graph.bellman(V, V_i, pi, A, Z, graph)

        self.assertListEqual(pi[:2].tolist(), ['E', 'E'])
        self.assertListEqual(V_.tolist(), [3, 1.5, 0])

    def test_value_iteration(self):
        Z = ['1']
        V = np.array([2.0, 1.0, 0.0])
        pi = np.array([None] * len(S))

        epsilon = 1e-3
        V_, pi = mdp_graph.value_iteration(V, V_i, pi, A, Z, graph, 1, epsilon)
        expected = np.array([3.0, 1.0, 0.0])
        arrays_diff = np.linalg.norm(V_ - expected, np.inf)

        self.assertListEqual(pi.tolist(), ['E', None, None])
        assert arrays_diff < epsilon

    def test_value_iteration_2(self):
        Z = ['1', '2']
        V = np.array([3.0, 1.0, 0.0])
        pi = np.array([None] * len(S))

        epsilon = 1e-3
        V_, pi = mdp_graph.value_iteration(V, V_i, pi, A, Z, graph, 1, epsilon)
        expected = np.array([4.0, 2.0, 0.0])
        arrays_diff = np.linalg.norm(V_ - expected, np.inf)

        self.assertListEqual(pi.tolist(), ['E', 'E', None])
        assert arrays_diff < epsilon

    def test_update_partial_solution(self):
        pi = ['E']

        new_bpsg = mdp_graph.update_partial_solution(pi, S, bpsg, graph)

        self.assertDictEqual(new_bpsg, {
            "1": {
                "Adj": [
                    {
                        "name": "1",
                        "A": {"E": 0.5}
                    },
                    {
                        "name": "2",
                        "A": {"E": 0.5}
                    }
                ]
            },
            "2": {
                "Adj": []
            },
        })

    def test_update_partial_solution_changes(self):
        expanded_states = ['1', '2', '3', '4']

        mdp = deepcopy(graph_env_1)
        # mark states as expanded
        for s in expanded_states:
            mdp[s]['expanded'] = True

        S_ = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        Z = ['3', '2', '1', '4']
        V = [6.99999999, 5.99882507, 3.99986267, 1.99999237,
             0., 5., 4., 3., 2., 1.]
        pi = ['S', 'E', 'E', 'E', None, None, None, None, None, None]
        bpsg_ = {'1': {'Adj': [{'A': {'E': 0.5}, 'name': '1'}, {'A': {'E': 0.5}, 'name': '2'}]},
                 '2': {'Adj': [{'A': {'E': 0.5}, 'name': '2'}, {'A': {'E': 0.5}, 'name': '3'}]},
                 '3': {'Adj': [{'A': {'E': 0.5}, 'name': '3'}, {'A': {'E': 0.5}, 'name': '4'}]},
                 '4': {'Adj': []}}

        new_bpsg = mdp_graph.update_partial_solution(pi, S_, bpsg_, mdp)

        self.assertDictEqual(new_bpsg, {
            '1': {'Adj': [{'A': {'S': 0.5}, 'name': '1'}, {'A': {'S': 0.5}, 'name': '6'}]},
            '6': {'Adj': []}

        })
