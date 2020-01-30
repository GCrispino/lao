import mdp_graph
import unittest
import numpy as np

graph = {
    "1": {
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
    }
}

A = ['N', 'S', 'E']
S = list(graph.keys())
V_i = {S[i]: i for i in range(len(S))}


class TestMDPGraph(unittest.TestCase):
    def test_init_graph(self):
        mdp_g = mdp_graph.init_graph(graph)
        success = True
        for k in mdp_g:
            success &= mdp_g[k]['expanded'] == False
            state = mdp_g[k].copy()
            state.pop('expanded')
            success &= state == graph[k]

        assert success, "Initializes the graph correctly"

    def test_unexpanded_states_1(self):
        mdp_g = mdp_graph.init_graph(graph)

        unexpanded = mdp_graph.get_unexpanded_states(mdp_g, bpsg)
        self.assertListEqual(unexpanded, ['1'])

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
        new_explicit_graph = mdp_graph.expand_state(
            init_state, mdp_g, explicit_graph)

        for s in init_state_neighbours:
            assert s in new_explicit_graph

        init_state_new_neighbours_explicit = map(
            lambda s: s["name"], new_explicit_graph[init_state]['Adj'])

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
        V_ = mdp_graph.bellman(V, V_i, A, Z, graph)

        self.assertListEqual(V_.tolist(), [2.5, 1, 0])

    def test_bellman_2(self):
        # Test bellman equation after expanding state '2'
        Z = ['1', '2']

        V = np.array([3.0, 1.0, 0.0])
        V_ = mdp_graph.bellman(V, V_i, A, Z, graph)

        self.assertListEqual(V_.tolist(), [3, 1.5, 0])

