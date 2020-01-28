import mdp_graph
import unittest

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
