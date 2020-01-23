import mdp_graph
import unittest

graph = {
    "1": {
        "Adj": [
            {
                "name": 1,
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": 2,
                "A": {"E": 0.5}
            }
        ]
    },
    "2": {
        "Adj": [
            {
                "name": 2,
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 0.5,
                }
            },
            {
                "name": 3,
                "A": {"E": 0.5}
            }
        ]
    },
    "3": {
        "Adj": [{
            "name": 3,
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

    # def test_unexpanded_states_2(self):
    #    mdp_g = mdp_graph.init_graph(graph)
    #    mdp_g['1']['expanded'] = True

    #    unexpanded = mdp_graph.get_unexpanded_states(mdp_g, bpsg)
    #    self.assertListEqual(unexpanded, ['1'])
