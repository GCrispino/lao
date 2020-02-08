import pytest
import unittest
import numpy as np
import mdp_graph as mg
import lao
from unittest.mock import patch

graph = {
    "1": {
        "goal": False,
        "Adj": [
            {
                "name": "1",
                "A": {
                    "N": 1,
                    "S": 0.5,
                    "E": 0.5
                }
            },
            {
                "name": "2",
                "A": {"E": 0.5}
            },
            {
                "name": "4",
                "A": {"S": 0.5}
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
                    "S": 0.5,
                    "E": 0.5
                }
            },
            {
                "name": "3",
                "A": {"E": 0.5}
            },
            {
                "name": "5",
                "A": {"S": 0.5}
            }
        ]
    },
    "3": {
        "goal": True,
        "Adj": [
            {
                "name": "3",
                "A": {
                    "N": 1,
                    "S": 1,
                    "E": 1
                }
            }
        ]
    },
    "4": {
        "goal": False,
        "Adj": [
            {
                "name": "1",
                "A": {"N": 1}
            },
            {
                "name": "4",
                "A": {"S": 1}
            },
            {
                "name": "5",
                "A": {"E": 1}
            }
        ]
    },
    "5": {
        "goal": False,
        "Adj": [
            {
                "name": "2",
                "A": {"N": 1}
            },
            {
                "name": "5",
                "A": {"S": 1}
            },
            {
                "name": "6",
                "A": {"E": 1}
            }
        ]
    },
    "6": {
        "goal": False,
        "Adj": [
            {
                "name": "3",
                "A": {"N": 1}
            },
            {
                "name": "6",
                "A": {"S": 1, "E": 1}
            }
        ]
    }
}

A = ['N', 'S', 'E']
S = list(graph.keys())
V_i = {S[i]: i for i in range(len(S))}

ct = lao.convergence_test


def mocked_convergence_test(obj):
    def fn(V, V_i, pi, A, Z, mdp, c=1, epsilon=1e-3):
        if obj['i'] == 0:
            new_pi = np.array(['S', 'E', None, None, None, None])
            obj['i'] += 1
            return V, new_pi, False

        obj['i'] += 1
        return ct(V, V_i, pi, A, Z, mdp, c=c, epsilon=epsilon)
    return fn


class TestLAO(unittest.TestCase):
    def test_convergence_test(self):
        V = np.array([2, 1.5, 0, 1, 2, 1])
        pi = np.array(['S', 'E', None, 'E', 'E', 'N'])
        Z = ['1', '4', '5', '6']
        V_, pi_, converged = lao.convergence_test(V, V_i, pi, A, Z, graph)

        self.assertListEqual(pi_.tolist(), ['S', 'E', None, 'S', 'E', 'N'])
        assert not converged

    def test_lao(self):
        V = np.array([2, 1.5, 0, 1, 2, 1])
        pi = np.array([None] * len(S))
        V_, pi_ = lao.lao('1', V, V_i, pi, S, A, mg.init_graph(graph))
        self.assertListEqual(pi_.tolist(), ['E', 'E', None, 'E', None, None])

    def test_lao_2(self):
        V = np.array([2.0, 1, 0, 3, 2, 1])
        pi = np.array([None] * len(S))
        V_, pi_ = lao.lao('1', V, V_i, pi, S, A, mg.init_graph(graph))
        self.assertListEqual(pi_.tolist(), ['E', 'E', None, None, None, None])

    @patch('lao.convergence_test', new=mocked_convergence_test({'i': 0}))
    def test_lao_with_convergence_test(self):
        V = np.array([2.0, 1, 0, 3, 2, 1])
        pi = np.array([None] * len(S))
        V_, pi_ = lao.lao('1', V, V_i, pi, S, A, mg.init_graph(graph))

        self.assertListEqual(pi_.tolist(), ['E', 'E', None, 'E', None, None])
