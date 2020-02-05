import json
import argparse
import mdp_graph as mg
import numpy as np
from pprint import pprint
from lao import lao


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


DEFAULT_FILE_INPUT = './env-paper.json'
DEFAULT_EPSILON = 1e-3

parser = argparse.ArgumentParser(description='LAO* algorithm implementation.')

parser.add_argument('--file', dest='file_input',
                    default=DEFAULT_FILE_INPUT, help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
parser.add_argument('--epsilon', dest='epsilon',
                    default=DEFAULT_EPSILON, help="Epsilon used for convergence (default: %s)" % str(DEFAULT_EPSILON))

args = parser.parse_args()

mdp = read_json(args.file_input)
A = mg.get_actions(mdp)
S = list(mdp.keys())
pi = np.array([None] * len(S))
V_i = {S[i]: i for i in range(len(S))}
print(A, S, V_i)

#heuristic = np.array([3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
heuristic = np.zeros(len(S))

#heuristic = np.array([4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0])

V, pi = lao('1', heuristic, V_i, pi, S, A,
            mg.init_graph(mdp), epsilon=args.epsilon)

print("V_i: ", V_i)
print('V: ', V)
print('pi: ', pi)
