import argparse
import numpy as np
import mdp_graph as mg
from lao import lao
from utils import read_json


DEFAULT_FILE_INPUT = './env-paper.json'
DEFAULT_EPSILON = 1e-3

parser = argparse.ArgumentParser(description='LAO* algorithm implementation.')

parser.add_argument('--file', dest='file_input',
                    default=DEFAULT_FILE_INPUT,
                    help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
parser.add_argument('--epsilon', dest='epsilon', type=float,
                    default=DEFAULT_EPSILON,
                    help="Epsilon used for convergence (default: %s)" % str(DEFAULT_EPSILON))

args = parser.parse_args()

mdp = read_json(args.file_input)
A = mg.get_actions(mdp)
S = list(mdp.keys())
pi = np.array([None] * len(S))
V_i = {S[i]: i for i in range(len(S))}

heuristic = np.fromiter((map(lambda s: s['heuristic'], mdp.values())), float)

V, pi = lao('1', heuristic, V_i, pi, S, A,
            mg.init_graph(mdp), epsilon=args.epsilon)

print('V: ', V)
print('pi: ', pi)
