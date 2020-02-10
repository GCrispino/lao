import numpy as np
import mdp_graph as mg
from lao import lao, ilao
from utils import read_json, parse_args

args = parse_args()

mdp = read_json(args.file_input)
A = mg.get_actions(mdp)
S = list(mdp.keys())
pi = np.array([None] * len(S))
V_i = {S[i]: i for i in range(len(S))}

heuristic = np.fromiter((map(lambda s: s['heuristic'], mdp.values())), float)
V = None

if args.algorithm == 'lao':
    V, pi = lao('1', heuristic, V_i, pi, S, A,
                mg.init_graph(mdp), epsilon=args.epsilon)
elif args.algorithm == 'ilao':
    V, pi = ilao('1', heuristic, V_i, pi, S, A,
                 mg.init_graph(mdp), epsilon=args.epsilon)

print('V: ', V)
print('pi: ', pi)
