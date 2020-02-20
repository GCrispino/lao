from datetime import datetime
import numpy as np
import mdp_graph as mg
from lao import lao, ilao
from utils import read_json, output, parse_args


def try_int(key):
    try:
        return int(key)
    except:
        return key


args = parse_args()

mdp = mg.init_graph(read_json(args.file_input))
A = mg.get_actions(mdp)
S = sorted(mdp.keys(), key=try_int)
pi = np.array([None] * len(S))
V_i = {S[i]: i for i in range(len(S))}

heuristic = np.fromiter((map(lambda s: mdp[s]['heuristic'], S)), float)

V = None

if args.algorithm == 'lao':
    V, pi = lao(args.initial_state, heuristic, V_i, pi, S, A,
                mdp, epsilon=args.epsilon, gamma=args.gamma)
elif args.algorithm == 'ilao':
    V, pi = ilao(args.initial_state, heuristic, V_i, pi, S, A,
                 mdp, epsilon=args.epsilon, gamma=args.gamma)

print('V: ', V)
print('pi: ', pi)
if args.output:
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_file_path = output(
        output_filename, {'V': V.tolist(), 'pi': pi.tolist()})
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
