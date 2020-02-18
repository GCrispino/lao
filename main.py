from datetime import datetime
import numpy as np
import mdp_graph as mg
from lao import lao, ilao
from utils import read_json, output, parse_args

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
if args.output:
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_file_path = output(
        output_filename, {'V': V.tolist(), 'pi': pi.tolist()})
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
