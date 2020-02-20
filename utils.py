import argparse
import json
import os


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


DEFAULT_FILE_INPUT = './env-paper.json'
DEFAULT_INITIAL_STATE = '1'
DEFAULT_EPSILON = 1e-3
DEFAULT_GAMMA = 1.0
DEFAULT_ALGORITHM = 'lao'
DEFAULT_OUTPUT = False
DEFAULT_OUTPUT_DIR = "./output"


def output(output_filename, data, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_filename)

    with open(output_file_path, 'w') as fp:
        json.dump(data, fp, indent=2)

    return output_file_path


def parse_args():

    parser = argparse.ArgumentParser(
        description='LAO* algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--initial_state', dest='initial_state',
                        default=DEFAULT_INITIAL_STATE,
                        help="Initial state (default: %s)" % DEFAULT_INITIAL_STATE)
    parser.add_argument('--epsilon', dest='epsilon', type=float,
                        default=DEFAULT_EPSILON,
                        help="Epsilon used for convergence (default: %s)" % str(DEFAULT_EPSILON))
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=DEFAULT_GAMMA,
                        help="Discount factor (default: %s)" % str(DEFAULT_GAMMA))
    parser.add_argument('--algorithm', dest='algorithm', choices=['lao', 'ilao'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm to run (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)

    return parser.parse_args()
