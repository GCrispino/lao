import argparse
import json


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


DEFAULT_FILE_INPUT = './env-paper.json'
DEFAULT_EPSILON = 1e-3
DEFAULT_ALGORITHM = 'lao'


def parse_args():

    parser = argparse.ArgumentParser(
        description='LAO* algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--epsilon', dest='epsilon', type=float,
                        default=DEFAULT_EPSILON,
                        help="Epsilon used for convergence (default: %s)" % str(DEFAULT_EPSILON))
    parser.add_argument('--algorithm', dest='algorithm', choices=['lao', 'ilao'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm to run (default: %s)" % DEFAULT_ALGORITHM)

    return parser.parse_args()
