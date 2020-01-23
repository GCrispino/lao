import json
import mdp_graph


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


def lao(s0, mdp):
    pass


file_input = './env-paper.json'
mdp = read_json(file_input)
# print(mdp)

lao(1, mdp)
