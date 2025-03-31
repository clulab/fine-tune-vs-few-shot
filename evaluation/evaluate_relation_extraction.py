import re
import argparse
import json
import numpy as np


RGX = re.compile(r"^[A-Z]+\s+##\s+.+\s+##\s+.+")


def evaluate(gold, pred):

    total_g, total_p, total_i = (0, 0, 0)
    for g, p in zip(gold, pred):
        direct_g = [" ## ".join([i[0], i[1], i[2]]) for i in g]
        inverse_g = [" ## ".join([i[0], i[2], i[1]]) for i in g]
        total_i += len(set(direct_g).intersection(set(p)))
        total_i += len(set(inverse_g).intersection(set(p)))
        total_g += len(g)
        total_p += len(p)
    precision = total_i / total_p
    recall = total_i / total_g
    f1 = 2 * precision * recall / (precision+recall) if precision+recall != 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall}


def post_process(predicted_answer):
    relation_lines = []
    for line in predicted_answer.split("\n"):
        if line == "":
            break
        if RGX.match(line):
            relation_lines.append(line)
    return relation_lines    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", type=str)
    args = parser.parse_args()

    pred = []
    gold = []

    with open(args.generated_file, "r") as f:
        data = json.load(f)
        
    print("Number of examples = {}".format(len(data)))
    for example in data:
        gold.append(example["output"])
        pred.append(post_process(example["predicted_answer"]))

    print(evaluate(gold, pred))

