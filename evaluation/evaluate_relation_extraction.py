import re
import argparse
import json
import numpy as np


RGX = re.compile(r"^([A-Z]+)\s+##\s+(.+)\s+##\s+([^\n]+)$")


def evaluate(gold, pred):
    total_g, total_p, total_i = (0, 0, 0)
    for g, p in zip(gold, pred):
        total_i += sum([any([i == j for j in g]) for i in p])
        total_g += len(g)
        total_p += len(p)
    precision = total_i / total_p if total_p != 0 else 0
    recall = total_i / total_g if total_g != 0 else 0
    f1 = 2 * precision * recall / (precision+recall) if precision+recall != 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall}


def post_process(predicted_answer, input_text):
    predicted_answer = predicted_answer.strip()
    if not predicted_answer.startswith("["):
        return []
    predicted_array = []
    try:
        predicted_array = json.loads(predicted_answer.replace("'", '"'))
    except Exception as e:
        try:
            predicted_array = json.loads("[" + predicted_answer.replace("'", '"'))
        except Exception as e:
            pass
    filtered_array = []
    for relation in predicted_array:
        if type(relation) != list or len(relation) != 3:
            continue
        if not all([type(r) == str for r in relation]):
            continue
        if relation[0] not in input_text or relation[2] not in input_text:
            continue
        filtered_array.append([relation[1], relation[0], relation[2]])
    return filtered_array

        
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
        pred.append(post_process(example["predicted_answer"], example["input"]))

    print(evaluate(gold, pred))

