import argparse
import json
from sklearn.metrics import f1_score, precision_score, recall_score


def negation_performance(labels_true, labels_pred, average=None):
    f1 = f1_score(y_true=labels_true, y_pred=labels_pred, average=average)
    precision = precision_score(y_true=labels_true, y_pred=labels_pred, average=average)
    recall = recall_score(y_true=labels_true, y_pred=labels_pred, average=average)

    return {"f1": f1, "precision": precision, "recall": recall}


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
        gold.append(1 if example["gold_answer"].lower().strip() == "yes" else 0)
        pred.append(1 if "yes" in example["predicted_answer"].lower().strip() else 0)

    print(negation_performance(gold, pred))
