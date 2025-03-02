import argparse
import json
from sklearn.metrics import classification_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", type=str)
    args = parser.parse_args()

    label_map = {
        0: "no claim",
        1: "correlational",
        2: "conditional causal",
        3: "direct causal",
    }
    label_map_reverse = {v: k for k, v in label_map.items()}

    pred = []
    gold = []

    with open(args.generated_file, "r") as f:
        data = json.load(f)
    print("Number of examples = {}".format(len(data)))
    for example in data:
        gold_label = example["gold_answer"].lower().strip()
        pred_label = example["predicted_answer"].lower().strip()
        # print(f"gold: {gold_label}, pred: {pred_label}")
        gold.append(label_map_reverse[gold_label])
        pred.append(label_map_reverse.get(pred_label, 0))
    print(classification_report(gold, pred))
