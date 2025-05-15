import argparse
import json
import os
import csv


TEMPLATE_RGX = re.compile(r"<\|.*")

def update_source_preds(data, generated_file):
    # write to json file overwrite if exist
    os.makedirs(os.path.dirname(generated_file), exist_ok=True)
    with open(generated_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
    predicted_answer = TEMPLATE_RGX.sub('', predicted_answer)    
    if not predicted_answer.startswith("["):
        return []
    predicted_array = []
    try:
        predicted_array = json.loads(predicted_answer.replace("'", '"'))
    except Exception as e:
        try:
            predicted_array = json.loads("[" + predicted_answer.replace("'", '"'))
        except Exception as e:
            print("Error in post_process: {}".format(e))
            print("Predicted answer: {}".format(predicted_answer))
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
    # New arguments for score table
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the score table")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being evaluated")
    parser.add_argument("--eval_set", type=str, required=True, help="Evaluation set being used")
    parser.add_argument("--source_dataset", type=str, required=True, help="Source dataset")
    parser.add_argument("--target_dataset", type=str, required=True, help="Target dataset")
    parser.add_argument("--run_n", type=int, required=True, help="Run number")
    parser.add_argument("--sample_n", type=int, required=True, help="Number of samples used")
    parser.add_argument("--gen_config", type=str, required=True, help="Generation configuration identifier to include in the score table")
    args = parser.parse_args()

    gold = []
    pred = []

    with open(args.generated_file, "r") as f:
        data = json.load(f)
        
    print("Number of examples = {}".format(len(data)))
    for example in data:
        gold.append(example["output"])
        example["postprocessed_predicted_answer"] = post_process(example["predicted_answer"], example["input"])
        pred.append(example["postprocessed_predicted_answer"])
    update_source_preds(data, args.generated_file)
    scores = evaluate(gold, pred)
    print(scores)

    # --- CSV output logic ---
    SCORE_FIELDNAMES = [
        'model_name', 'run_n', 'source_dataset', 'target_dataset', 'gen_config', 'sample_n', 'eval_set',
        'precision', 'recall', 'f1'
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(
        args.output_dir,
        f"all_run_scores.csv"
    )

    # Write header if file does not exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SCORE_FIELDNAMES)
            writer.writeheader()

    # Write results
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDNAMES)
        writer.writerow({
            'model_name': args.model_name,
            'run_n': args.run_n,
            'source_dataset': args.source_dataset,
            'target_dataset': args.target_dataset,
            'gen_config': args.gen_config,
            'sample_n': args.sample_n,
            'eval_set': args.eval_set,
            'precision': round(scores['precision'], 4),
            'recall': round(scores['recall'], 4),
            'f1': round(scores['f1'], 4)
        })
    print("Writing to CSV done")
