import argparse
import csv
import json
import re
import os
from collections import Counter

def extract_label(text: str) -> str:
    """
    Extract the label between <output> and </output> tags.
    Returns the label as a string, stripped of whitespace.
    """
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def calculate_scores(data):
    """
    Calculate accuracy and per-class accuracy.
    Returns (metrics, predictions)
    """
    correct = 0
    total = 0
    gold_labels = []
    pred_labels = []
    predictions = []

    for sample in data:
        gold = extract_label(sample['gold_answer'])
        pred = extract_label(sample['predicted_answer'])
        gold_labels.append(gold)
        pred_labels.append(pred)
        is_correct = int(gold == pred)
        correct += is_correct
        total += 1

        predictions.append({
            'id': sample['id'],
            'question': sample['question'],
            'gold_answer': sample['gold_answer'],
            'predicted_answer': sample['predicted_answer'],
            'processed_gold_label': gold,
            'processed_predicted_label': pred,
            'is_correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    for g, p in zip(gold_labels, pred_labels):
        class_total[g] += 1
        if g == p:
            class_correct[g] += 1
    per_class_accuracy = {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in class_total}

    return {
        'accuracy': round(accuracy, 4),
        'per_class_accuracy': {k: round(v, 4) for k, v in per_class_accuracy.items()}
    }, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", type=str, default="source_preds.json", 
                       help="Path to the JSON file with predictions (default: source_preds.json)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of the model being evaluated")
    parser.add_argument("--source_dataset", type=str, required=True,
                       help="Source dataset being used (e.g., train, test)")
    parser.add_argument("--target_dataset", type=str, required=True,
                       help="Target dataset being used (e.g., train, test)")
    parser.add_argument("--eval_set", type=str, required=True,
                       help="Evaluation set being used (e.g., train, test)")
    parser.add_argument("--run_n", type=int, required=True,
                       help="Run number")
    parser.add_argument("--sample_n", type=int, required=True,
                       help="Number of samples used for training or few shot")
    args = parser.parse_args()
    
    # trim model name after the last /
    args.model_name = args.model_name.split("/")[-1]
    
    # read json file
    with open(args.generated_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # evaluate
    metrics, predictions = calculate_scores(data)
    
    # Print model name, evaluation set, and scores
    print(f"Model: {args.model_name}")
    print(f"Source Dataset: {args.source_dataset}")
    print(f"Target Dataset: {args.target_dataset}")
    print(f"Evaluation Set: {args.eval_set}")
    print(f"Scores: {metrics}")
    # get the directory from the generated file
    directory = os.path.dirname(args.generated_file)
    with open(os.path.join(directory, "detailed_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
        
    # CSV fieldnames
    fieldnames = (
        [ 'model_name', 'run_n', 'source_dataset', 'target_dataset', 'sample_n', 'eval_set' ]
        + [f'per_class_{k}' for k in metrics['per_class_accuracy'].keys()]
        + ['accuracy']
    )

    # Prepare row for CSV
    row = {
        'model_name': args.model_name,
        'run_n': args.run_n,
        'source_dataset': args.source_dataset,
        'target_dataset': args.target_dataset,
        'sample_n': args.sample_n,
        'eval_set': args.eval_set,
    }
    for k, v in metrics['per_class_accuracy'].items():
        row[f'per_class_{k}'] = v
    row['accuracy'] = metrics['accuracy']
    csv_file = f'output/{args.model_name}_{args.source_dataset}_{args.target_dataset}_all_run_scores.csv'
    # if file does not exist, write header
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print("Writing to CSV done")
