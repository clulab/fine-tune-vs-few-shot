import argparse
import csv
from typing import List
import json
import re
import os
def postprocess_entities(text: str) -> set:
    """
    Extract entities between <output> and </output> tags.
    Returns a set of entities split by newlines.
    """
    # Try to find content between <output> and </output> tags
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Split by newlines and create a set of non-empty items
        entities = {item.strip() for item in content.split('\n') if item.strip()}
        return entities
    
    # If no tags found, return empty set
    return set()

def calculate_scores(data: List[dict]) -> tuple[dict, list]:
    """
    Calculate entity-level metrics 
    Returns tuple with (metrics, predictions)
    """
    tp = fp = fn = 0
    predictions = []

    for sample in data:
        # Process gold_answer and predicted_answer instead
        gold = postprocess_entities(sample['gold_answer'])
        predicted = postprocess_entities(sample['predicted_answer'])
        
        # Calculate sample-specific counts
        sample_tp = len(gold & predicted)
        sample_fp = len(predicted - gold)
        sample_fn = len(gold - predicted)
        
        # Calculate sample metrics using individual counts
        sample_precision = sample_tp / (sample_tp + sample_fp) if (sample_tp + sample_fp) > 0 else 0
        sample_recall = sample_tp / (sample_tp + sample_fn) if (sample_tp + sample_fn) > 0 else 0
        sample_f1 = 2 * (sample_precision * sample_recall) / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0
        
        # Accumulate totals for final metrics
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn

        # Collect predictions, adapting keys as needed
        prediction_entry = {
            'id': sample['id'],
            'doc_id': sample['doc_id'],
            'sent_idx': sample['sent_idx'],
            'question': sample['question'],
            'gold_answer': sample['gold_answer'],
            'predicted_answer': sample['predicted_answer'],
            'processed_gold_answer': list(gold),  # Convert set to list
            'processed_predicted_answer': list(predicted),  # Convert set to list
            'f1_score': sample_f1
        }
        

                
        predictions.append(prediction_entry)

    # Calculate standard metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }, predictions



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", type=str, default="source_preds.json", 
                       help="Path to the JSON file with predictions (default: source_preds.json)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of the model being evaluated")
    parser.add_argument("--source_dataset", type=str, required=True,
                       help="Source dataset being used (e.g., toy, dev, test)")
    parser.add_argument("--target_dataset", type=str, required=True,
                       help="Target dataset being used (e.g., toy, dev, test)")
    parser.add_argument("--eval_set", type=str, required=True,
                       help="Evaluation set being used (e.g., toy, dev, test)")
    parser.add_argument("--run_n", type=int, required=True,
                       help="Run number")
    parser.add_argument("--sample_n", type=int, required=True,
                       help="number of samples used for training or few shot")
    args = parser.parse_args()
    
    # read json file
    with open(args.generated_file, "r") as f:
        data = json.load(f)
        
    # trim model name after the last /
    args.model_name = args.model_name.split("/")[-1]
    
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
    with open(os.path.join(directory, "detailed_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
        
        # Update CSV fieldnames
    fieldnames = [
        'model_name', 'run_n', 'source_dataset', 'target_dataset', 'sample_n', 'eval_set',
        'precision', 'recall', 'f1'
    ]

    # Update CSV writing
    csv_file = f'output/{args.model_name}_{args.source_dataset}_{args.target_dataset}_all_run_scores.csv'
    # if file does not exist, write header
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    print("Writing to CSV")

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Removed header writing completely
        writer.writerow({
            'model_name': args.model_name,
            'run_n': args.run_n,
            'source_dataset': args.source_dataset,
            'target_dataset': args.target_dataset,
            'sample_n': args.sample_n,
            'eval_set': args.eval_set,
            **metrics
        })
    print("Writing to CSV done")
    
    