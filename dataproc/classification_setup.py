import random
import json
from pathlib import Path
from datasets import load_dataset

random.seed(31)

def convert_to_io_samples(name, split, data, instruction):
    samples = []
    for s in data:
        sample_id = f"{name}_{split}_{s['id']}"
        question = f"<instruction>{instruction}</instruction>\n<input>{s['question']}</input>\n"
        gold_answer = f"<output>{s['gold_answer']}</output>"
        samples.append({
            "id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "original_text": s["question"],
            "original_label": s["gold_answer"]
        })
    return samples

def get_one_per_label(samples, label_list):
    label_to_id = {}
    for sample in samples:
        label = sample["original_label"]
        if label in label_list and label not in label_to_id:
            label_to_id[label] = sample["id"]
        if len(label_to_id) == len(label_list):
            break
    return list(label_to_id.values())

class AGNewsDataset:
    def __init__(self):
        self.name = "ag_news"
        self.label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        self.data = {
            'train': [],
            'test': []
        }
        self.instruction = (
            "Determine topic of the sentence using following options: World, Sports, Business, or Sci/Tech."
        )
        self.labels = list(self.label_map.values())

    def load(self):
        agnews = load_dataset("ag_news")
        self.data['train'] = [
            {
                "id": str(i),
                "question": s["text"],
                "gold_answer": self.label_map[s["label"]]
            }
            for i, s in enumerate(agnews["train"])
        ]
        self.data['test'] = [
            {
                "id": str(i),
                "question": s["text"],
                "gold_answer": self.label_map[s["label"]]
            }
            for i, s in enumerate(agnews["test"])
        ]

class SnipsDataset:
    def __init__(self):
        self.name = "snips"
        self.data = {
            'train': [],
            'test': []
        }
        self.instruction = (
            "Determine intent of the sentence using following options: AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, or SearchScreeningEvent."
        )
        self.labels = [
            "AddToPlaylist",
            "BookRestaurant",
            "GetWeather",
            "PlayMusic",
            "RateBook",
            "SearchCreativeWork",
            "SearchScreeningEvent"
        ]

    def load(self):
        snips = load_dataset("benayas/snips")
        self.data['train'] = [
            {
                "id": str(i),
                "question": s["text"],
                "gold_answer": s["category"]
            }
            for i, s in enumerate(snips["train"])
        ]
        self.data['test'] = [
            {
                "id": str(i),
                "question": s["text"],
                "gold_answer": s["category"]
            }
            for i, s in enumerate(snips["test"])
        ]

def build_dataset(dataset_name: str):
    if dataset_name == "ag_news":
        dataset = AGNewsDataset()
        output_prefix = "processed-data/ag_news"
        min_shot = 4
    elif dataset_name == "snips":
        dataset = SnipsDataset()
        output_prefix = "processed-data/snips"
        min_shot = None  # Will be set after loading
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset.load()
    if dataset_name == "snips":
        min_shot = len(dataset.labels)
    Path(output_prefix).mkdir(parents=True, exist_ok=True)
    Path(f"{output_prefix}/run_samples").mkdir(parents=True, exist_ok=True)

    # Convert to IO samples
    train_samples = convert_to_io_samples(dataset.name, "train", dataset.data['train'], dataset.instruction)
    test_samples = convert_to_io_samples(dataset.name, "test", dataset.data['test'], dataset.instruction)

    # Save train.jsonl
    with open(f"{output_prefix}/train.jsonl", "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Save test.jsonl
    with open(f"{output_prefix}/test.jsonl", "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Create 5 different 200-sample training sets and save shot indices
    shot_values = [min_shot, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    for i in range(5):
        train_200 = random.sample(train_samples, 200)
        train_200_indices = [s["id"] for s in train_200]
        # For the minimum shot, pick the first sample for each label from the 200-sample pool
        min_shot_indices = get_one_per_label(train_200, dataset.labels)
        for shot in shot_values:
            if shot == min_shot:
                indices = min_shot_indices
            else:
                indices = train_200_indices[:shot]
            with open(f"{output_prefix}/run_samples/train{i}_{shot}.json", "w") as f:
                json.dump(indices, f)

if __name__ == "__main__":
    build_dataset("ag_news")
    build_dataset("snips") 