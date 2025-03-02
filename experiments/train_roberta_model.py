import json
import logging
import os
import argparse
import sys

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import torch
import numpy as np

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path
parser.add_argument(
    "--train_data_path",
    type=str,
    default="/home/xinsu/Projects/General/source-data-model-free-da/raw-data/negation-qa/i2b2/dev-clustered-sampled-100-examples.json",
)
parser.add_argument(
    "--test_data_pth",
    type=str,
    default="/home/xinsu/Projects/General/source-data-model-free-da/raw-data/negation-qa/mimic/test.json",
)

# Checkpoints
parser.add_argument(
    "--output_path",
    type=str,
    default="/home/xinsu/Projects/General/source-data-model-free-da/trained_models/negation-qa-i2b2-roberta-random-100",
)

# Model and tokenizer names
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
parser.add_argument("--config_name", type=str, default="roberta-base")

# MISC
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--do_train", action="store_true")

# Tunable
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
parser.add_argument("--per_gpu_eval_batch_size", type=int, default=32)
parser.add_argument("--grad_accum_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--warmup_steps", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--learning_rate", type=float, default=5e-5)

os.environ["WANDB_DISABLED"] = "true"


# Dataset Class
class NegationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, pretrained_tokenizer, max_length, labels=None):
        self.examples = examples
        self.labels = labels
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        tokenized_features = self.pretrained_tokenizer(
            self.examples[item], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = tokenized_features.input_ids.squeeze()
        attention_mask = tokenized_features.attention_mask.squeeze()

        outputs = dict(input_ids=input_ids, attention_mask=attention_mask)

        if self.labels is not None:
            outputs["labels"] = torch.tensor(self.labels[item])

        return outputs

    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Setup the logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Will log the info only in the main process
    # Warning will be logged in all processes
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Make the output directory
    # args.output_path = os.path.join(args.output_path, datetime.now().strftime('%Y-%m-%d-%H-%M'))
    os.makedirs(args.output_path, exist_ok=True)

    # Count the available number of GPUs and save the number to args.
    print("-" * 80)
    args.num_gpus = torch.cuda.device_count()
    print("Number of GPUs = {}".format(args.num_gpus))
    set_seed(43)

    # label map
    label_str2id = {"no claim": 0, "correlational": 1, "conditional causal": 2, "direct causal": 3}
    label_id2str = {v: k for k, v in label_str2id.items()}

    # Load the model
    print("-" * 80)
    print(f"Load model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.config_name, num_labels=len(label_str2id))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Load the raw text data
    print("-" * 80)
    print("Load the data")
    with open(args.train_data_path) as f:
        train_data = json.load(f)
    with open(args.test_data_pth) as f:
        test_data = json.load(f)

    train_examples = []
    train_label_ids = []

    for example in train_data:
        train_examples.append(example["question"])
        label_id = label_str2id[example["gold_answer"].lower()]
        train_label_ids.append(label_id)

    test_examples = []
    test_label_ids = []
    for example in test_data:
        test_examples.append(example["question"])
        label_id = label_str2id[example["gold_answer"].lower()]
        test_label_ids.append(label_id)

    print(f"Number of dev data = {len(train_examples)}")
    print(f"Number of test data = {len(test_examples)}")

    # Prepare the dataset
    test_dataset = NegationDataset(
        examples=test_examples, labels=test_label_ids, max_length=args.max_length, pretrained_tokenizer=tokenizer
    )
    dev_dataset = NegationDataset(
        examples=train_examples, labels=train_label_ids, max_length=args.max_length, pretrained_tokenizer=tokenizer
    )

    # Prepare the training argument and trainer
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        save_total_limit=1,
        logging_steps=100,
        seed=args.seed,
        disable_tqdm=False,
        save_steps=10000,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dev_dataset)

    if args.do_train:
        # Train oracle model on the dev set
        print("-" * 80)
        print("Start training the model on dev set")
        print("=" * 80)
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        print("=" * 80)

    # Eval on test set
    print("\nPredict on the test set")
    outputs = trainer.predict(test_dataset=test_dataset)
    labels_pred = np.argmax(outputs.predictions, 1).tolist()
    labels_pred = [label_id2str[label_id] for label_id in labels_pred]
    to_save = []
    for example, pred_label in zip(test_data, labels_pred):
        example["predicted_answer"] = pred_label

    with open(os.path.join(args.output_path, "test_preds.json"), "w") as f:
        json.dump(test_data, f, indent=4)