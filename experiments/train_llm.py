import logging
import sys
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, List, Any

from torch.utils.data import Dataset
import transformers
from transformers import Trainer, set_seed
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: str = field(default="right", metadata={"help": "The padding side in tokenizer"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sample_path: str = field(default=None, metadata={"help": "Path to the sampled indexes."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


class SupervisedDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs, tokenizer):  # List of input strings  # List of dictionaries
        self.src_seqs = []
        self.tgt_seqs = []
        self.tokenizer = tokenizer

        for src, tgt in zip(src_seqs, tgt_seqs):
            src = [{"role": "user", "content": src}]
            src = self.tokenizer.apply_chat_template(src, add_generation_prompt=True, tokenize=False)

            tgt = f"{tgt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            self.src_seqs.append(src)
            self.tgt_seqs.append(tgt)

    def __getitem__(self, item):
        outputs = {
            "src_seq": self.src_seqs[item],
        }

        if self.tgt_seqs is not None:
            outputs["tgt_seq"] = self.tgt_seqs[item]

        return outputs

    def __len__(self):
        return len(self.src_seqs)


class SupervisedDataCollator(object):
    def __init__(self, max_len, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, features):
        prompts = []
        for feature in features:
            prompts.append(feature["src_seq"])

        answers = []
        if "tgt_seq" in features[0]:
            for feature in features:
                answers.append(feature["tgt_seq"])

        examples = []
        for prompt, answer in zip(prompts, answers):
            examples.append(prompt + answer)

        tokenized_examples = self.tokenizer(
            examples, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        input_ids = tokenized_examples["input_ids"]
        attention_mask = tokenized_examples["attention_mask"]
        labels = tokenized_examples["input_ids"].clone()

        # first mask the padded tokens in the labels
        attention_mask_bool = attention_mask.bool()
        labels = labels.masked_fill(~attention_mask_bool, -100)

        # mask the prompts in the input_ids
        for prompt, label in zip(prompts, labels):
            label[: len(self.tokenizer.encode(prompt))] = -100

        collated_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        return collated_batch


def train():
    global local_rank

    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Setup the logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Will log the info only in the main process
    # Warning will be logged in all processes
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARN)
    transformers.utils.logging.set_verbosity(logging.INFO if local_rank in [-1, 0] else logging.WARN)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    with open(data_args.data_path, "r") as f:
        if data_args.data_path.endswith(".jsonl"):
            data = []
            for l in f:
                data.append(json.loads(l))
        else:
            data = json.load(f)
    if data_args.sample_path:
        with open(data_args.sample_path, "r") as f:
            sample = set(json.load(f))
        data = [i for i in data if i["id"] in sample]
    src_seqs = [i["question"] for i in data]
    tgt_seqs = [i["gold_answer"] for i in data]

    # Start trainner
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=SupervisedDataset(src_seqs, tgt_seqs, tokenizer),
        data_collator=SupervisedDataCollator(
            tokenizer=tokenizer,
            max_len=training_args.model_max_length,
        ),
    )

    trainer.train()
    logger.info("-" * 80)
    logger.info(f"Training complete! Save the model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
