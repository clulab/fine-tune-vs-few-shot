import argparse
import json
import logging
import sys

import transformers
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


def main():

    # Prepare the data
    with open(args.test_data) as f:
        if args.test_data.endswith(".jsonl"):
            test_data = []
            for l in f:
                test_data.append(json.loads(l))
        else:
            test_data = json.load(f)
        
    print(f"Number of examples loaded: {len(test_data)}")

    prompts = []

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    for example in test_data:
        question = example["question"]
        message = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # Load the model
    model = LLM(
        model=args.ckpt_dir,
        tensor_parallel_size=args.tensor_parallel_size, 
        seed=42,
        enable_lora=args.enable_lora,
    )
    
    if args.gen_config_path:
        with open(args.gen_config_path, "r") as f:
            gen_config = json.load(f)
        sampling_params = SamplingParams(**gen_config)
    else:
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=32)

    # Setup LoRA request if needed
    lora_request = None
    if args.enable_lora and args.lora_path:
        lora_request = LoRARequest(
            "lora_adapter",  # default name
            1,               # default ID
            args.lora_path
        )

    # Start the generation
    print("Prompt example")
    print(prompts[0])
    print("Generating...")
    outputs = model.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)

    # Prepare the results
    generated = []
    request_ids = []
    for output in outputs:
        generated_text = output.outputs[0].text
        request_ids.append(int(output.request_id))
        generated.append(generated_text)

    # the generated texts are not in the same order as the input prompts
    paired = list(zip(request_ids, generated))
    paired.sort()
    sorted_generated = [item[1] for item in paired]

    # Prepare the results
    for generated, example in zip(sorted_generated, test_data):
        example["predicted_answer"] = generated

    print("Generated example")
    print(sorted_generated[0])
    with open(args.output_dir, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--test_data",type=str)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--gen_config_path",type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    # LoRA related parameters
    parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA adapter")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()
    main()
