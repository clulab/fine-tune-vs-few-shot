import argparse
import json
import logging
import os
import random
from vllm import LLM, SamplingParams

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


    with open(args.prompt_data) as f:
        if args.prompt_data.endswith(".jsonl"):
            prompt_data = []
            for l in f:
                prompt_data.append(json.loads(l))
        else:
            prompt_data = json.load(f)
    if args.sample:
        with open(args.sample) as f:
            sample = set(json.load(f))
        prompt_data = [i for i in prompt_data if i["id"] in sample]
    print(f"Number of prompt examples loaded: {len(prompt_data)}")

    print("model: ", args.ckpt_dir)     

    # prepare the few-shot examples
    few_shot_examples = []
    for example in prompt_data:
        few_shot_example = f"{example['question']}{example['gold_answer']}\n"
        few_shot_examples.append(few_shot_example)
    few_shot_examples = "\n".join(few_shot_examples)

    print("Few-shot examples")
    print("-" * 50)
    print(few_shot_examples)
    print("-" * 50)

    model_input_texts = []
    for example in test_data:
        current_in = f"{example['instruction']}\n{few_shot_examples}\n{example['question']}"
        # message = [{"role": "user", "content": current_in}]
        # prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        model_input_texts.append(current_in)
        
    # save model_input_texts 
    if args.input_texts_save_path:
        # check if file exists, if so skip
        if os.path.exists(args.input_texts_save_path):
            print(f"Input texts file already exists: {args.input_texts_save_path}")
        else:
            with open(args.input_texts_save_path, "w", encoding="utf-8") as f:
                for text in model_input_texts:
                    f.write(text + "\n")
            print(f"Input texts saved to {args.input_texts_save_path}") 
        

    # Load the model
    model = LLM(
        model=args.ckpt_dir, tensor_parallel_size=args.tensor_parallel_size, seed=42
    )
    if args.gen_config_type and args.gen_config_path:
        with open(args.gen_config_path, "r") as f:
            gen_config = json.load(f)
            print(f"Using {args.gen_config_type} gen_config")
            print(gen_config[args.gen_config_type])
            sampling_params = SamplingParams(**gen_config[args.gen_config_type])

    else:
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=100, stop=["\n"])

    # Start the generation
    print("Model input example: ")
    print("-" * 50)
    print(model_input_texts[-1])
    print("-" * 50)

    print("Generating...")
    outputs = model.generate(model_input_texts, sampling_params=sampling_params)

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
    import random
    print(random.choice(sorted_generated))
    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--prompt_data", type=str)
    parser.add_argument("--sample", type=str)
    parser.add_argument("--gen_config_type", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--main_prompt_path", type=str, default=None)
    parser.add_argument("--gen_config_path", type=str, default=None)
    parser.add_argument("--input_texts_save_path", type=str, default=None)
    
    args = parser.parse_args()
    main()
