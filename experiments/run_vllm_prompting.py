import argparse
import json
import logging

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def main():

    # Prepare the data
    with open(args.test_data) as f:
        test_data = json.load(f)
    print(f"Number of examples loaded: {len(test_data)}")

    with open(args.prompt_data) as f:
        prompt_data = json.load(f)
    print(f"Number of prompt examples loaded: {len(prompt_data)}")

    # prepare the few-shot examples
    few_shot_examples = []
    for example in prompt_data:
        few_shot_example = f"In: {example['question']} Out: {example['gold_answer']}"
        few_shot_examples.append(few_shot_example)
    few_shot_examples = "\n".join(few_shot_examples)

    print("Few-shot examples")
    print("-" * 50)
    print(few_shot_examples)
    print("-" * 50)

    model_input_texts = []
    for example in test_data:
        current_in = f"In: {example['question']} Out:"
        current_in = f"{few_shot_examples}\n{current_in}"
        # message = [{"role": "user", "content": current_in}]
        # prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        model_input_texts.append(current_in)

    # Load the model
    model = LLM(
        model=args.ckpt_dir, tensor_parallel_size=args.tensor_parallel_size, seed=42
    )
    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=32, stop=["\n"])

    # Start the generation
    print("Model input example: ")
    print("-" * 50)
    print(model_input_texts[0])
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
    print(sorted_generated[0])
    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--prompt_data", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tensor_parallel_size", type=int)
    args = parser.parse_args()
    main()
