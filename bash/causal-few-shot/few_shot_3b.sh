#!/usr/bin/env bash

export PYTHONPATH=/homes/xinsu/source-data-model-free-da/:${PYTHONPATH}
export MODEL_NAME="meta-llama/Llama-3.2-3B"

# Loop through the values you want to test
for SHOT in 20 40 60 80 100 120 140 160 180 200
  do
    export PROMPT_DATA_PATH=/home/xinsu/source-data-model-free-da/raw-data/causal_language_press_few_shot_examples/press_releases_causal_language_train_sent_to_class_${SHOT}.json
    export PREDS_PATH=/export/share/projects/mcai/xinsu/sfda/few_shot_experiments_outputs/causal_from_press_to_pubmed/Llama-3.2-3B-Instruct-shot_${SHOT}

    # Ensure output directory exists
    if [ ! -d "$PREDS_PATH" ]; then
        echo "Output directory $PREDS_PATH does not exist. Attempting to create it..."
        if mkdir -p "$PREDS_PATH"; then
            echo "Successfully created output directory: $PREDS_PATH"
        else
            echo "Failed to create output directory: $PREDS_PATH"
            exit 1
        fi
    else
        echo "Output directory $PREDS_PATH already exists."
    fi

    # print the model name, prompt data path
    printf "Model name: $MODEL_NAME\n"
    printf "Prompt data path: $PROMPT_DATA_PATH\n"

    python /home/xinsu/source-data-model-free-da/experiments/run_vllm_prompting.py \
        --ckpt_dir $MODEL_NAME \
        --tokenizer $MODEL_NAME \
        --test_data /home/xinsu/source-data-model-free-da/raw-data/causal_language_use/press_releases_causal_language_use_test_sent_to_class.json \
        --prompt_data $PROMPT_DATA_PATH \
        --output_dir $PREDS_PATH/source_preds.json \
        --tensor_parallel_size 2

    python /home/xinsu/source-data-model-free-da/experiments/run_vllm_prompting.py \
        --ckpt_dir $MODEL_NAME \
        --tokenizer $MODEL_NAME \
        --test_data /home/xinsu/source-data-model-free-da/raw-data/causal_language_use/pubmed_causal_language_use_test_sent_to_class.json \
        --prompt_data $PROMPT_DATA_PATH \
        --output_dir $PREDS_PATH/target_preds.json \
        --tensor_parallel_size 2

    # start do evaluation
    printf "Start evaluation\n"
    printf "Source data\n"
    printf "====================\n"

    python /home/xinsu/source-data-model-free-da/evaluation/evaluate_causal_language.py \
        --generated_file $PREDS_PATH/source_preds.json

    printf "Target data\n"
    printf "====================\n"
    python /home/xinsu/source-data-model-free-da/evaluation/evaluate_causal_language.py \
        --generated_file $PREDS_PATH/target_preds.json

  done