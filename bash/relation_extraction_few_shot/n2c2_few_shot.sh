#!/usr/bin/env bash

export SOURCE=n2c2
export TARGET=made
export MODEL_NAME="meta-llama/Llama-3.2-3B"

export CODE_HOME=/path/to/fine-tune-vs-few-shot/
export HF_HOME=/path/to/cache/huggingface/
export TRAIN_DATA=/path/to/${SOURCE}_train.jsonl
export SRC_TEST_DATA=/path/to/${SOURCE}_toy.jsonl
export TGT_TEST_DATA=/path/to/${TARGET}_toy.jsonl
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
export SAMPLE_DIR=/path/to/samples_dir/
export OUTPUT_DIR=/path/to/output_dir/
export NUM_GPUS=2

IFS='/' read -r -a array <<< "$MODEL_NAME"
export MODEL_BASE_NAME=${array[1]}

printf "Model name: $MODEL_NAME\n"
printf "Source: $SOURCE\n"
printf "Target: $TARGET\n"
printf "Train data path: $TRAIN_DATA\n"
printf "Sample path: $SAMPLE\n"
printf "Source test data path: $SRC_TEST_DATA\n"
printf "Target test path: $TGT_TEST_DATA\n"
printf "Output path: $OUTPUT_DIR\n"
printf "Number of GPUS: $NUM_GPUS\n"

for RUN in {0..4}
do
    # Loop through the values you want to test
    for SHOT in 6 20 40 60 80 100 120 140 160 180 200
    do
        export SAMPLE=$SAMPLE_DIR/${SOURCE}_train_${RUN}_${NUM}.json
        export PREDS_PATH=$OUTPUT_DIR/${MODEL_BASE_NAME}_${SOURCE}_${RUN}_${NUM}

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
        printf "\n\n*************************************************\n"
        printf "Current run: %d\n" $NUM
        printf "Current shot: %d\n" $SHOT
        printf "*************************************************\n"

        printf "====================\n"
        printf "Source data\n"
        printf "Running few-shot prediction\n"
        python $CODE_HOME/experiments/run_vllm_prompting.py \
            --ckpt_dir $MODEL_NAME \
            --tokenizer $MODEL_NAME \
            --test_data $IN_TEST_DATA \
            --prompt_data $TRAIN_DATA \
            --sample $SAMPLE \
            --output_dir $PREDS_PATH/source_preds.json \
            --tensor_parallel_size $NUM_GPUS

        printf "Running evaluation\n"
        python $CODE_HOME/evaluation/evaluate_relation_extraction.py \
            --generated_file $PREDS_PATH/source_preds.json
 
        printf "====================\n"
        printf "Target data\n"
        printf "Running few-shot prediction\n"
        python $CODE_HOME/experiments/run_vllm_prompting.py \
            --ckpt_dir $MODEL_NAME \
            --tokenizer $MODEL_NAME \
            --test_data $IN_TEST_DATA \
            --prompt_data $TRAIN_DATA \
            --sample $SAMPLE \
            --output_dir $PREDS_PATH/target_preds.json \
            --tensor_parallel_size $NUM_GPUS

        printf "Running evaluation\n"
        python $CODE_HOME/evaluation/evaluate_relation_extraction.py \
            --generated_file $PREDS_PATH/target_preds.json
    done
done
