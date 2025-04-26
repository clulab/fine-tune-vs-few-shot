#!/usr/bin/env bash

export SOURCE=n2c2
export TARGET=made
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

export CODE_HOME=/path/to/code/
export DATA_HOME=/path/to/data/
export TRAIN_DATA=${DATA_HOME}/relation_extraction/${SOURCE}/${SOURCE}_train.jsonl
export SRC_TEST_DATA=${DATA_HOME}/relation_extraction/${SOURCE}/${SOURCE}_toy.jsonl
export TGT_TEST_DATA=${DATA_HOME}/relation_extraction/${TARGET}/${TARGET}_toy.jsonl
export SAMPLE_DIR=${DATA_HOME}/relation_extraction/${SOURCE}/run_samples/
export OUTPUT_DIR=${DATA_HOME}/relation_extraction/outputs/fine_tune/
export TRAINED_MODELS_DIR=${DATA_HOME}/relation_extraction/trained_models/

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
printf "Trained models path: $TRAINED_MODELS_DIR\n"

for RUN in {0..4}
do
    # Loop through the values you want to test
    for SHOT in 6 20 40 60 80 100 120 140 160 180 200
    do
        export SAMPLE=$SAMPLE_DIR/${SOURCE}_train_${RUN}_${SHOT}.json
        export PREDS_PATH=$OUTPUT_DIR/${MODEL_BASE_NAME}_${SOURCE}_${RUN}_${SHOT}
        export TRAINED_MODEL=$TRAINED_MODELS_DIR/${MODEL_BASE_NAME}_${SOURCE}_${RUN}_${SHOT}

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

        printf "\n\n*************************************************\n"
        printf "Current run: %d\n" $RUN
        printf "Current shot: %d\n" $SHOT
        printf "*************************************************\n"

        printf "====================\n"
        printf "Running fine-tuning"
        python $CODE_HOME/experiments/train_llm.py \
            --model_name_or_path $MODEL_NAME \
            --data_path $TRAIN_DATA \
            --sample_path $SAMPLE \
            --bf16 True \
            --output_dir $TRAINED_MODEL \
            --gradient_accumulation_steps 1 \
            --learning_rate 2e-5 \
            --log_level info \
            --lr_scheduler_type "cosine" \
            --num_train_epochs 5 \
            --optim adamw_torch \
            --per_device_train_batch_size 4 \
            --save_strategy "no" \
            --seed 42 \
            --warmup_ratio 0.1 \
            --logging_steps 10 \
            --report_to none \
            --gradient_checkpointing True \
            --bf16 True \
            --remove_unused_columns False

        printf "====================\n"
        printf "Source data\n"
        printf "Running prediction\n"
        python $CODE_HOME/experiments/run_vllm.py \
	    --base_model $MODEL_NAME \
            --ckpt_dir $TRAINED_MODEL \
            --tokenizer $MODEL_NAME \
            --test_data $SRC_TEST_DATA \
            --output_dir $PREDS_PATH/source_preds.json \

	printf "Running evaluation\n"
        python $CODE_HOME/evaluation/evaluate_relation_extraction.py \
           --generated_file $PREDS_PATH/source_preds.json

        printf "====================\n"
        printf "Target data\n"
        printf "Running prediction\n"
        python $CODE_HOME/experiments/run_vllm.py \
	   --base_model $MODEL_NAME \
           --ckpt_dir $TRAINED_MODEL \
           --tokenizer $MODEL_NAME \
           --test_data $TGT_TEST_DATA \
           --output_dir $PREDS_PATH/target_preds.json \

        printf "Running evaluation\n"
        python $CODE_HOME/evaluation/evaluate_relation_extraction.py \
           --generated_file $PREDS_PATH/target_preds.json
    done
done
