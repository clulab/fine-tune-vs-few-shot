#!/usr/bin/env bash


# Ensure NUM is set; exit if not provided
if [ -z "${NUM}" ]; then
    echo "Error: NUM variable is not set."
    exit 1
fi

printf "\n\n*************************************************\n"
printf "Current num: %d\n" $NUM
printf "*************************************************\n"

export PYTHONPATH=/home/xinsu/source-data-model-free-da/:${PYTHONPATH}
export TRAIN_DATA=/home/xinsu/source-data-model-free-da/raw-data/causal_language_press_few_shot_examples/press_releases_causal_language_train_sent_to_class_${NUM}.json
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
export TRAINED_MODEL=/export/share/projects/mcai/xinsu/sfda/trained_models/causal_from_press_to_pubmed_llama_3.2_3b_${NUM}

python /home/xinsu/source-data-model-free-da/experiments/train_llm.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $TRAIN_DATA \
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

# run on the source data
python /home/xinsu/source-data-model-free-da/experiments/run_vllm.py \
    --ckpt_dir $TRAINED_MODEL \
    --tokenizer $MODEL_NAME \
    --test_data /home/xinsu/source-data-model-free-da/raw-data/causal_language_use/press_releases_causal_language_use_test_sent_to_class.json \
    --output_dir $TRAINED_MODEL/source_preds.json \

# run on the target data
python /home/xinsu/source-data-model-free-da/experiments/run_vllm.py \
    --ckpt_dir $TRAINED_MODEL \
    --tokenizer $MODEL_NAME \
    --test_data /home/xinsu/source-data-model-free-da/raw-data/causal_language_use/pubmed_causal_language_use_test_sent_to_class.json \
    --output_dir $TRAINED_MODEL/target_preds.json \

# start do evaluation
printf "====================\n"
printf "Start evaluation\n"
printf "Source data\n"
# show the num
printf "Current num: %d\n" $NUM
printf "====================\n"

python /home/xinsu/source-data-model-free-da/evaluation/evaluate_causal_language.py \
    --generated_file $TRAINED_MODEL/source_preds.json

printf "====================\n"
printf "Target data\n"
# show the num
printf "Current num: %d\n" $NUM
printf "====================\n"
python /home/xinsu/source-data-model-free-da/evaluation/evaluate_causal_language.py \
    --generated_file $TRAINED_MODEL/target_preds.json
