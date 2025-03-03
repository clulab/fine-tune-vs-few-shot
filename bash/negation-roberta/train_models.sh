#!/usr/bin/env bash

# Ensure NUM is set; exit if not provided
if [ -z "${NUM}" ]; then
    echo "Error: NUM variable is not set."
    exit 1
fi

# Setup the python path
export PYTHONPATH=/home/xinsu/source-data-model-free-da/:${PYTHONPATH}
export TRAIN_DATA=/home/xinsu/source-data-model-free-da/raw-data/negation_few_shot_examples/negation_few_shot_examples_${NUM}.json
export MODEL_NAME=roberta-base
export TRAINED_MODEL=/export/share/projects/mcai/xinsu/sfda/trained_models/negation_from_i2b2_to_mimic_roberta-base_${NUM}
export SOURCE_DATA=/home/xinsu/source-data-model-free-da/raw-data/negation-qa/i2b2/test_sent_to_class_name.json
export TARGET_DATA=/home/xinsu/source-data-model-free-da/raw-data/negation-qa/mimic/test_sent_to_class_name.json

# Train the model
python /home/xinsu/source-data-model-free-da/experiments/train_classification_model_negation.py \
    --model_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --train_data_path $TRAIN_DATA \
    --output_path $TRAINED_MODEL \
    --do_train \
    --epochs 10 \
    --learning_rate 2e-5

# Predict on the source data
python /home/xinsu/source-data-model-free-da/experiments/train_classification_model_negation.py \
    --model_name $TRAINED_MODEL \
    --tokenizer_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --output_path $TRAINED_MODEL \
    --test_data_pth $SOURCE_DATA \
    --do_eval \
    --pred_file_name source_preds

# Predict on the target data
python /home/xinsu/source-data-model-free-da/experiments/train_classification_model_negation.py \
    --model_name $TRAINED_MODEL \
    --tokenizer_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --output_path $TRAINED_MODEL \
    --test_data_pth $TARGET_DATA \
    --do_eval \
    --pred_file_name target_preds

# Evaluate the model
printf "Model name: $MODEL_NAME\n"
printf "Train data path: $TRAIN_DATA\n"

printf "Start evaluation\n"
printf "Source data\n"
python /home/xinsu/source-data-model-free-da/evaluation/evaluate_negation.py \
    --generated_file $TRAINED_MODEL/source_preds.json
printf "Target data\n"
python /home/xinsu/source-data-model-free-da/evaluation/evaluate_negation.py \
    --generated_file $TRAINED_MODEL/target_preds.json
