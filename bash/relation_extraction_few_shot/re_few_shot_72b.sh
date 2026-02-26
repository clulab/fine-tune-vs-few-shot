#!/bin/bash

# MAJOR PATHS

HOME_DIR="/xin/projects/fine-tune-vs-few-shot"
DATA_DIR=$HOME_DIR"/raw-data"
OUTPUT_DIR=$HOME_DIR"/output/fewshot/re"


. /root/miniconda3/etc/profile.d/conda.sh
conda activate ft-vs-icl

# clear previously created directories
rm -rf $OUTPUT_DIR

echo "Starting experiments"

# Define experiment parameters

GEN_CONFIG_PATH=$HOME_DIR"/gen_config.json"
MODELS=(
    "Qwen/Qwen2.5-72B-Instruct"
)

SOURCE_DATASETS=("n2c2") # "made"
TARGET_DATASETS=("n2c2" "made") # "made"

FEW_SHOT_NS=(7 20 40 60 80 100 120 140 160 180 200)
EVAL_SET="test"
RUN_N=5
INSTRUCTION_TYPE="reasoning"
GEN_CONFIG_TYPES=("default2")
# Create array of all combinations
declare -A configs
index=0
for model in "${MODELS[@]}"; do
    for source_dataset in "${SOURCE_DATASETS[@]}"; do
        for target_dataset in "${TARGET_DATASETS[@]}"; do
            for few_shot in "${FEW_SHOT_NS[@]}"; do
                for gen_config_type in "${GEN_CONFIG_TYPES[@]}"; do
                    configs[$index]="$model $source_dataset $target_dataset $few_shot $gen_config_type"
                    ((index++))
                done
            done
        done
    done
done

# Run experiments
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        # unpack model, source/target datasets, shot count, and gen_config_type
        read model source_dataset target_dataset few_shot gen_config_type <<< "${configs[$i]}"
        model_path="${model}"
        model_name="${model##*/}"
        test_data_path="$DATA_DIR/${target_dataset}/${target_dataset}_${EVAL_SET}.jsonl"
        sample_path="$DATA_DIR/${source_dataset}/run_samples/${source_dataset}_train_${run_n}_${few_shot}.json"
        prompt_data_path="$DATA_DIR/${source_dataset}/${source_dataset}_train.jsonl"
        preds_path=$OUTPUT_DIR"/${source_dataset}_${target_dataset}/${model_name}/${model_name}_${run_n}_${few_shot}_shot_${gen_config_type}_config"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"

        echo "============================================="
        echo "Model: $model_name | Source: $source_dataset | Target: $target_dataset | Shot: $few_shot | Run: $run_n"
        echo "============================================="

        ### Run experiment
        python $HOME_DIR/experiments/run_vllm_prompting.py \
            --ckpt_dir $model_path \
            --tokenizer $model_path \
            --test_data $test_data_path \
            --prompt_data $prompt_data_path \
            --gen_config_type $gen_config_type \
            --output_dir $preds_path/source_preds.json \
            --sample $sample_path \
            --gen_config_path $GEN_CONFIG_PATH \
            --tensor_parallel_size 4

        ### Run evaluation
        python $HOME_DIR/evaluation/evaluate_relation_extraction.py \
            --generated_file $preds_path/source_preds.json \
            --output_dir $OUTPUT_DIR \
            --model_name $model_name \
            --eval_set $EVAL_SET \
            --source_dataset $source_dataset \
            --target_dataset $target_dataset \
            --run_n $run_n \
            --sample_n $few_shot \
            --gen_config $gen_config_type
    done
done
# copy this file to OUTPUT_DIR
cp $HOME_DIR/bash/relation_extraction_few_shot/re_few_shot_72b.sh $OUTPUT_DIR/re_few_shot_72b.sh
