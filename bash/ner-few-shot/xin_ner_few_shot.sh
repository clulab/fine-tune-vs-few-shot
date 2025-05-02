set -e
source /home/xinsu/miniforge3/etc/profile.d/conda.sh
conda activate source-data-free-da

echo "Starting experiments"

# Define experiment parameters
MODEL_PATH="/export/share/projects/mcai/xinsu/huggingface_models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
# Define paths
BASE_DIR="/home/xinsu/fine-tune-vs-few-shot"
DATA_DIR="/home/xinsu/fine-tune-vs-few-shot/data/"
OUTPUT_DIR="/home/xinsu/fine-tune-vs-few-shot/experiment_outputs/ner-few-shot"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

GEN_CONFIG_PATH="$BASE_DIR/gen_config.json"
MODEL_NAME="DeepSeek-R1-Distill-Llama-70B"
SOURCE_DATASETS=("cdr") # "ehealth"``
TARGET_DATASETS=("cdr" "ehealth") # "ehealth"

FEW_SHOT_NS=(2 20 40 60 80 100 120 140 160 180 200)
EVAL_SET="test"
RUN_N=5

# Create array of all combinations
declare -A configs
index=0
for source_dataset in "${SOURCE_DATASETS[@]}"; do
    for target_dataset in "${TARGET_DATASETS[@]}"; do
        for few_shot in "${FEW_SHOT_NS[@]}"; do
            configs[$index]="$source_dataset $target_dataset $few_shot"
            ((index++))
        done
    done
done

# Generate and submit PBS jobs
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        identifier="fewshot_$(date +%Y%m%d_%H%M%S)_config_${i}_run_${run_n}"
        read source_dataset target_dataset few_shot <<< "${configs[$i]}"
        model_path="${MODEL_PATH}"
        model_name="${MODEL_NAME}"
        test_data_path="${DATA_DIR}/${target_dataset}/${EVAL_SET}.jsonl"
        sample_path="${SAMPLE_DIR}/${source_dataset}/run_samples/train${run_n}_${few_shot}.json"
        prompt_data_path="${DATA_DIR}/${source_dataset}/train.jsonl"
        preds_path="${OUTPUT_DIR}/ner/${source_dataset}_${target_dataset}/${model_name}/${model_name}_${run_n}_${few_shot}_shot"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"
        ### Run experiment
        CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python experiments/run_vllm_prompting.py \
            --ckpt_dir $model_path \
            --tokenizer $model_path \
            --test_data $test_data_path \
            --prompt_data $prompt_data_path \
            --output_dir $preds_path/source_preds.json \
            --sample $sample_path \
            --gen_config_path $GEN_CONFIG_PATH \
            --tensor_parallel_size 4
        ### Run evaluation
        CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python eval?uation/evaluate_ner.py \
            --generated_file $preds_path/source_preds.json \
            --model_name $model_name \
            --eval_set $EVAL_SET \
            --source_dataset $source_dataset \
            --target_dataset $target_dataset \
            --run_n $run_n \
            --sample_n $few_shot
    done
done