#!/bin/bash

# Clear created directories
rm -rf fewshot_pbs_files
rm -rf output

# Clear old jobs (if using PBS/qstat)
qstat | awk 'NR>1 && ($1 ~ /^[rq]$/) && $5 == "bulut" {print $4}' | xargs -I {} qdel {}

# Clear old job log files
rm -rf job.*.stdout.txt
rm -rf job.*.stderr.txt

# Create PBS file dir
mkdir -p fewshot_pbs_files

echo "Starting classification few-shot experiments"

# Define experiment parameters
LOCAL_MODEL_PATH="/media/networkdisk/bulut2/local-models"
GEN_CONFIG_PATH="/home/bulut/fine-tune-vs-few-shot/gen_config.json"
MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen2.5-3-Instruct")
# "ag_news" "snips"
SOURCE_DATASETS=("ag_news")
TARGET_DATASETS=("ag_news")
FEW_SHOT_NS=(2 20 40 60 80 100 120 140 160 180 200)
EVAL_SET="test"
RUN_N=5

# Create array of all combinations
declare -A configs
index=0
for model in "${MODELS[@]}"; do
    for source_dataset in "${SOURCE_DATASETS[@]}"; do
        for target_dataset in "${TARGET_DATASETS[@]}"; do
            for few_shot in "${FEW_SHOT_NS[@]}"; do
                configs[$index]="$model $source_dataset $target_dataset $few_shot"
                ((index++))
            done
        done
    done
done

# Generate and submit PBS jobs
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        identifier="fewshot_$(date +%Y%m%d_%H%M%S)_config_${i}_run_${run_n}"
        read model source_dataset target_dataset few_shot <<< "${configs[$i]}"
        model_path="${model}"
        test_data_path="processed-data/${target_dataset}/${EVAL_SET}.jsonl"
        sample_path="processed-data/${source_dataset}/run_samples/train${run_n}_${few_shot}.json"
        prompt_data_path="processed-data/${source_dataset}/train.jsonl"
        preds_path="output/classification/${source_dataset}_${target_dataset}/${model}_${run_n}_${few_shot}_shot"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"
        # Create PBS script
        cat <<EOF > "fewshot_pbs_files/${identifier}.pbs"
#!/bin/bash
### Job Name
#PBS -N fewshot_${model}_${source_dataset}_${target_dataset}_${few_shot}_${EVAL_SET}_run${run_n}
### Project code
#PBS -A classification_fewshot
### Maximum time this job can run before being killed (here, 1 day)
#PBS -l walltime=01:00:00:00
### Resource Request (must contain cpucore, memory, and gpu (even if requested amount is zero)
#PBS -l cpucore=2:memory=50gb:gpu=4
### Output Options (default is stdout_and_stderr)
#PBS -l outputMode=stdout_and_stderr

. /home/bulut/miniconda3/etc/profile.d/conda.sh

conda activate ft-vs-icl

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
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python evaluation/evaluate_classification.py \
    --generated_file $preds_path/source_preds.json \
    --model_name $model \
    --eval_set $EVAL_SET \
    --source_dataset $source_dataset \
    --target_dataset $target_dataset \
    --run_n $run_n \
    --sample_n $few_shot
EOF

        # Submit job
        qsub "fewshot_pbs_files/${identifier}.pbs"
    done
done 