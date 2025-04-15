#!/bin/bash

# clear created directories
rm -rf finetune_pbs_files
rm -rf output
# clear old jobs
qstat | awk 'NR>1 && ($1 ~ /^[rq]$/) && $5 == "bulut" {print $4}' | xargs -I {} qdel {}

# clear old job log files that are stdout.txt and stderr.txt
rm -rf job.*.stdout.txt
rm -rf job.*.stderr.txt

# create pbs file dir
mkdir -p finetune_pbs_files

echo "Starting experiments"

# Define experiment parameters
LOCAL_MODEL_PATH="/media/networkdisk/bulut2/local-models"
GEN_CONFIG_PATH="/home/bulut/fine-tune-vs-few-shot/gen_config.json"
# MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "Llama/Llama-3.2-3B-Instruct" "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-32B")
MODELS=("meta-llama/Llama-3.2-3B-Instruct")
DATASETS=("cdr")
TRAIN_SET_NS=(10)
EVAL_SET="toy"
RUN_N=1

# Create array of all combinations
declare -A configs
index=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for train_set_n in "${TRAIN_SET_NS[@]}"; do
            configs[$index]="$model $dataset $train_set_n"
            ((index++))
        done
    done
done

# Generate and submit PBS jobs
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        identifier="finetune_$(date +%Y%m%d_%H%M%S)_config_${i}_run_${run_n}"
        read model dataset train_set_n <<< "${configs[$i]}"
        model_path="${model}"
        test_data_path="processed-data/${dataset}/${EVAL_SET}.jsonl"
        sample_path="processed-data/${dataset}/run_samples/train${run_n}_${train_set_n}.json"
        prompt_data_path="processed-data/${dataset}/train.jsonl"
        preds_path="output/ner/${dataset}/${model}_${run_n}_${train_set_n}_shot"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"
        # Create PBS script
        cat <<EOF > "finetune_pbs_files/${identifier}.pbs"
#!/bin/bash
### Job Name
#PBS -N finetune_${model}_${dataset}_${train_set_n}_${EVAL_SET}_run${run_n}
### Project code
#PBS -A ner_fewshot
### Maximum time this job can run before being killed (here, 1 day)
#PBS -l walltime=01:00:00:00
### Resource Request (must contain cpucore, memory, and gpu (even if requested amount is zero)
#PBS -l cpucore=2:memory=50gb:gpu=4
### Output Options (default is stdout_and_stderr)
#PBS -l outputMode=stdout_and_stderr


. /home/bulut/miniconda3/etc/profile.d/conda.sh

conda activate ft-vs-icl



### Run experiment
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python experiments/train_llm.py \
    --model_name_or_path $model_path \
    --data_path $prompt_data_path \
    --sample_path $sample_path \
    --bf16 True \
    --output_dir $preds_path \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --log_level info \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 5 \
    --optim adamw_torch \
    --per_device_train_batch_size 2 \
    --save_strategy "no" \
    --seed 42 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --report_to none \
    --gradient_checkpointing True \
    --bf16 True \
    --remove_unused_columns False
    
### Run inference
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python experiments/run_vllm.py \
            --ckpt_dir $preds_path \
            --tokenizer $model_path \
            --test_data $test_data_path \
            --output_dir $preds_path/source_preds.json \

### Run evaluation
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python evaluation/evaluate_ner.py \
    --generated_file $preds_path/source_preds.json \
    --model_name $model \
    --eval_set $EVAL_SET \
    --dataset $dataset \
    --run_n $run_n \
    --sample_n $train_set_n
EOF

        # Submit job
        qsub "finetune_pbs_files/${identifier}.pbs"
    done
done




