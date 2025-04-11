#!/bin/bash



# clear created directories
rm -rf fewshot_pbs_files
rm -rf fewshot-results

# clear old jobs
qstat | awk 'NR>1 && ($1 ~ /^[rq]$/) && $5 == "bulut" {print $4}' | xargs -I {} qdel {}


# clear old job log files that are stdout.txt and stderr.txt
rm -rf job.*.stdout.txt
rm -rf job.*.stderr.txt

# create pbs file dir
mkdir -p fewshot_pbs_files



echo "Starting experiments"

# Define experiment parameters
LOCAL_MODEL_PATH="/media/networkdisk/bulut2/local-models"
MODELS=("Llama-3.1-8B-Instruct")
DATASETS=("cdr")
FEW_SHOT_NS=(1)
EVAL_SET="dev"
RUN_N=1

# Create array of all combinations
declare -A configs
index=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for few_shot in "${FEW_SHOT_NS[@]}"; do
            configs[$index]="$model $dataset $few_shot"
            ((index++))
        done
    done
done

# Generate and submit PBS jobs
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        identifier="fewshot_$(date +%Y%m%d_%H%M%S)_config_${i}_run_${run_n}"
        read model dataset few_shot <<< "${configs[$i]}"
        model_path="${LOCAL_MODEL_PATH}/${model}"
        test_data_path="processed-data/${dataset}/${EVAL_SET}.jsonl"
        sample_path="processed-data/${dataset}/run_samples/train${run_n}_${few_shot}.json"
        prompt_data_path="processed-data/${dataset}/train.jsonl"
        preds_path="output/ner/${dataset}/run_${run_n}_${few_shot}"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"
        # Create PBS script
        cat <<EOF > "fewshot_pbs_files/${identifier}.pbs"
#!/bin/bash
### Job Name
#PBS -N fewshot_${model}_${dataset}_${few_shot}_${EVAL_SET}_run${run_n}
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
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python experiments/run_vllm_prompting.py \
    --ckpt_dir $model_path \
    --tokenizer $model_path \
    --test_data $test_data_path \
    --prompt_data $prompt_data_path \
    --output_dir $preds_path/source_preds.json \
    --sample $sample_path \
    --tensor_parallel_size 4
### Run evaluation
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python evaluation/evaluate_ner.py \
    --generated_file $preds_path/source_preds.json \
    --model_name $model \
    --eval_set $EVAL_SET
EOF

        # Submit job
        qsub "fewshot_pbs_files/${identifier}.pbs"
    done
done