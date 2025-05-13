#!/bin/bash

# MAJOR PATHS

HOME_DIR="/home/bulut/repositories/fine-tune-vs-few-shot"
ASSETS_DIR="/media/turtle-mechanical/bulut2/assets/fine-tune-vs-few-shot" # where datasets and outputs are stored for this project
OUTPUT_DIR=$ASSETS_DIR"/output/fewshot/re"
DATA_DIR=$ASSETS_DIR"/processed-data"
CACHE_DIR="/media/turtle-mechanical/bulut2/.cache"


# clear old jobs
#qstat | awk 'NR>1 && ($1 ~ /^[rq]$/) && $5 == "bulut" {print $4}' | xargs -I {} qdel {}
#sleep 5
#nvidia-smi | awk '/python/ {print $5}' | xargs -I {} kill -9 {}
#sleep 5
# wait 5 seconds
#sleep 5
# clear previously created directories
rm -rf $HOME_DIR/fewshot_pbs_files
rm -rf $OUTPUT_DIR

# clear old job log files that are stdout.txt and stderr.txt
rm -rf job.*.stdout.txt
rm -rf job.*.stderr.txt

# create pbs file dir
mkdir -p fewshot_pbs_files

echo "Starting experiments"

# Define experiment parameters

GEN_CONFIG_PATH=$HOME_DIR"/gen_config.json"
MAIN_PROMPT_PATH=$ASSETS_DIR"/improved_instruction.txt"
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct" 
    "Qwen/Qwen2.5-3B-Instruct" 
    "Qwen/Qwen2.5-7B-Instruct" 
    "Qwen/Qwen2.5-14B-Instruct" 
    "Qwen/Qwen2.5-32B-Instruct"
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

# Generate and submit PBS jobs
for i in $(seq 0 $((${#configs[@]} - 1))); do
    for run_n in $(seq 0 $((${RUN_N}-1))); do
        # unpack model, source/target datasets, shot count, and gen_config_type
        read model source_dataset target_dataset few_shot gen_config_type <<< "${configs[$i]}"
        identifier="fewshot_$(date +%Y%m%d_%H%M%S)_config_${i}_run_${run_n}"
        model_path="${model}"
        model_name="${model##*/}"
        test_data_path="$DATA_DIR/${target_dataset}/${target_dataset}_${EVAL_SET}.jsonl"
        sample_path="$DATA_DIR/${source_dataset}/run_samples/${source_dataset}_train_${run_n}_${few_shot}.json"
        prompt_data_path="$DATA_DIR/${source_dataset}/${source_dataset}_train.jsonl"
        preds_path=$OUTPUT_DIR"/${source_dataset}_${target_dataset}/${model_name}/${model_name}_${run_n}_${few_shot}_shot_${gen_config_type}_config"
        input_texts_save_path=$ASSETS_DIR"/input-texts/${source_dataset}_${target_dataset}_run_${run_n}_${few_shot}_shot/"
        # Create output directory if it doesn't exist, delete if it does
        if [ -d "$preds_path" ]; then   
            rm -rf "$preds_path"
        fi
        mkdir -p "$preds_path"

        
        # Create PBS script
        cat <<EOF > "fewshot_pbs_files/${identifier}.pbs"
#!/bin/bash
### Job Name
#PBS -N fewshot_${model_name}_${source_dataset}_${target_dataset}_${few_shot}_${gen_config_type}_${EVAL_SET}_run${run_n}
### Project code
#PBS -A re_fewshot
### Maximum time this job can run before being killed (here, 1 day)
#PBS -l walltime=01:00:00:00
### Resource Request (must contain cpucore, memory, and gpu (even if requested amount is zero)
#PBS -l cpucore=5:memory=50gb:gpu=4
### Output Options (default is stdout_and_stderr)
#PBS -l outputMode=stdout_and_stderr


. /home/bulut/miniconda3/etc/profile.d/conda.sh
export XDG_CACHE_HOME=$CACHE_DIR
conda activate ft-vs-icl

### Run experiment
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python $HOME_DIR/experiments/run_vllm_prompting.py \
    --ckpt_dir $model_path \
    --tokenizer $model_path \
    --test_data $test_data_path \
    --prompt_data $prompt_data_path \
    --gen_config_type $gen_config_type \
    --output_dir $preds_path/source_preds.json \
    --sample $sample_path \
    --gen_config_path $GEN_CONFIG_PATH \
    --tensor_parallel_size 4 \
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
EOF

        # Submit job
        qsub "fewshot_pbs_files/${identifier}.pbs"
    done
done
# copy this file ("bash/relation_extraction_few_shot/re_few_shot.sh") to OUTPUT_DIR
cp $HOME_DIR/bash/relation_extraction_few_shot/re_few_shot.sh $OUTPUT_DIR/re_few_shot.sh