#!/bin/bash

name=$1
path=${2:-"domainbed/data"}
dataset=${3:-"PACS"}
model=${4:-"nf_vitbase_moek_every_qkv_new"}

# hparams range
n_hparams_from=${5:-5}
n_hparams=${6:-10}

seed=${7:-42}
n_trials=${8:-3}
gpu_id=${9:-0}

# jobs management
hparams_range=$(($n_hparams - $n_hparams_from))
n_hparams_to=$(($n_hparams - 1))
n_jobs=$(( $n_trials * $hparams_range ))

pending_jobs=0
completed_jobs=0
for trial in $(seq 1 $n_trials); do
    for hparams_seed in $(seq $n_hparams_from $n_hparams_to); do
        output_dir="sweep_output/${dataset}/${model}/${name}/MoA_sweep_h_${hparams_seed}_t_${trial}"
        if [ ! -f "$output_dir/done" ]; then
            pending_jobs=$(( pending_jobs + 1 ))
        else
            completed_jobs=$(( completed_jobs + 1 ))
        fi
    done
done

echo "$n_jobs jobs: $completed_jobs done, $pending_jobs not launched"
echo "About to launch $pending_jobs jobs."
echo "Launching..."

seed=$(( $seed + $n_hparams_from * $n_trials ))

# sweep loop
for trial in $(seq 0 $((n_trials - 1)))
do
    echo "Starting trial $trial..."
    job=0

    for hparams_seed in $(seq $n_hparams_from $n_hparams_to)
    do

        output_dir="sweep_output/NEW/${dataset}/${model}/${name}/MoA_sweep_h_${hparams_seed}_t_${trial}"
        if [ -f "$output_dir/done" ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Skipping job $job trial $trial (already done)"
            job=$((job + 1))
            continue
        fi

        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Launching Job $job/$hparams_range trial $trial/$n_trials, pending jobs $pending_jobs..."

        CUDA_VISIBLE_DEVICES=$gpu_id python train_all.py $name \
            --data_dir $path \
            --output_dir $output_dir \
            --algorithm ERM \
            --dataset $dataset \
            --model $model \
            --l_aux \
            --seed $seed \
            --trial_seed $trial \
            --hparams_seed $hparams_seed \
            --es_val=-1 \
            --lodoval \
            --holdout_fraction=0.0

        if [ $? -eq 0 ]; then
            touch "$output_dir/done"
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job $job/$hparams_range trial $trial/$n_trials Completed. Results saved in $output_dir"
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job $job/$hparams_range trial $trial/$n_trials FAILED. Check logs."
        fi

        ((job++))
        ((pending_jobs--))
        ((seed++))

    done

    echo "Completed jobs for trial $trial"

done

echo "All $n_jobs jobs completed."
