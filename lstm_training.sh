#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting ..."
python lstm_standalone.py
echo "Job finished."

