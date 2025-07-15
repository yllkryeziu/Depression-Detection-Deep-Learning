#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting ..."
autrainer fetch
echo "Job finished."

