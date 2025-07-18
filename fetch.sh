#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=4000
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting ..."
autrainer fetch -cn config-original.yaml
echo "Job finished."

