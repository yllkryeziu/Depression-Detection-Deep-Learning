#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting ..."
autrainer postprocess results cnn10-binary-experiment-accuracy -a model
echo "Job finished."

