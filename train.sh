#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --output=.slurm_outputs/slurm_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yll.kryeziu@tum..de

echo "Starting ..."
autrainer train -cn config-fixed.yaml
echo "Job finished."

