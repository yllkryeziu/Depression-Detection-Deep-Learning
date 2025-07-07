#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5000
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting Python training script..."
python --version
echo "Script finished."

