#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting data copy..."
rsync -av --progress data/ExtendedDAIC-16k-fixed/patients/ data/ExtendedDAIC-16k-lstm/patients/
echo "Data copy finished." 