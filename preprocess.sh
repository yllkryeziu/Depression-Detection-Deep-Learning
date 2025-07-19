#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=.slurm_outputs/slurm_%A.out

echo "Starting ..."
python extract_cnn_features.py --data_path data/ExtendedDAIC-lstm --output_path data/ExtendedDAIC-lstm/ --model_path cnn10model.pt
echo "Job finished."

