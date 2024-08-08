#!/bin/sh

#SBATCH -J FinLlama
#SBATCH -p gpu_2080ti
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./log/%x.o%j
#SBATCH -e ./log/%x.e%j
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1

module purge
# module load cuda/11.4
# module load conda/pytorch_1.10.0_cuda_11

CONDA_HOME=/apps/applications/miniconda3
source $CONDA_HOME/etc/profile.d/conda.sh

conda activate chanhyuk

# check CUDA driver version
nvcc --version

srun python -m bitsandbytes

# srun python ./resource/src/llamaTraining.py
# srun python ./resource/src/verifyCUDA.py