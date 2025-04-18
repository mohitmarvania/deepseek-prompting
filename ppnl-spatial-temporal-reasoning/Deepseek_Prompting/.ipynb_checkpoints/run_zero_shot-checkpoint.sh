#!/bin/bash
#SBATCH --job-name=deepseek-zero
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:2

#SBATCH --output=deepseek_job_zero.out
#SBATCH --error=deepseek_job_zero.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G
#SBATCH --time=9:00:00

module load gnu10
module load python
module load cuda

source /home/mmarvani/deepseek-py-env/bin/activate

python zero_shot_prompting.py
