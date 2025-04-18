#!/bin/bash
#SBATCH --job-name=deepseek-few
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:2

#SBATCH --output=deepseek_job_few.out
#SBATCH --error=deepseek_job_few.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G
#SBATCH --time=1-24:00:00

module load gnu10
module load python
module load cuda

source /home/mmarvani/deepseek-py-env/bin/activate

python few_shot_prompting.py
