#!/bin/bash
#SBATCH --job-name=LLMFinetune
#SBATCH --partition=ice-gpu
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/GRet-%j.out
#SBATCH --error=slurm/GRet-%j.err
cd $SLURM_SUBMIT_DIR
export HF_HOME="/home/hice1/av84/scratch/hf_cache"
echo "Starting Python script: gretriever.py"
PYTHON_EXEC="/home/hice1/av84/scratch/conda_envs/mlg_/bin/python"
srun $PYTHON_EXEC gretriever.py
echo "Script finished."