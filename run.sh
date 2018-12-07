#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --account=ece-gpu-high
#SBATCH -p ece-gpu-high --gres=gpu:1
#SBATCH -c 6
srun singularity exec --nv ~dec18/Containers/tfgpu.simg python $HOME/Deep-Trading-Agent/DeepTradingAgent.py
