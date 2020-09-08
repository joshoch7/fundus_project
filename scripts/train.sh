#!/bin/bash
#
#SBATCH --job-name="train"
#SBATCH -p owners,gpu,mrivas
#SBATCH -G 1
#SBATCH -C GPU_MEM:12GB
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user="joch@stanford.edu"
unset XDG_RUNTIME_DIR
export PATH="../../../../software/anaconda3/bin:$PATH" # Specify anaconda3 path
export PATH="../../../../software/anaconda3/condabin:$PATH"
source ../../../../software/anaconda3/etc/profile.d/conda.sh # Find the anaconda3 folder to source 
conda activate age

ml load cudnn/7.6.5   
ml load cuda/10.1.168

python3 ../py_files/train.py
