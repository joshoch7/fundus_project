#!/bin/bash
#
#SBATCH --job-name="explain"
#SBATCH -p owners,normal,mrivas
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user="joch@stanford.edu"
unset XDG_RUNTIME_DIR
export PATH="../../../../software/anaconda3/bin:$PATH" # Specify anaconda3 path
export PATH="../../../../software/anaconda3/condabin:$PATH"
source ../../../../software/anaconda3/etc/profile.d/conda.sh # Find the anaconda3 folder to source 
conda activate age
python3 ../py_files/explain.py
