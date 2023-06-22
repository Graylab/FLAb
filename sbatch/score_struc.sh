#!/bin/bash

#SBATCH --nodes=1 # request one node
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --account=jgray21_gpu

csv_path=$1 # ex: data/tm/Hie2022_C143_Tm.csv
score_method=$2 # ex: iglm
device=${3:-cpu} # cuda:0

if [ "$score_method" = "inverse" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/inverse
fi

python scripts/score_struc.py $csv_path $score_method $device
