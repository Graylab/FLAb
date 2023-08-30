#!/bin/bash

#SBATCH --nodes=1 # request one node
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --account=jgray21_gpu

fitness_path=$1 # ex: data/tm/
score_method=$2 # ex: progen
model_version=$3
device=$4

if [ "$score_method" = "iglm" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/iglm
fi

if [ "$score_method" = "antiberty" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/antiberty
fi

if [ "$score_method" = "progen" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /scratch16/jgray21/mchungy1/conda_envs/progen
fi

for CSV in $fitness_path*; do python scripts/002_score.py $CSV $score_method $model_version $device; done
