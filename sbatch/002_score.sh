#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --mem-per-cpu=2GB

fitness_path=$1 # ex: data/tm/
score_method=$2 # ex: iglm

if [ "$score_method" = "iglm" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/iglm
fi

if [ "$score_method" = "antiberty" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/antiberty
fi

for CSV in $fitness_path*; do python scripts/002_score.py $CSV $score_method; done
