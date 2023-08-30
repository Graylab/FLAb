#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=2:0:0
#SBATCH --mem-per-cpu=2GB

csv_path=$1 # ex: data/binding/Hie2022_C143_Kd.csv
score_method=$2 # ex: iglm

if [ "$score_method" = "iglm" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/iglm
fi

if [ "$score_method" = "antiberty" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/antiberty
fi

python scripts/002_score_cst.py $csv_path $score_method
