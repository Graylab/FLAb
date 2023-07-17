#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

csv_path=$1 # ex: data/tm/Hie2022_C143_Tm.csv
score_method=$2 # ex: iglm
progen_model=${3:-None} # optional args if score_method is progen
device=${4:-cpu} # cuda:0

if [ "$score_method" = "iglm" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/iglm
fi

if [ "$score_method" = "antiberty" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/antiberty
fi

if [ "$score_method" = "progen" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /scratch16/jgray21/mchungy1/conda_envs/progen
fi

python scripts/score_seq_mut.py $csv_path $score_method $progen_model $device
