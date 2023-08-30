#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=2G

csv_path=$1 # ex: data/tm/Hie2022_C143_Tm.csv
score_method=$2 # ex: pyrosetta
device=${3:-cpu} # cuda:0

if [ "$score_method" = "esmif" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/inverse
fi

if [ "$score_method" = "pyrosetta" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/pyrosetta
fi

if [ "$score_method" = "mpnn" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/data_jgray21/mchungy1/miniconda3/envs/mlfold
fi

python scripts/score_struc.py $csv_path $score_method $device
