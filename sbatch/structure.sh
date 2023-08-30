#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=2GB

source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/igfold

csv_path=$1

python scripts/structure.py $csv_path
