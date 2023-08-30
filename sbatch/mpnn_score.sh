#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=2GB

source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/data_jgray21/mchungy1/miniconda3/envs/mlfold

python scripts/mpnn_score.py structure/tm/Hie2022_C143_Tm/Hie2022_C143_Tm_0.pdb
