#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /scratch16/jgray21/mchungy1/conda_envs/antiberty

python scripts/002_score_cst.py data/cst/sabdab.csv antiberty
