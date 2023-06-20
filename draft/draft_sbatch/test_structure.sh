#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate igfold

python scripts/001_structure_test.py
