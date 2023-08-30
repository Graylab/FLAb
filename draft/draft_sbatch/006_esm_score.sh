#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

source activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/inverse

path_to_pdb = $1 # ex: structure/binding/Hie2022_C143_Kd/

for PDB in $path_to_pdb*.pdb; do python scripts/esmif_score.py $PDB; done
