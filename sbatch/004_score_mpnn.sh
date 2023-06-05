
#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GBq

source activate mlfold

path_to_PDB=$1
# ex: /inputs/PDB_complexes/pdbs/3HTN.pdb

output_dir=$2
# ex: /outputs/example_3_score_only_outputs

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design="A B"

python /home/mchungy1/scr4_jgray21/mchungy1/AbDesign/models/ProteinMPNN/protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 10 \
        --sampling_temp "0.1" \
        --score_only 1 \
        --seed 37 \
        --batch_size 1
