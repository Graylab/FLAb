#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

path_to_csv_structures=$1

parent_dir=$(dirname "$path_to_csv_structures")
fitness=$(basename "$parent_dir")
CSV_name=$(basename "$path_to_csv_structures")

output_dir=score/RosettaE/$fitness/$CSV_name/

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

# create list of pdb names
ls "$path_to_csv_structures"*.pdb > pdblist

# score pdb list
/home/mchungy1/scr4_jgray21/mchungy1/Rosetta/main/source/bin/score_jd2.linuxgccrelease -database /home/mchungy1/scr4_jgray21/mchungy1/Rosetta/main/database/ -l pdblist

# rename score file to CSV_name.sc
mv score.sc $CSV_name.sc

# move files to score directory
mv pdblist "$output_dir"
mv $CSV_name.sc "$output_dir"
