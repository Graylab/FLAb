#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem-per-cpu=2GB

source activate mlfold

path_to_csv_structures=$1 #structure/tm/Hie2022_C143_Tm/


fitness_dir = $(basename $(dirname "$path_to_fitness_dir"))
csv_name = $(basename "$path_to_fitness_dir")

output_dir="score/proteinMPNN"/"$fitness_dir"/"$csv_name"/
# scores/proteinMPNN/tm/Hie2022_C143_Tm/

# if [ ! -d $output_dir ]
# then
#     mkdir -p $output_dir
# fi

echo "fitness_dir: $fitness_dir"
echo "csv_name: $csv_name"
echo "output_dir: $output_dir"
