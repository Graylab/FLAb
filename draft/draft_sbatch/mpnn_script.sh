#!/bin/bash

path_to_csv_structures=$1

parent_dir=$(dirname "$path_to_csv_structures")
tm=$(basename "$parent_dir")
Hie=$(basename "$path_to_csv_structures")

output_dir=score/proteinMPNN/$tm/$Hie/

echo "tm: $tm"
echo "Hie: $Hie"
echo "output: $output_dir"
