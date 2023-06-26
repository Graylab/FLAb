#imports
import argparse
import torch
import pandas as pd
import pyrosetta
from igfold import IgFoldRunner, init_pyrosetta
import os
from extra import dir_create

init_pyrosetta()

"""
Input:  relative path to csv file (header format: heavy,light,fitness)
Output: IgFold-predicted structure
"""


def _get_args():
   """Gets command line arguments"""

   desc = ("""Script for predicting antibody Fv structures in csv format""")
   parser = argparse.ArgumentParser(description=desc)

   parser.add_argument("csv_path",
                      type=str,
                      help="csv file with heavy and light chain sequences.")
   parser.add_argument("--use_gpu",
                      default=False,
                      action="store_true")
   return parser.parse_args()


def _cli():
   args = _get_args()

   csv_path = args.csv_path

   device_type = 'cuda' if torch.cuda.is_available(
   ) and args.use_gpu else 'cpu'
   device = torch.device(device_type)

   # CREATE DIRECTORY PATH

   # split csv_path data/fitness/filename into variables
   dir_name, filename = os.path.split(csv_path)
   data_dir, fitness_dir = os.path.split(dir_name)

   # remove file extension from filename
   name_only, extension = os.path.splitext(filename)

   structure_dir = 'structure'
   # create score/
   dir_create(structure_dir)

   # check score/fitness
   dir_create(structure_dir, fitness_dir)

   # create score/fitness/csv
   dir_create(structure_dir, fitness_dir, name_only)

   df = pd.read_csv(csv_path)

   # change to the 'structure/dir_name/name_only/' directory
   output_dir = os.path.join('.', structure_dir, fitness_dir, name_only)
   os.chdir(output_dir)

   for row in range(len(df)):
      sequences = {
         "H": df['heavy'][row],
         "L": df["light"][row]
      }

      pred_pdb = "{}_{}.pdb".format(name_only, row)

      igfold = IgFoldRunner()
      igfold.fold(
         pred_pdb, # Output PDB file
         sequences=sequences, # Antibody sequences
         do_refine=True, # Refine the antibody structure with PyRosetta
         do_renum=False, # Renumber predicted antibody structure (Chothia)
      )

   """
   out.prmsd # Predicted RMSD for each residue's N, CA, C, CB atoms (dim: 1, L, 4)
   """

   # change the working directory back to original directory
   os.chdir('..')


if __name__=='__main__':
   _cli()
