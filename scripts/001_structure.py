#imports
import argparse
import torch
import pandas as pd
import pyrosetta
from igfold import IgFoldRunner, init_pyrosetta
import os

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

   # split csv_path into the directory path and filename
   dir_path, filename = os.path.split(csv_path)
   # remove file extension from filename
   name_only, extension = os.path.splitext(filename)
   # print filename without directory path and extension
   print(f'The filename is {name_only}')
   # Split the directory path into the parent directory and directory name
   parent_dir, dir_name = os.path.split(dir_path)
   # print directory name without full path
   print(f'The directory name is {dir_name}')

   structure_dir='structure'
   # check if the 'structure' directory exists, create if it doesn't
   if not os.path.exists(os.path.join('.', structure_dir)):
      print(f'Directory "{structure_dir}" does not exist, creating directory')
      os.mkdir(os.path.join('.', structure_dir))
   else:
      print(f'Directory "{structure_dir}" exists already')

   # check if the 'binding' directory exists inside 'structure/', create if it doesn't
   if not os.path.exists(os.path.join('.', structure_dir, dir_name)):
      print(f'Directory "{structure_dir}/{dir_name}" does not exist, creating directory')
      os.mkdir(os.path.join('.', structure_dir, dir_name))
   else:
      print(f'Directory "{structure_dir}/{dir_name}" exists already')

   # check if the 'name_only' directory exists inside 'structure/dir_name/', create if it doesn't
   if not os.path.exists(os.path.join('.', structure_dir, dir_name, name_only)):
      print(f'Directory "{structure_dir}/{dir_name}/{name_only}" does not exist, creating directory')
      os.mkdir(os.path.join('.', structure_dir, dir_name, name_only))
   else:
      print(f'Directory "{structure_dir}/{dir_name}/{name_only}" exists already')

   device_type = 'cuda' if torch.cuda.is_available(
   ) and args.use_gpu else 'cpu'
   device = torch.device(device_type)

   df = pd.read_csv(csv_path)

   # change to the 'structure/dir_name/name_only/' directory
   output_dir = os.path.join('.', structure_dir, dir_name, name_only)
   os.chdir(output_dir)

   for row in range(len(df)):
      sequences = {
         "H": df['heavy'][row],
         "L": df["light"][row]
      }
      """
      pred_pdb = "{}_{}.pdb".format(name_only, row)

      igfold = IgFoldRunner()
      igfold.fold(
         pred_pdb, # Output PDB file
         sequences=sequences, # Antibody sequences
         do_refine=True, # Refine the antibody structure with PyRosetta
         #do_renum=True, # Renumber predicted antibody structure (Chothia)
      )
      """
      print(sequences)
      print()

   """
   out.prmsd # Predicted RMSD for each residue's N, CA, C, CB atoms (dim: 1, L, 4)
   """
   
   # change the working directory back to original directory
   os.chdir('..')


if __name__=='__main__':
   _cli()
