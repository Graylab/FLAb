#imports
import argparse
import torch
import pandas as pd
import os
import numpy as np
import csv

def _get_args():
   """Gets command line arguments"""

   desc = ("""Script for converting npz files of proteinMPNN to csv""")
   parser = argparse.ArgumentParser(description=desc)

   parser.add_argument("npz_path", # like scores/proteinMPNN/binding/Hie2022_C143_Kd/score_only/
                      type=str,
                      help="npz file pwd from proteinMPNN")

   parser.add_argument("csv_name", # current directory, .
                      type=str,
                      help="outpath for csv")

   return parser.parse_args()


def _cli():
   args = _get_args()

   npz_path = args.npz_path
   csv_name = args.csv_name

   # list of all npz files
   npz_files = [f for f in os.listdir(npz_path) if f.endswith('.npz')]

   # empty list to hold data
   data = []

   # loop through all npz files
   for npz_file in npz_files:
      # load file
      npz_data = np.load(os.path.join(npz_path, npz_file))

      # add new row to data list with the npz file name and number
      data.append([npz_file, npz_data['score'][0]])

   # write data to a new csv file
   with open(f"{csv_name}_proteinMPNN.csv", "w") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['filename', "score"])
      writer.writerows(data)


if __name__=='__main__':
    _cli()