import re
import os

# directory creator
def dir_create(*args):
  path = os.path.join('.', *args)
  if not os.path.exists(path):
    print(f'Directory "{path}" does not exist, creating directory')
    os.mkdir(path)
  else:
    print(f'Directory "{path}" exists already')

# custom function that extracts numerical part of each pdb_name
def extract_last_digits(value):
  last_digits = re.search(r'\d+$', value)
  return int(last_digits.group(0))
