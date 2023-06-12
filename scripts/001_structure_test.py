#imports
import argparse
import torch
import pandas as pd
import pyrosetta
from igfold import IgFoldRunner, init_pyrosetta
import os

init_pyrosetta()

sequences = {
   #"H": 'EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS',
   "H": 'EAQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEWVAGITPAGGYTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARFVFFLPYAMDYWGQGTLVTVSS',
   #"L": 'DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK'
   "L": 'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYTTPPTFGQGTKVEIKR'
}
pred_pdb = "{}_{}.pdb".format('test', '01')

igfold = IgFoldRunner()
igfold.fold(
   pred_pdb, # Output PDB file
   sequences=sequences, # Antibody sequences
   do_refine=True, # Refine the antibody structure with PyRosetta
   do_renum=True, # Renumber predicted antibody structure (Chothia)
)
