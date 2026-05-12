import numpy as np
from igfold import IgFoldRunner
import uuid
import os

import pyrosetta
from pyrosetta import init, pose_from_pdb, get_score_function
from pyrosetta.rosetta.protocols.relax import FastRelax

# Initialize PyRosetta
pyrosetta.init(extra_options="-mute all")  # mute output for clarity

def pyrosetta_score(sequence, chain):
    """
    Predict structure, relax, and calculate PyRosetta energy
    """
    sequences = {chain: sequence}
    
    # Create unique filenames
    unique_id = uuid.uuid4()
    pred_pdb = f"test_prediction_{unique_id}.pdb"
    pred_fasta = f"test_prediction_{unique_id}.fasta"
    
    # Run IgFold
    igfold = IgFoldRunner()
    igfold.fold(pred_pdb, sequences=sequences, do_refine=False, do_renum=False)
    
    # Load PDB into Pose
    pose = pose_from_pdb(pred_pdb)
    
    # Relax the Pose
    scorefxn = get_score_function(True)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)
    
    # Compute energy after relaxation
    energy = scorefxn(pose)
    
    # Delete pdb and fasta
    os.remove(pred_pdb)
    os.remove(f"test_prediction_{unique_id}.fasta")
    
    return energy

# Example usage
#energy = pyrosetta_score(
#    "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
#    "H"
#)
#print("Total energy:", energy)
