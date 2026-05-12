import numpy as np
from igfold import IgFoldRunner
import uuid
import os

def igfold_score(sequence, chain):
    """
    Calculate the average RMSD of the predicted structure for a given single protein sequence.
    :param sequence: A single protein sequence string.
    :param chain: The chain identifier for the sequence.
    :return: Average RMSD value.
    """
    sequences = {
        chain: sequence,
    }
    
    # Create a unique identifier using UUID
    unique_id = uuid.uuid4()
    pred_pdb = f"test_prediction_{unique_id}.pdb"
    
    igfold = IgFoldRunner()
    
    out = igfold.fold(
        pred_pdb,
        sequences=sequences,
        do_refine=False,
        do_renum=False,
    )
    
    # out.prmsd contains the predicted RMSD values
    prmsd = out.prmsd  # Shape: (1, L, 4), we take the first item and average over residues.
    
    # Move prmsd to CPU before converting to numpy
    prmsd_cpu = prmsd.cpu()  # If prmsd is a torch tensor
    
    # Calculate the average RMSD for the backbone atoms (N, CA, C)
    average_rmsd = np.mean(np.linalg.norm(prmsd_cpu[0, :, :3].detach().numpy(), axis=1))
    
    # Delete pdb and fasta
    os.remove(pred_pdb)
    os.remove(f"test_prediction_{unique_id}.fasta")
    
    return average_rmsd

# Example usage:
average_score = igfold_score("EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS", "H")
print("Average pRMSD:", average_score)
