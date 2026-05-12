import numpy as np
import torch
import esm
import esm.inverse_folding
import uuid
import os.path

from igfold import IgFoldRunner

def esmif_score(sequence: str, chain_id: str = 'A') -> float:

    sequences = {
        chain_id: sequence,
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

    pdb_path = pred_pdb

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    coords, native_seq = esm.inverse_folding.util.load_coords(pdb_path, chain_id)

    ll, _ = esm.inverse_folding.util.score_sequence(
        model, alphabet, coords, native_seq
    )

    # Delete pdb and fasta
    os.remove(pred_pdb)
    os.remove(f"test_prediction_{unique_id}.fasta")

    return np.exp(-ll)


#score = esmif_score("GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDTRSIAVAGTNFDYWGQGTLVTVSS", "H")
#print(score)

