import os
import math

# esm if
try:
    ## Verify that pytorch-geometric is correctly installed
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing

    # load model
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    
    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

except ImportError:
    pass

# pyrsetta energy
try:
    import pyrosetta
    from pyrosetta.teaching import *
    pyrosetta.init()

except ImportError:
    pass


def esmif_score(pdb):

    # load pdb
    pdb_path = pdb
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    # load chains
    fpath = pdb_path
    chain_ids = ['H', 'L']
    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    # conditional sequence log-likelihoods for given BB coordinates
    target_chain_id = 'H'
    target_seq = native_seqs[target_chain_id]
    ll_h, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    target_chain_id = 'L'
    target_seq = native_seqs[target_chain_id]
    ll_l, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    ll_avg =(ll_l + ll_h) / 2

    perplexity = math.exp(-ll_avg)
    
    return perplexity

def pyrosetta_score(pdb):

    # load pdb into pose
    pose = pyrosetta.pose_from_pdb(pdb)

    sfxn = get_score_function(True)

    return(sfxn(pose))