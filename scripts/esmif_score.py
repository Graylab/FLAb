import os
import argparse

## Verify that pytorch-geometric is correctly installed
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

# load model
import esm
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
# use eval mode for deterministic output e.g. without random dropout
model = model.eval()

def _get_args():
    """Gets command line arguments"""

    desc = ("""Script for scoring antibody sequences""")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("pdb_path", # ex: structure/binding/Hie2022_C143_Kd/Hie2022_C143_Kd_0.pdb
                      type=str,
                      help="pdb")

    return parser.parse_args()

def _cli():
    args = _get_args()

    # load pdb
    pdb_path = args.pdb_path

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

    # split pdb pathway into directory components
    directories = []
    while True:
        pdb_path, directory = os.path.split(pdb_path)
        if directory != "":
            directories.append(directory)
        else:
            if pdb_path != "":
                directories.append(pdb_path)
            break

    # reverse list to get directories in correct order
    directories = directories[::-1]
    
    # save required directories as variables
    fitness_dir = directories[1]
    csv_name = directories[2]
    pdb_name = os.path.splitext(directories[3])[0]

    # DIRECTORY CREATION

    score_dir = 'score'
    # create "score/" if doesn't exist
    if not os.path.exists(os.path.join('.', score_dir)):
        os.mkdir(os.path.join('.', score_dir))
    
    score_method='esm_if'
    # create "score/esm_if" if doesn't exist
    if not os.path.exists(os.path.join('.', score_dir, score_method)):
        os.mkdir(os.path.join('.', score_dir, score_method))

    # create "score/esm_if/fitness_dir" if it doesn't exist
    if not os.path.exists(os.path.join('.', score_dir, score_method, fitness_dir)):
        os.mkdir(os.path.join('.', score_dir, score_method, fitness_dir))

    # create "score/esm_if/fitness_dir" if it doesn't exist
    if not os.path.exists(os.path.join('.', score_dir, score_method, fitness_dir, csv_name)):
        os.mkdir(os.path.join('.', score_dir, score_method, fitness_dir, csv_name))
    
    with open(f"{pdb_name}.csv", "w") as f:
        f.write("{},{}\n".format('pdb_name', 'll'))
        f.write("{},{}\n".format(pdb_name, ll_avg))


if __name__=='__main__':
    _cli()

