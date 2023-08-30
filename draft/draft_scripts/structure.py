from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EAQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEWVAGITPAGGYTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARFVFFLPYAMDYWGQGTLVTVSS",
    "L": "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYTTPPTFGQGTKVEIKR"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=False, # Renumber predicted antibody structure (Chothia)
)
