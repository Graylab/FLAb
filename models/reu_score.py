import uuid
import os.path
import pyrosetta
from pyrosetta.teaching import *
from pyrosetta import pose_from_pdb, MoveMap
from pyrosetta.rosetta.protocols.relax import FastRelax
import os
from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta

init_pyrosetta()

def pyrosetta_score(sequence: str, chain_id: str) -> float:

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

    # Load into Pose
    pose = pose_from_pdb(pred_pdb)

    # Apply FastRelax
    scorefxn = get_score_function(True)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)

    # Score the relaxed pose
    total_score = scorefxn(pose)


    # Delete pdb and fasta
    os.remove(pred_pdb)
    os.remove(f"test_prediction_{unique_id}.fasta")

    return(total_score)

score = pyrosetta_score("GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDTRSIAVAGTNFDYWGQGTLVTVSS", "H")
print(score)


