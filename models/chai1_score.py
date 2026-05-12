import logging
import shutil
from pathlib import Path
import numpy as np
import torch
import uuid
from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)

def chai1_score(sequence):
    # Check if the input sequence is not empty
    if not sequence:
        raise ValueError("The protein sequence cannot be empty.")

    # Create a FASTA representation of the sequence
    fasta_content = f">protein|name=input-antibody-chain\n{sequence}"
    fasta_path = Path("/tmp/example.fasta")
    fasta_path.write_text(fasta_content)

    # Generate a unique output directory name using UUID
    unique_id = str(uuid.uuid4())
    output_dir = Path(f"/tmp/outputs_{unique_id}")

    # Create the output directory
    output_dir.mkdir(exist_ok=True)

    # Specify the device as a torch.device
    device = torch.device("cuda:0")  # You can change "cuda:0" to other indices if needed

    # Run inference
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=42,
        device=device,
        use_esm_embeddings=True,
    )

    # Load scores from the NPZ file
    scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))
    ptm = scores['ptm'][0]

    # Remove the output directory after usage
    shutil.rmtree(output_dir)
    logging.info(f"Removed output directory: {output_dir}")
    
    return ptm

# Example usage:
#score = chai1_score("GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDTRSIAVAGTNFDYWGQGTLVTVSS")
#print(score)

