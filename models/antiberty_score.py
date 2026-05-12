import math
import torch
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()

def antiberty_score(seq):
    """
    Calculate antiberty psuedo-PPL of a given sequence
    """
    sequences = [seq]
    with torch.no_grad():
        log_likelihood = antiberty.pseudo_log_likelihood(sequences, batch_size=16)[0].item()
    return math.exp(-log_likelihood)