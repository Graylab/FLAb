import math
from iglm import IgLM

iglm = IgLM()

def iglm_score(seq, chain_token):
    """
    Calculate iglm PPL of a given sequence
    """
    log_likelihood = iglm.log_likelihood(seq, chain_token, "[HUMAN]")
    return math.exp(-log_likelihood)



