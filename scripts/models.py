import math

# iglm
import iglm
from iglm import IgLM
iglm = IgLM()

def iglm_score(df):
    """
    input: df with columns: heavy,
                            light,
                            fitness
    output: df with columns: heavy,
                             light,
                             fitness,
                             heavy_perplexity,
                             light_perplexity,
                             average_perplexity
    """
    
    # score heavy sequences
    heavy_score = []

    for seq in df['heavy']:
        sequence = seq
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )
            
        perplexity = math.exp(-log_likelihood)
        heavy_score.append(perplexity)

    df['heavy_perplexity'] = heavy_score

    # score light sequences
    light_score = []

    for seq in df['light']:
        sequence = seq
        chain_token = "[LIGHT]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )

        perplexity = math.exp(-log_likelihood)
        light_score.append(perplexity)

    df['light_perplexity'] = light_score

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df