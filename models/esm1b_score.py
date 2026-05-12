import torch
import numpy as np
from transformers import EsmTokenizer, EsmForMaskedLM

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def esm1b_650M_score(seq: str) -> float:
    """
    calculate the perplexity of a sequence using ESM-2 650M model
    """
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    model = EsmForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device)
    model.eval()

    # Encode the input sequence with special tokens ([CLS], [SEP])
    tensor_input = tokenizer.encode(seq, return_tensors='pt')
    
    # Repeat the input to create masked versions for each token (excluding [CLS] and [SEP])
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    
    # Create a mask for each position (except [CLS] and [SEP])
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    
    # Replace each token with [MASK] in the corresponding masked_input
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    
    # Create labels: only compute loss for the masked positions
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    
    with torch.no_grad():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    
    # Convert loss to perplexity
    return float(np.exp(loss.item()))

#sequence = "QVQLQQSGAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGAIDPETGGTAYNQKFKGKAILTADKSSSTAYMELRSLTSEDSAVYYCTRHYYGSSFFDYWGQGTTLTVPS"
#print(esm1b_650M_score(sequence))

