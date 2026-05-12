import esm
import numpy as np
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from esm2_score import esm2_650M_score
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ism_650M_uc30pdb_score(seq: str) -> float:
    """
    Calculate the pseudo-perplexity of a sequence using ESM-2 model
    loaded from a specified checkpoint.
    """
    # Ensure that the model is loaded and available in the global scope
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    ckpt = torch.load(os.path.expanduser('~/models/ism/ism_t33_650M_uc30pdb/checkpoint.pth')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    encoded_seq = torch.tensor(alphabet.encode(seq)).unsqueeze(0).to(device)  # (1, L)
    
    repeat_input = encoded_seq.repeat(encoded_seq.size(1) - 2, 1)  # (L-2, L)
    
    mask = torch.ones(encoded_seq.size(1) - 1, device=device).diag(1)[:-2]
    
    # important: use alphabet.mask_idx, not tokenizer.mask_token_id
    masked_input = repeat_input.masked_fill(mask == 1, alphabet.mask_idx)
    
    labels = repeat_input.clone()
    labels[masked_input != alphabet.mask_idx] = -100  # Ignore loss except masked position

    with torch.no_grad():
        outputs = model(masked_input)  # forward() returns a dictionary
        logits = outputs["logits"]  # shape (batch_size, seq_len, vocab_size)
        
        # Compute loss manually
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            labels.view(-1),                   # (batch_size * seq_len)
            ignore_index=-100,
            reduction="mean",
        )

    return float(np.exp(loss.item()))

def ism_650M_uc30_score(seq: str) -> float:
    """
    Calculate the pseudo-perplexity of a sequence using ESM-2 model
    loaded from a specified checkpoint.
    """
    # Ensure that the model is loaded and available in the global scope
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    ckpt = torch.load(os.path.expanduser('~/models/ism/ism_t33_650M_uc30/checkpoint.pth')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    encoded_seq = torch.tensor(alphabet.encode(seq)).unsqueeze(0).to(device)  # (1, L)
    
    repeat_input = encoded_seq.repeat(encoded_seq.size(1) - 2, 1)  # (L-2, L)
    
    mask = torch.ones(encoded_seq.size(1) - 1, device=device).diag(1)[:-2]
    
    # important: use alphabet.mask_idx, not tokenizer.mask_token_id
    masked_input = repeat_input.masked_fill(mask == 1, alphabet.mask_idx)
    
    labels = repeat_input.clone()
    labels[masked_input != alphabet.mask_idx] = -100  # Ignore loss except masked position

    with torch.no_grad():
        outputs = model(masked_input)  # forward() returns a dictionary
        logits = outputs["logits"]  # shape (batch_size, seq_len, vocab_size)
        
        # Compute loss manually
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            labels.view(-1),                   # (batch_size * seq_len)
            ignore_index=-100,
            reduction="mean",
        )

    return float(np.exp(loss.item()))

def ism_3B_uc30_score(seq: str) -> float:
    """
    Calculate the pseudo-perplexity of a sequence using ESM-2 model
    loaded from a specified checkpoint.
    """
    # Ensure that the model is loaded and available in the global scope
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    ckpt = torch.load(os.path.expanduser('~/models/ism/ism_t36_3B_uc30/checkpoint.pth')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")

    encoded_seq = torch.tensor(alphabet.encode(seq)).unsqueeze(0).to(device)  # (1, L)
    
    repeat_input = encoded_seq.repeat(encoded_seq.size(1) - 2, 1)  # (L-2, L)
    
    mask = torch.ones(encoded_seq.size(1) - 1, device=device).diag(1)[:-2]
    
    # important: use alphabet.mask_idx, not tokenizer.mask_token_id
    masked_input = repeat_input.masked_fill(mask == 1, alphabet.mask_idx)
    
    labels = repeat_input.clone()
    labels[masked_input != alphabet.mask_idx] = -100  # Ignore loss except masked position

    with torch.no_grad():
        outputs = model(masked_input)  # forward() returns a dictionary
        logits = outputs["logits"]  # shape (batch_size, seq_len, vocab_size)
        
        # Compute loss manually
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            labels.view(-1),                   # (batch_size * seq_len)
            ignore_index=-100,
            reduction="mean",
        )

    return float(np.exp(loss.item()))


#example = "GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDTRSIAVAGTNFDYWGQGTLVTVSS"
#print(ism_650M_uc30pdb_score(example))
#print(ism_650M_uc30_score(example))
#print(ism_3B_uc30_score(example))

