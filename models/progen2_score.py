import os
import time
import random
import sys
import math
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.expanduser('~/models'))
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

### utils
class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


### models
def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

def progen2_151M_small_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-small"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_764M_medium_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-medium"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_764M_oas_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-oas"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_764M_base_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-base"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_2p7B_large_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-large"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_2p7B_bfd90_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-BFD90"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


def progen2_6p4B_xlarge_score(sequence, device='cuda'):
    ### main
    set_env()
    set_seed(42, deterministic=True)

    device = torch.device(device)
    
    ckpt = os.path.expanduser("~/models/progen/progen2/checkpoints/progen2-xlarge"

    # Define fp16 initially
    fp16 = True  # Assume fp16 is True by default
    if device.type == 'cpu':
        fp16 = False

    model = create_model(ckpt=ckpt, fp16=fp16).to(device)

    tokenizer = create_tokenizer_custom(file=os.path.expanduser('~/models/progen/progen2/tokenizer.json')

    def log_likelihood(tokens):
        with torch.no_grad():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token+1)]
            target = target - first_token

            return -torch.nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target.view(-1)).item()

    # Calculate the log likelihood for the sequence
    ll_mean = log_likelihood(sequence)
    
    # Calculate perplexity
    perplexity = math.exp(-ll_mean)

    return perplexity


#seq_example = "GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDTRSIAVAGTNFDYWGQGTLVTVSS"
#print(progen2_151M_small_score(seq_example))
#print(progen2_764M_medium_score(seq_example))
#print(progen2_764M_oas_score(seq_example))
#print(progen2_764M_base_score(seq_example))
#print(progen2_2p7B_large_score(seq_example))
#print(progen2_2p7B_bfd90_score(seq_example))
#print(progen2_6p4B_xlarge(seq_example))


