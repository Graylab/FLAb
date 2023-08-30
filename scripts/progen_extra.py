import os
import time
import random
import sys

import torch

from tokenizers import Tokenizer
sys.path.append('/home/mchungy1/scr16_jgray21/mchungy1')
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

import subprocess
import os


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


### sample
def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):

    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]


### likelihood
def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)

def log_likelihood(logits, target, reduction='mean'):
    return -cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction=reduction)

def log_likelihood_custom_1(logits, target, reduction='mean'):
    return -torch.nn.functional.nll_loss(input=torch.log_softmax(logits, dim=1), target=target, reduction=reduction)

def log_likelihood_custom_2(logits, target, reduction='mean'):
    assert len(target.shape) == 1
    assert logits.shape[0] == target.shape[0]

    log_likelihood = 0.0
    n = logits.shape[0]
    for i in range(n):
        log_likelihood += torch.log_softmax(logits, dim=1)[i, target[i]] / (1. if reduction == 'sum' else n)
    return log_likelihood

