import torch

from progen3.modeling import ProGen3ForCausalLM
from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.scorer import ProGen3Scorer

model = ProGen3ForCausalLM.from_pretrained("Profluent-Bio/progen3-3b", torch_dtype=torch.bfloat16)
model = model.eval().to("cuda:0")
batch_preparer = ProGen3BatchPreparer()

# Direct Usage
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

inputs = batch_preparer.get_batch_kwargs([sequence], device="cuda:0", reverse=False)
outputs = model(**inputs, return_dict=True)
print(outputs.logits)

# Usage with scorer (returns averaged log likelihood of forward and reverse direction)
# Would suggest using Scoring CLI below if scoring very large number of sequences
scorer = ProGen3Scorer(model=model)
scores = scorer.score_batch(sequences=[sequence])
print(scores["log_likelihood"][0])
