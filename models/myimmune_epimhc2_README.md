# MyImmune EpiMHC2 v10.0.0 (Hosted API Client)

**MyImmune EpiMHC2 v10.0.0** is a 26-model deep learning meta-ensemble for HLA Class II
(`HLA-DR`, `HLA-DQ`, `HLA-DP`) peptide presentation and therapeutic antibody CD4+ T-cell
immunogenicity deconvolution, developed by **MyImmune** ([myimmune.ai](https://myimmune.ai)).

### Key architecture and methods
1. **Dual-orientation groove epistasis:** vectorized evaluation of forward ($N \to C$) and
   inverted ($C \to N$) 9-mer binding registers with asymmetric pocket epistasis.
2. **27-allele global population coverage:** the standard Greenbaum et al. (2011) `IEDB-27`
   reference panel (>97% global population coverage).
3. **Observed Antibody Space (OAS) natural tolerance:** lossless repertoire indexing across
   5,467 OAS NGS runs (4,126 human IGH, 1,561 human IGL) with 140-subfamily xenogeneic
   log-likelihood ratios, to suppress framework false positives.

### Zero-shot results on `immunogenicity/Prihoda2021_mAb_immunogenicity`

N = 217. Fitness is the dataset's `%ADA response`. Both modes are zero-shot: no FLAb data was
used in training, and no per-dataset fitting was performed.

| method | Pearson $r$ | Spearman $\rho$ | Kendall $\tau$ |
| --- | --- | --- | --- |
| `myimmune_epimhc2_v10_clin` | 0.5233 | 0.4312 | 0.3123 |
| `myimmune_epimhc2_v10_seq` | 0.4838 | 0.4194 | 0.3058 |

On this dataset `_clin` is the highest of the 13 methods scored in this repository on both
Pearson and Spearman; the next highest are `progen/small` ($r = 0.4799$) and
`progen/medium` ($r = 0.4616$). `_seq` leads `progen/small` by $\Delta r = 0.004$, which at
N = 217 is within noise — the two should be read as comparable, not ranked.

Per-antibody scores, correlations and plots are under
`score/myimmune_epimhc2_v10_{seq,clin}/immunogenicity/Prihoda2021_mAb_immunogenicity/`.

### Access

This is a client for a hosted service, not a local model: it requires no PyTorch, CUDA, or
model checkpoints. The endpoint is access-controlled and no default URL is published. To
reproduce these numbers, request an endpoint and API key at
[myimmune.ai](https://myimmune.ai), then set:

```bash
export MYIMMUNE_EPIMHC2_URL=...   # issued with your access
export MYIMMUNE_API_KEY=...
```

```python
from myimmune_epimhc2_score import epimhc2_score

# Sequence-intrinsic score (IEDB-27 x OAS):
score_seq = epimhc2_score(heavy_seq="EVQL...", light_seq="DIQM...", mode="seq")

# Clinical pharmacology-adjusted ADA risk:
score_clin = epimhc2_score(heavy_seq="EVQL...", light_seq="DIQM...", mode="clin")
```

### Benchmark runner

```bash
python models/scoring_myimmune_epimhc2.py \
    data/immunogenicity/marks2021humanization_immunogenicity.csv --mode clin
```
