import os
import numpy as np
from abnumber import Chain
from Bio import Align
import pandas as pd
import Levenshtein
import matplotlib.pyplot as plt
from anarci import run_anarci

# ---------------------------------------------------------------------------
# Cache: load precomputed LD scores on import to avoid redundant ANARCI calls
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ld_cache.csv.gz")
_ld_cache: dict = {}

if os.path.exists(_CACHE_PATH):
    try:
        _cache_df = pd.read_csv(_CACHE_PATH, compression="gzip")
        _ld_cache = dict(zip(_cache_df["sequence"], _cache_df["ld_score"]))
    except Exception:
        pass  # cache unreadable — fall back to live computation

def align_truncate(target, query):
    # Check if either input is None
    if target is None or query is None:
        return None, None
    aligner = Align.PairwiseAligner()
    # Modifications to penalize gap generation
    aligner.mode = 'local'
    aligner.open_gap_score = -10  # Higher penalty for opening gaps
    aligner.extend_gap_score = -0.5  # Lower penalty for extending gaps
    alignments = aligner.align(target, query)
    best_alignment = alignments[0]
    tas = best_alignment.aligned[0][0][0] # Target alignment start
    tae = best_alignment.aligned[0][-1][-1] # Target alignment end
    qas = best_alignment.aligned[1][0][0] # query alignment start
    qae = best_alignment.aligned[1][-1][-1] # query alignment end
    aligned_target = target[tas:tae]
    aligned_query = query[qas:qae]
    return aligned_target, aligned_query

def find_human_gl_v(aa_seq):
    try:
        chain = Chain(aa_seq, 'imgt')
        vgene = str(chain.find_human_germlines(1)[0][0])
        aln_v, aln_seq_v = align_truncate(vgene, aa_seq)
        aln_v_ld = Levenshtein.distance(aln_seq_v, aln_v)
        return pd.Series({
            'v_call': vgene,
            'aln_v': aln_v,
            'aln_query_v': aln_seq_v,
            'aln_v_ld': aln_v_ld
        })
    except Exception as e:
        return pd.Series({
            'v_call': None,
            'aln_v': None,
            'aln_query_v': None,
            'aln_v_ld': None
        })

def find_human_gl_j(aa_seq):
    try:
        chain = Chain(aa_seq, 'imgt')
        jgene = str(chain.find_human_germlines(1)[1][0])
        aln_j, aln_seq_j = align_truncate(jgene, aa_seq)
        aln_j_ld = Levenshtein.distance(aln_seq_j, aln_j)
        return pd.Series({
            'j_call': jgene,
            'aln_j': aln_j,
            'aln_query_j': aln_seq_j,
            'aln_j_ld': aln_j_ld
        })
    except Exception as e:
        return pd.Series({
            'j_call': None,
            'aln_j': None,
            'aln_query_j': None,
            'aln_j_ld': None
        })

def ld_score(seq):
    """
    Calculate LDv + LDj of a given sequence.

    Returns the cached value if available (populated by precompute_ld_cache.py),
    otherwise falls back to live ANARCI + Levenshtein computation.
    """
    if seq in _ld_cache:
        return _ld_cache[seq]

    ld_v = find_human_gl_v(seq).aln_v_ld
    ld_j = find_human_gl_j(seq).aln_j_ld

    # Handle NaN values
    if pd.isna(ld_v) and pd.isna(ld_j):
        return np.nan  # Both values are NaN
    elif pd.isna(ld_v):
        return ld_j  # Return the non-NaN value (ld_j)
    elif pd.isna(ld_j):
        return ld_v  # Return the non-NaN value (ld_v)
    else:
        return ld_v + ld_j  # Sum of both non-NaN values

