"""
scoring_myimmune_epimhc2.py
---------------------------
FLAb benchmark runner for MyImmune EpiMHC2 v10.0.0.
Calculates Spearman correlation between dataset fitness (%ADA) and MyImmune EpiMHC2 risk score.
"""

import os
import argparse
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from myimmune_epimhc2_score import epimhc2_score


def score(df_path: str, mode: str = "seq"):
    df = pd.read_csv(df_path)
    if "heavy" not in df.columns or "fitness" not in df.columns:
        print("Required columns heavy and fitness not found.")
        return None, None

    drop_cols = ["heavy", "fitness"] + (["light"] if "light" in df.columns else [])
    df = df.dropna(subset=drop_cols)

    scores = []
    print(f"Scoring {len(df)} antibodies with MyImmune EpiMHC2 (mode={mode})...")
    for _, row in df.iterrows():
        h = str(row["heavy"])
        l = str(row["light"]) if "light" in df.columns else ""
        s = epimhc2_score(h, l, mode=mode)
        scores.append(s)

    corr, pval = spearmanr(scores, df["fitness"])
    pr, pr_pval = pearsonr(scores, df["fitness"])
    print(f"Results: Spearman rho={corr:.4f} (p={pval:.4e}), Pearson r={pr:.4f} (p={pr_pval:.4e})")
    return corr, pval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score FLAb dataset with MyImmune EpiMHC2 hosted API")
    parser.add_argument("file_path", type=str, help="Path to input CSV (e.g. data/immunogenicity/marks2021humanization_immunogenicity.csv)")
    parser.add_argument("--mode", type=str, default="seq", choices=["seq", "clin"], help="Scoring mode: seq or clin")
    args = parser.parse_args()
    score(args.file_path, mode=args.mode)
