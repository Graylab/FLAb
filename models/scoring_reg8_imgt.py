import argparse
import gzip
import os

import pandas as pd
from scipy.stats import spearmanr

try:
    from ld_score import ld_score
except ImportError:
    ld_score = None
from reg8_imgt_score import reg8_bp_imgt_z_pair

METHODS = {
    "clss_reg8_imgt_z": reg8_bp_imgt_z_pair,
}


def score(method, file_path):
    """Score a FLAb CSV (heavy[, light], fitness) with zero-shot regional IMGT Z-score."""
    df = pd.read_csv(file_path)

    if "heavy" not in df.columns or "fitness" not in df.columns:
        print("Warning: Required 'heavy' or 'fitness' column missing. Skipping.")
        return None, None, None, None

    has_light = "light" in df.columns
    drop_cols = ["heavy", "fitness"] + (["light"] if has_light else [])
    df = df.dropna(subset=drop_cols)
    if len(df) < 2:
        print("Warning: Fewer than 2 valid rows after dropping NaN. Skipping.")
        return None, None, None, None

    try:
        if not has_light:
            df["average_perplexity"] = [method(h, None) for h in df["heavy"]]
            if ld_score is not None:
                df["heavy_ld"] = df["heavy"].apply(ld_score)
                df["average_ld"] = df["heavy_ld"]
        else:
            df["average_perplexity"] = [method(h, l) for h, l in zip(df["heavy"], df["light"])]
            if ld_score is not None:
                df["heavy_ld"] = df["heavy"].apply(ld_score)
                df["light_ld"] = df["light"].apply(ld_score)
                df["average_ld"] = 0.5 * (df["heavy_ld"] + df["light_ld"])
    except KeyError as exc:
        print(f"Warning: Non-standard amino acid {exc} in {file_path}. Skipping.")
        skipped_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "skipped_nonstandard_aa.txt"
        )
        with open(skipped_log, "a", encoding="utf-8") as fh:
            fh.write(f"{file_path}\n")
        return None, None, None, None

    corr_ppl, p_value_ppl = spearmanr(df["average_perplexity"], df["fitness"])
    if ld_score is not None:
        corr_ld, p_value_ld = spearmanr(df["average_perplexity"], df["average_ld"])
    else:
        corr_ld, p_value_ld = float("nan"), float("nan")
    return corr_ppl, p_value_ppl, corr_ld, p_value_ld


def process_file(file_path, method_name):
    method = METHODS[method_name]
    corr_ppl, p_value_ppl, corr_ld, p_value_ld = score(method, file_path)
    if corr_ppl is None:
        print(f"Skipping {file_path} due to validation error.")
        return

    folder_name = os.path.basename(os.path.dirname(file_path))
    csv_base = os.path.splitext(os.path.basename(file_path))[0]
    results_df = pd.DataFrame(
        [
            {
                "folder": folder_name,
                "csv": csv_base,
                f"{method_name}": corr_ppl,
                f"{method_name}_pval": p_value_ppl,
                f"{method_name}_ld": corr_ld,
                f"{method_name}_ld_pval": p_value_ld,
            }
        ]
    )

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "score", method_name)
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, f"{folder_name}_{csv_base}.gz")
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    with open(output_csv_path, "wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh:
            gz_fh.write(csv_bytes)
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Spearman correlation between input CSV fitness and regional IMGT Z-score."
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV file to be scored.")
    parser.add_argument("method", nargs="?", type=str, default="clss_reg8_imgt_z", help="Scoring method name.")
    args = parser.parse_args()
    process_file(args.file_path, args.method)
