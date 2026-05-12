import pandas as pd
from scipy.stats import spearmanr
import os
import argparse

from ld_score import ld_score
from biopython_score import bp_isoelectric_point

# Define a mapping from method names to actual scoring functions
METHODS = {
    'bp_isoelectric_point': bp_isoelectric_point
}

def score(method, df):
    """
    input: df with columns: heavy,
                            light,
                            fitness
    output: pearson correlation and pvalue
    """

    file_path = df
    df = pd.read_csv(df)

    # Validate required columns
    if 'heavy' not in df.columns:
        print(f"Warning: No 'heavy' column found. Skipping.")
        return None, None, None, None
    if 'fitness' not in df.columns:
        print(f"Warning: No 'fitness' column found. Skipping.")
        return None, None, None, None

    # Drop rows with NaN in sequence or fitness columns
    drop_cols = ['heavy', 'fitness'] + (['light'] if 'light' in df.columns else [])
    df = df.dropna(subset=drop_cols)

    if len(df) < 2:
        print(f"Warning: Fewer than 2 valid rows after dropping NaN. Skipping.")
        return None, None, None, None

    ### CALCULATE PERPLEXITIES ###
    try:
        if 'light' not in df.columns:
            # for dataframe with heavy only (Nbs)
            df['heavy_ld'] = df['heavy'].apply(lambda seq: ld_score(seq))
            df['heavy_perplexity'] = df['heavy'].apply(lambda seq: method(seq))
        else:
            # for dataframe with heavy,light (Fvs)
            df['heavy_ld'] = df['heavy'].apply(lambda seq: ld_score(seq))
            df['light_ld'] = df['light'].apply(lambda seq: ld_score(seq))
            df['heavy_perplexity'] = df['heavy'].apply(lambda seq: method(seq))
            df['light_perplexity'] = df['light'].apply(lambda seq: method(seq))
    except KeyError as e:
        print(f"Warning: Non-standard amino acid {e} in {file_path}. Skipping.")
        skipped_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'skipped_nonstandard_aa.txt')
        with open(skipped_log, 'a') as f:
            f.write(f"{file_path}\n")
        return None, None, None, None

    # Calculate average perplexity
    df['average_perplexity'] = (df['heavy_perplexity'] + df.get('light_perplexity', 0)) / 2
    df['average_ld'] = (df['heavy_ld'] + df.get('light_ld', 0)) / 2

    # Compute Spearman's correlation
    corr_ppl, p_value_ppl = spearmanr(df['average_perplexity'], df['fitness'])
    corr_ld, p_value_ld = spearmanr(df['average_perplexity'], df['average_ld'])

    return corr_ppl, p_value_ppl, corr_ld, p_value_ld

def process_file(file_path, method_name):
    method = METHODS[method_name]  # Get the actual method function from the name

    # Calculate score and p-value using the specified method
    corr_ppl, p_value_ppl, corr_ld, p_value_ld = score(method, file_path)
    if corr_ppl is None:
        print(f"Skipping {file_path} due to validation error.")
        return

    # Get the base name of the file
    folder_name = os.path.basename(os.path.dirname(file_path))
    csv_base = os.path.splitext(os.path.basename(file_path))[0]

    # Create a dictionary for the results
    results = {
        'folder': folder_name,
        'csv': csv_base,
        f'{method_name}': corr_ppl,
        f'{method_name}_pval': p_value_ppl,
        f'{method_name}_ld': corr_ld,
        f'{method_name}_ld_pval': p_value_ld
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame([results])

    # Save the DataFrame to a compressed CSV file
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'score', method_name)
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    output_csv_path = os.path.join(output_folder, f'{folder_name}_{csv_base}.gz')
    results_df.to_csv(output_csv_path, index=False, compression='gzip')
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Calculate Spearman correlation between input CSV fitness and scoring method.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file to be scored.")
    parser.add_argument("method", type=str, help="Scoring method to use.")

    # Parse arguments
    args = parser.parse_args()

    # Process the specified file
    process_file(args.file_path, args.method)
