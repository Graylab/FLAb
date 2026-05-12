import pandas as pd
from scipy.stats import spearmanr
import os
import argparse

from sklearn.model_selection import train_test_split

from ld_score import ld_score
from ft_esm2_score import ft_esm2_score


# Define a mapping from method names to actual scoring functions
METHODS = {
    'ft_esm2_150M_score': ft_esm2_score
}

def score(method, df):
    """
    input: df with columns: heavy,
                            light,
                            fitness
    output: pearson correlation and pvalue
    """

    df = pd.read_csv(df)
    df['label'] = df['fitness'].astype(float)
    ### split dataset into train, val, test ###
    # Bin the continuous labels into discrete categories for stratification
    df['label_bin'] = pd.qcut(df['label'], q=10, duplicates='drop')  # 10 quantile bins
    train_df, temp_df = train_test_split(df,
                                        test_size=0.2,
                                        stratify=df['label_bin'],
                                        random_state=42) # First split: train (80%) and temp (20%)
    test_df, valid_df = train_test_split(temp_df,
                                         test_size=0.5,
                                         stratify=temp_df['label_bin'],
                                         random_state=42) # Second split: test (10%) and valid (10%) from temp (20%)
                        
    for subset in [train_df, test_df, valid_df]:
        subset.drop(columns='label_bin', inplace=True)
        subset.reset_index(drop=True, inplace=True)

    ### CALCULATE PERPLEXITIES ###
    if 'light' not in test_df.columns:
        # for dataframe with heavy only (Nbs)
        test_df['heavy_ld'] = test_df['heavy'].apply(lambda seq: ld_score(seq))
        
        train_df['sequence'] = train_df['heavy']
        valid_df['sequence'] = valid_df['heavy']
        test_df['sequence'] = test_df['heavy']
        test_df['label_prediction'] = method(train_df, valid_df, test_df, 'esm2_t30_150M_UR50D')
    else:
        # for dataframe with heavy,light (Fvs)
        test_df['heavy_ld'] = test_df['heavy'].apply(lambda seq: ld_score(seq))
        test_df['light_ld'] = test_df['light'].apply(lambda seq: ld_score(seq))

        train_df['sequence'] = train_df['heavy'] + 'GGGGSGGGGSGGGGS' + train_df['light']
        valid_df['sequence'] = valid_df['heavy'] + 'GGGGSGGGGSGGGGS' + valid_df['light']
        test_df['sequence'] = test_df['heavy'] + 'GGGGSGGGGSGGGGS' + test_df['light']
        test_df['label_prediction'] = method(train_df, valid_df, test_df, 'esm2_t30_150M_UR50D')

    # Calculate average perplexity
    test_df['average_ld'] = (test_df['heavy_ld'] + test_df.get('light_ld', 0)) / 2

    # Compute Spearman's correlation
    corr_ppl, p_value_ppl = spearmanr(test_df['label_prediction'], test_df['fitness'])
    print('corr to fitness', corr_ppl, p_value_ppl)
    corr_ld, p_value_ld = spearmanr(test_df['label_prediction'], test_df['average_ld'])
    print('corr to gl', corr_ld, p_value_ld)

    return corr_ppl, p_value_ppl, corr_ld, p_value_ld

def process_file(file_path, method_name):
    method = METHODS[method_name]  # Get the actual method function from the name

    # Calculate score and p-value using the specified method
    corr_ppl, p_value_ppl, corr_ld, p_value_ld = score(method, file_path)

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
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'score_ft', method_name)
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
