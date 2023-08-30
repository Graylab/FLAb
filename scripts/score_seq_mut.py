# imports
import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from models import iglm_score,antiberty_score,progen_score
from extra import dir_create

"""
Input: relative path to csv file (columns: heavy,light,fitness)

Output:
    csv_1 - perplexity scores
    csv_2 - correlations
    pdf - perplexity plots
"""

def _get_args():
    """Gets command line arguments"""

    desc = ("""Script for scoring antibody sequences""")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("csv_path",
                      type=str,
                      help="csv file with heavy and light chain sequences, and fitness metric.")

    parser.add_argument("score_method",
                      type=str,
                      help="model for scoring (ex: iglm).")

    parser.add_argument("progen_model",
                      type=str,
                      default=None,
                      nargs='?',
                      help="specific progen model (ex: small).")
    
    parser.add_argument("--use_gpu",
                      action='store_true',
                      help="specify whether to use gpu (default is cpu)")

    #parser.add_argument("device",
    #                  default='cpu',
    #                  nargs='?',
    #                  help="specify whether using cpu (default) or gpu (cuda:0)")
    return parser.parse_args()

def _cli():
    args = _get_args()

    csv_path = args.csv_path
    score_method = args.score_method
    progen_model = args.progen_model
    device = args.device

    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.use_gpu else 'cpu'
    device = torch.device(device_type)

    # CREATE DIRECTORY PATH

    # split csv_path data/fitness/filename into variables
    dir_name, filename = os.path.split(csv_path)
    data_dir, fitness_dir = os.path.split(dir_name)

    # remove file extension from filename
    name_only, extension = os.path.splitext(filename)

    score_dir = 'score'
    # create score/
    dir_create(score_dir)

    # create score/model/
    dir_create(score_dir, score_method)

    if score_method=='progen':
        # create score/model/progen_size/
        dir_create(score_dir, score_method, progen_model)

        # create score/model/progen_model/fitness/
        dir_create(score_dir, score_method, progen_model, fitness_dir)

        # create score/model/progen_model/fitness/csv/
        dir_create(score_dir, score_method, progen_model, fitness_dir, name_only)
    
    else:
        # create score/model/fitness/
        dir_create(score_dir, score_method, fitness_dir)

        # create score/model/fitness/csv/
        dir_create(score_dir, score_method, fitness_dir, name_only)

    df = pd.read_csv(csv_path)

    if score_method=='progen':
        # change to the 'score/score_method/fitness/name/' directory
        output_dir = os.path.join('.', score_dir, score_method, progen_model, fitness_dir, name_only)
        os.chdir(output_dir)
    else:
        # change to the 'score/score_method/fitness/name/' directory
        output_dir = os.path.join('.', score_dir, score_method, fitness_dir, name_only)
        os.chdir(output_dir)

    # CALCULATE PERPLEXITY
    if score_method == 'iglm':
        df = iglm_score(df)

    elif score_method == 'antiberty':
        df = antiberty_score(df)
    
    elif score_method == 'progen':
        df = progen_score(df, progen_model, device)

    # PLOT PERPLEXITY

    # Determine x axis
    if fitness_dir == 'binding':
        fitness_metric = 'negative log Kd'
        df[fitness_metric] = -np.log(df['fitness'])

    elif fitness_dir == 'immunogenicity':
        fitness_metric = 'immunogenic response (%)'
        df[fitness_metric] = df['fitness']

    elif fitness_dir == 'tm':
        fitness_metric = 'Melting temperature (°C)'
        df[fitness_metric] = df['fitness']

    elif fitness_dir == 'expression':
        fitness_metric = 'negative log expression'
        df[fitness_metric] = -np.log(df['fitness'])

    elif fitness_dir == 'aggregation':
        fitness_metric = 'aggregation metric'
        df[fitness_metric] = df['fitness']

    elif fitness_dir == 'polyreactivity':
        fitness_metric = 'polyreactivity metric'
        df[fitness_metric] = df['fitness']

    # df['is_first_row'] = df.index == 0
    plt.scatter(df[fitness_metric], df['average_perplexity'])
    # plt.scatter(df[fitness_metric][0], df['average_perplexity'][0], c=['orange'])
    # plt.legend(handles=[plt.Line2D([], [], marker='o', color='orange', label='wildtype'),
    #                     plt.Line2D([], [], marker='o', color='blue', label='mutants')])
    plt.xlabel(fitness_metric)
    plt.ylabel('average perplexity')
    plt.title(name_only)

    plt.savefig(f"{name_only}_plot.png")
    plt.show()

    # SAVE PERPLEXITY

    df.to_csv(f"{name_only}_ppl.csv", index=False)

    # CALCULATE CORRELATION

    name_list, correlation_list, p_list = ['pearson', 'spearman', 'kendal tau'], [], []

    # Pearson: linear relation between 2 variables
    pearson_corr, pearson_p_value = pearsonr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(pearson_corr)
    p_list.append(pearson_p_value)

    # Spearman's rank measures monotonic relationship between 2 variables
    spearman_corr, spearman_p_value = spearmanr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(spearman_corr)
    p_list.append(spearman_p_value)

    # Kendall's tau measures ordinal relationship between 2 variables
    kendall_corr, kendall_p_value = kendalltau(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(kendall_corr)
    p_list.append(kendall_p_value)

    df_corr = pd.DataFrame({'correlation_name': name_list, 'value': correlation_list, 'p-value': p_list})

    # SAVE CORRELATION

    df_corr.to_csv(f"{name_only}_corr.csv", index=False)


if __name__=='__main__':
    _cli()

