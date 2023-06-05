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

    parser.add_argument("device",
                      default='cpu',
                      nargs='?',
                      help="specify whether using cpu (default) or gpu (cuda:0)")
    return parser.parse_args()

def _cli():
    args = _get_args()

    csv_path = args.csv_path
    score_method = args.score_method
    progen_model = args.progen_model
    device = args.device

    # CREATE DIRECTORY PATH

    # split csv_path data/fitness/filename into variables
    dir_name, filename = os.path.split(csv_path)
    data_dir, fitness_dir = os.path.split(dir_name)

    # remove file extension from filename
    name_only, extension = os.path.splitext(filename)

    # print filename without directory path and extension
    print(f'The filename is {name_only}')

    # print directory name without full path
    print(f'The fitness directory name is {fitness_dir}')

    score_dir = 'score'
    # create score directory if it does not exist
    if not os.path.exists(os.path.join('.', score_dir)):
        print(f'Directory "{score_dir}" does not exist, creating directory')
        os.mkdir(os.path.join('.', score_dir))
    else:
        print(f'Directory "{score_dir}" exists already')

    # check if the 'iglm' directory exists inside 'score/', create if it doesn't
    if not os.path.exists(os.path.join('.', score_dir, score_method)):
        print(f'Directory "{score_dir}/{score_method}" does not exist, creating directory')
        os.mkdir(os.path.join('.', score_dir, score_method))
    else:
        print(f'Directory "{score_dir}/{score_method}" exists already')

    if score_method=='progen':
        # if the model is progen, check if progen_model exists inside progen/, create if it doesn't
        if not os.path.exists(os.path.join('.', score_dir, score_method, progen_model)):
            print(f'Directory "{score_dir}/{score_method}/{progen_model}" does not exist, creating directory')
            os.mkdir(os.path.join('.', score_dir, score_method, progen_model))
        else:
            print(f'Directory "{score_dir}/{score_method}/{progen_model}" exists already')

        # check if the fitness diretcory exists inside of 'score/score_method/progen_model/, create if not
        if not os.path.exists(os.path.join('.', score_dir, score_method, progen_model, fitness_dir)):
            print(f'Directory "{score_dir}/{score_method}/{progen_model}/{fitness_dir}" does not exist, creating directory')
            os.mkdir(os.path.join('.', score_dir, score_method, progen_model, fitness_dir))
        else:
            print(f'Directory "{score_dir}/{score_method}/{progen_model}/{fitness_dir}" exists already')
        
        # check if the 'name_only' directory (ex: Hie2022_C143_Kd) exists inside 'score/score_method/progen_model/binding/', create if it doesn't
        if not os.path.exists(os.path.join('.', score_dir, score_method, progen_model, fitness_dir, name_only)):
            print(f'Directory "{score_dir}/{score_method}/{progen_model}/{fitness_dir}/{name_only}" does not exist, creating directory')
            os.mkdir(os.path.join('.', score_dir, score_method, progen_model, fitness_dir, name_only))
        else:
            print(f'Directory "{score_dir}/{score_method}/{progen_model}/{fitness_dir}/{name_only}" exists already')
    else:
        # check if the fitness directory (ex: binding) exists inside 'score/score_method/', create if it doesn't
        if not os.path.exists(os.path.join('.', score_dir, score_method, fitness_dir)):
            print(f'Directory "{score_dir}/{score_method}/{fitness_dir}" does not exist, creating directory')
            os.mkdir(os.path.join('.', score_dir, score_method, fitness_dir))
        else:
            print(f'Directory "{score_dir}/{score_method}/{fitness_dir}" exists already')

        # check if the 'name_only' directory (ex: Hie2022_C143_Kd) exists inside 'score/score_method/binding/', create if it doesn't
        if not os.path.exists(os.path.join('.', score_dir, score_method, fitness_dir, name_only)):
            print(f'Directory "{score_dir}/{score_method}/{fitness_dir}/{name_only}" does not exist, creating directory')
            os.mkdir(os.path.join('.', score_dir, score_method, fitness_dir, name_only))
        else:
            print(f'Directory "{score_dir}/{score_method}/{fitness_dir}/{name_only}" exists already')

    #device_type = 'cuda' if torch.cuda.is_available(
    #) and args.use_gpu else 'cpu'
    #device = torch.device(device_type)

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


    df['is_first_row'] = df.index == 0
    plt.scatter(df[fitness_metric], df['average_perplexity'], c=['orange' if row else 'blue' for row in df['is_first_row']])
    plt.scatter(df[fitness_metric][0], df['average_perplexity'][0], c=['orange'])
    plt.legend(handles=[plt.Line2D([], [], marker='o', color='orange', label='wildtype'),
                        plt.Line2D([], [], marker='o', color='blue', label='mutants')])
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
    spearman_corr, spearman_p_value = spearmanr(df[fitness_metric], df['heavy_perplexity'])
    correlation_list.append(spearman_corr)
    p_list.append(spearman_p_value)

    # Kendall's tau measures ordinal relationship between 2 variables
    kendall_corr, kendall_p_value = kendalltau(df[fitness_metric], df['heavy_perplexity'])
    correlation_list.append(kendall_corr)
    p_list.append(kendall_p_value)

    df_corr = pd.DataFrame({'correlation_name': name_list, 'value': correlation_list, 'p-value': p_list})

    # SAVE CORRELATION

    df_corr.to_csv(f"{name_only}_corr.csv", index=False)


if __name__=='__main__':
    _cli()

