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

    # CALCULATE PERPLEXITY

    sns.set()

    fig, axes = plt.subplots(5, 3, figsize=(15, 20), sharey=True)

    sns.boxplot(data=df, x='Type', y='average_perplexity', ax=axes[0,0])
    sns.boxplot(data=df, x='Original mAb Isotype or Format', y='average_perplexity', ax=axes[0,1])
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    sns.boxplot(data=df, x='Clinical Status', y='average_perplexity', ax=axes[0,2])

    sns.scatterplot(data=df, x='HEK Titer (mg/L)', y='average_perplexity', ax=axes[1,0])
    sns.scatterplot(data=df, x='Fab Tm by DSF (°C)', y='average_perplexity', ax=axes[1,1])
    sns.scatterplot(data=df, x='SGAC-SINS AS100 ((NH4)2SO4 mM)', y='average_perplexity', ax=axes[1,2])

    sns.scatterplot(data=df, x='HIC Retention Time (Min)a', y='average_perplexity', ax=axes[2,0])
    sns.scatterplot(data=df, x='SMAC Retention Time (Min)a', y='average_perplexity', ax=axes[2,1])
    sns.scatterplot(data=df, x='Slope for Accelerated Stability', y='average_perplexity', ax=axes[2,2])

    sns.scatterplot(data=df, x='Poly-Specificity Reagent (PSR) SMP Score (0-1)', y='average_perplexity', ax=axes[3,0])
    sns.scatterplot(data=df, x='Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average', y='average_perplexity', ax=axes[3,1])
    axes[3,1].set_xlabel('AC-SINS')
    sns.scatterplot(data=df, x='CIC Retention Time (Min)', y='average_perplexity', ax=axes[3,2])

    sns.scatterplot(data=df, x='CSI-BLI Delta Response (nm)', y='average_perplexity', ax=axes[4,0])
    sns.scatterplot(data=df, x='ELISA', y='average_perplexity', ax=axes[4,1])
    sns.scatterplot(data=df, x='BVP ELISA', y='average_perplexity', ax=axes[4,2])

    plt.suptitle(f'Perplexity scores of {score_method} versus 12 developability assays')

    plt.savefig(f"{name_only}_plot.png")
    plt.show()

    # SAVE PERPLEXITY

    df.to_csv(f"{name_only}_ppl.csv", index=False)

    # CALCULATE CORRELATION

    name_list = ['pearson', 'spearman', 'kendal tau']
    df_corr = pd.DataFrame({'correlation_name': name_list})
    
    assay_names = ['HEK Titer (mg/L)',
                   'Fab Tm by DSF (°C)',
                   'SGAC-SINS AS100 ((NH4)2SO4 mM)',
                   'HIC Retention Time (Min)a',
                   'SMAC Retention Time (Min)a',
                   'Slope for Accelerated Stability',
                   'Poly-Specificity Reagent (PSR) SMP Score (0-1)',
                   'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average',
                   'CIC Retention Time (Min)',
                   'CSI-BLI Delta Response (nm)',
                   'ELISA',
                   'BVP ELISA']
    
    for assay in assay_names:
        correlation_list, p_list = [], []

        # Pearson: linear relation between 2 variables
        pearson_corr, pearson_p_value = pearsonr(df[assay], df['average_perplexity'])
        correlation_list.append(pearson_corr)
        p_list.append(pearson_p_value)

        # Spearman's rank measures monotonic relationship between 2 variables
        spearman_corr, spearman_p_value = spearmanr(df[assay], df['average_perplexity'])
        correlation_list.append(spearman_corr)
        p_list.append(spearman_p_value)

        # Kendall's tau measures ordinal relationship between 2 variables
        kendall_corr, kendall_p_value = kendalltau(df[assay], df['average_perplexity'])
        correlation_list.append(kendall_corr)
        p_list.append(kendall_p_value)

        df_corr[assay] = correlation_list
        df_corr[f'{assay} p-value'] = p_list
    

    # SAVE CORRELATION

    df_corr.to_csv(f"{name_only}_corr.csv", index=False)


if __name__=='__main__':
    _cli()

